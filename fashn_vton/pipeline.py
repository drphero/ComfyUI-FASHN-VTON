"""TryOn Pipeline."""

import logging
import os
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import cv2
import numpy as np
import torch
from fashn_human_parser import CATEGORY_TO_BODY_COVERAGE, FashnHumanParser
from PIL import Image
from tqdm.auto import tqdm

import comfy.model_management

from .preprocessing import (
    BODY_COVERAGE_TO_FASHN_LABELS,
    FASHN_LABELS_TO_IDS,
    AspectPreserveResize,
    ResizePad,
    create_clothing_agnostic_image,
    create_garment_image,
)
from .pose_keypoints import convert_pose_keypoints_to_dwpose
from .providers import DWPoseProvider, FashnHumanParserProvider
from .tryon_mmdit import TryOnModel
from .utils import (
    get_dummy_dw_keypoints,
    get_rf_schedule,
    load_checkpoint,
    normalize_uint8_to_neg1_1,
    numpy_to_torch,
    setup_logger,
    tensor_to_pil,
)


@dataclass
class PipelineOutput:
    """Pipeline output container."""

    images: List[Image.Image]


class TryOnPipeline:
    """
    TryOn inference pipeline.

    Args:
        weights_dir: Directory containing model weights (model.safetensors, dwpose/)
        logger: Optional logger instance

    Example:
        pipeline = TryOnPipeline(weights_dir="./weights")
        result = pipeline(person_image, garment_image, category="tops")
    """

    CATEGORY_TO_LABEL = {"tops": 1, "bottoms": 2, "one-pieces": 3}
    POSE_SOURCES = (
        "auto",
        "internal_dwpose",
        "external_pose_keypoints",
    )
    PARSER_BACKENDS = ("fashn_human_parser", "external_fashn_labelmap")
    MAX_FASHN_LABEL_ID = max(FASHN_LABELS_TO_IDS.values())

    def __init__(
        self,
        weights_dir: str,
        logger: Optional[logging.Logger] = None,
    ):
        self.weights_dir = os.path.abspath(weights_dir)
        self.logger = logger or setup_logger("TryOnPipeline", level=logging.INFO)

        # Setup device
        self.offload_device = torch.device("cpu")
        self.device = comfy.model_management.get_torch_device()

        # Setup inference dtype
        self.inference_dtype = torch.float32
        if comfy.model_management.should_use_bf16():
            self.inference_dtype = torch.bfloat16
        elif comfy.model_management.should_use_fp16():
            self.inference_dtype = torch.float16
        self.logger.info(f"Using dtype: {self.inference_dtype}")

        # Validate weights exist
        self._validate_weights()

        # Load models
        self._setup_tryon_model()
        self._setup_hp_model()

        # Lazily loaded providers
        self.pose_provider: Optional[DWPoseProvider] = None

        # Setup transforms (derived from model input shape)
        h, w = self.tryon_model.input_shape
        max_dim = max(h, w)
        self.pre_resize = AspectPreserveResize(target_size=(max_dim, max_dim), mode="fit", backend="pil")
        self.resize_pad_fn = ResizePad((w, h), backend="opencv")

    def _validate_weights(self):
        """Check that required weight files exist."""
        tryon_path = os.path.join(self.weights_dir, "model.safetensors")
        dwpose_dir = os.path.join(self.weights_dir, "dwpose")
        yolox_path = os.path.join(dwpose_dir, "yolox_l.onnx")
        dwpose_path = os.path.join(dwpose_dir, "dw-ll_ucoco_384.onnx")

        missing = []
        if not os.path.exists(tryon_path):
            missing.append(tryon_path)
        if not os.path.exists(yolox_path):
            missing.append(yolox_path)
        if not os.path.exists(dwpose_path):
            missing.append(dwpose_path)

        if missing:
            raise FileNotFoundError(
                "Missing model weights:\n"
                + "\n".join(f"  - {p}" for p in missing)
                + f"\n\nPlease run:\n  python scripts/download_weights.py --weights-dir {self.weights_dir}"
            )

    def _setup_tryon_model(self):
        """Load the TryOn model."""
        model_path = os.path.join(self.weights_dir, "model.safetensors")
        self.logger.info(f"Loading TryOnModel from {model_path}")

        self.tryon_model = TryOnModel()
        state_dict = load_checkpoint(model_path, device="cpu")
        self.tryon_model.load_state_dict(state_dict)
        self.tryon_model.to(self.offload_device, dtype=self.inference_dtype).eval()

        self.logger.info("TryOnModel loaded")

    def _get_pose_provider(self) -> DWPoseProvider:
        """Create DWPose provider lazily so external-pose workflows can skip it."""
        if self.pose_provider is None:
            dwpose_dir = os.path.join(self.weights_dir, "dwpose")
            dwpose_device = f"cuda:{self.device.index or 0}" if self.device.type == "cuda" else "cpu"
            self.pose_provider = DWPoseProvider(checkpoints_dir=dwpose_dir, device=dwpose_device)
            # Backward-compatible attribute used by existing unload logic.
            self.pose_model = self.pose_provider.detector

        return self.pose_provider

    def _setup_hp_model(self):
        """Load human parsing model."""
        self.logger.info("Loading FashnHumanParser")

        self.hp_model = FashnHumanParser(device="cpu")

        if hasattr(self.hp_model, "model"):
            self.hp_model.model.to(self.offload_device)

        self.hp_model.device = self.offload_device
        self.parser_provider = FashnHumanParserProvider(self.hp_model)
        self.logger.info(f"FashnHumanParser loaded on {self.offload_device}")

    def _normalize_external_pose_keypoints(self, pose_keypoints: Optional[dict]) -> Optional[dict]:
        """Convert arbitrary keypoint payload to canonical DWPose-style dict."""
        if pose_keypoints is None:
            return None

        converted = convert_pose_keypoints_to_dwpose(pose_keypoints, single_person=True)
        if converted is None:
            self.logger.warning("External pose keypoints payload is unsupported. Falling back to other pose sources.")
            return None
        return converted

    def _normalize_external_segmentation(
        self, seg_img: Optional[Image.Image], image_name: str
    ) -> Optional[np.ndarray]:
        """Convert external segmentation image to uint8 label-id map."""
        if seg_img is None:
            return None

        seg_np = np.array(seg_img)

        if seg_np.ndim == 3:
            if seg_np.shape[2] == 1:
                seg_np = seg_np[..., 0]
            else:
                if not (
                    np.array_equal(seg_np[..., 0], seg_np[..., 1])
                    and np.array_equal(seg_np[..., 0], seg_np[..., 2])
                ):
                    self.logger.warning(
                        "%s segmentation has non-identical RGB channels. Using channel 0 as label ids.",
                        image_name,
                    )
                seg_np = seg_np[..., 0]
        elif seg_np.ndim != 2:
            self.logger.warning(
                "%s segmentation has invalid rank %s. Falling back to internal parser.",
                image_name,
                seg_np.ndim,
            )
            return None

        seg_np = seg_np.astype(np.uint8)

        if np.any(seg_np > self.MAX_FASHN_LABEL_ID):
            self.logger.warning(
                "%s segmentation contains label ids > %s. Falling back to internal parser.",
                image_name,
                self.MAX_FASHN_LABEL_ID,
            )
            return None

        return seg_np

    def _detect_internal_pose_image(self, image_np: np.ndarray) -> np.ndarray:
        """Detect and render internal DWPose grayscale map."""
        pose_provider = self._get_pose_provider()
        pose = pose_provider.detect(image_np)
        return pose_provider.render_grayscale(pose, image_np.shape[0], image_np.shape[1])

    def _render_pose_keypoints(self, pose_keypoints: dict, height: int, width: int) -> Optional[np.ndarray]:
        """Render normalized DWPose-style keypoints into grayscale pose map."""
        try:
            pose_provider = self._get_pose_provider()
            return pose_provider.render_grayscale(pose_keypoints, height, width)
        except Exception as exc:
            self.logger.warning("Failed to render external keypoints (%s). Falling back to other pose sources.", exc)
            return None

    def _render_dummy_pose_image(self, height: int, width: int) -> np.ndarray:
        """Render dummy pose map used for flat-lay garment images."""
        pose_provider = self._get_pose_provider()
        dummy_pose = get_dummy_dw_keypoints()
        return pose_provider.render_grayscale(dummy_pose, height, width)

    def _get_pose_images(
        self,
        *,
        person_image_np: np.ndarray,
        garment_image_np: np.ndarray,
        garment_photo_type: str,
        pose_source: str,
        person_pose_keypoints: Optional[dict],
        garment_pose_keypoints: Optional[dict],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Resolve pose maps from external inputs or internal DWPose."""
        if pose_source not in self.POSE_SOURCES:
            self.logger.warning("Unknown pose_source=%s. Using auto.", pose_source)
            pose_source = "auto"

        use_external_keypoints = pose_source in ("auto", "external_pose_keypoints")

        person_pose_img = None
        garment_pose_img = None
        person_pose_source = "internal_dwpose"
        garment_pose_source = "internal_dwpose"

        if use_external_keypoints:
            person_keypoints = self._normalize_external_pose_keypoints(person_pose_keypoints)
            garment_keypoints = self._normalize_external_pose_keypoints(garment_pose_keypoints)

            if person_keypoints is not None:
                person_pose_img = self._render_pose_keypoints(
                    person_keypoints,
                    person_image_np.shape[0],
                    person_image_np.shape[1],
                )
                if person_pose_img is not None:
                    person_pose_source = "external_keypoints"

            if garment_keypoints is not None:
                garment_pose_img = self._render_pose_keypoints(
                    garment_keypoints,
                    garment_image_np.shape[0],
                    garment_image_np.shape[1],
                )
                if garment_pose_img is not None:
                    garment_pose_source = "external_keypoints"

        if person_pose_img is None:
            person_pose_img = self._detect_internal_pose_image(person_image_np)
            if use_external_keypoints:
                person_pose_source = "internal_fallback"

        if garment_pose_img is None:
            if garment_photo_type == "flat-lay":
                garment_pose_img = self._render_dummy_pose_image(garment_image_np.shape[0], garment_image_np.shape[1])
                garment_pose_source = "dummy_flat_lay"
            else:
                garment_pose_img = self._detect_internal_pose_image(garment_image_np)
                if use_external_keypoints:
                    garment_pose_source = "internal_fallback"

        self.logger.info(
            "pose_source_resolved person=%s garment=%s",
            person_pose_source,
            garment_pose_source,
        )

        return person_pose_img, garment_pose_img

    def _get_segmentation_maps(
        self,
        *,
        person_image_np: np.ndarray,
        garment_image_np: np.ndarray,
        parser_backend: str,
        person_segmentation_image: Optional[Image.Image],
        garment_segmentation_image: Optional[Image.Image],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Resolve segmentation maps from selected backend with fallback."""
        if parser_backend not in self.PARSER_BACKENDS:
            self.logger.warning("Unknown parser_backend=%s. Using fashn_human_parser.", parser_backend)
            parser_backend = "fashn_human_parser"

        person_source = "internal"
        garment_source = "internal"

        if parser_backend == "external_fashn_labelmap":
            person_seg_pred = self._normalize_external_segmentation(person_segmentation_image, "person")
            garment_seg_pred = self._normalize_external_segmentation(garment_segmentation_image, "garment")

            if person_seg_pred is None:
                person_seg_pred = self.parser_provider.predict(person_image_np)
                person_source = "internal_fallback"
            else:
                person_source = "external"

            if garment_seg_pred is None:
                garment_seg_pred = self.parser_provider.predict(garment_image_np)
                garment_source = "internal_fallback"
            else:
                garment_source = "external"
        else:
            person_seg_pred = self.parser_provider.predict(person_image_np)
            garment_seg_pred = self.parser_provider.predict(garment_image_np)

        self.logger.info(
            "parser_backend_resolved person=%s garment=%s",
            person_source,
            garment_source,
        )

        return person_seg_pred, garment_seg_pred

    @torch.inference_mode()
    def _sample(
        self,
        *,
        ca_images: torch.Tensor,
        garment_images: torch.Tensor,
        person_poses: torch.Tensor,
        garment_poses: torch.Tensor,
        garment_categories: torch.Tensor,
        num_timesteps: int = 30,
        time_shift_mu: float = 1.5,
        guidance_scale: float = 1.5,
        skip_cfg_last_n_steps: int = 1,
        use_tqdm: bool = True,
        callback: Optional[callable] = None,
    ) -> List[Image.Image]:
        """Euler sampling with CFG."""
        device, dtype = ca_images.device, ca_images.dtype
        batch_size = ca_images.shape[0]

        # Init noisy images
        c, h, w = self.tryon_model.channels_in, *self.tryon_model.input_shape
        images = torch.randn((batch_size, c, h, w), dtype=dtype, device=device)

        # Time schedule (from 0 -> 1)
        timesteps = get_rf_schedule(num_steps=num_timesteps, mu=time_shift_mu)

        model_kwargs = {
            "person_poses": person_poses,
            "garment_poses": garment_poses,
            "ca_images": ca_images,
            "garment_images": garment_images,
            "garment_categories": garment_categories,
        }

        # Euler sampling loop
        total_steps = len(timesteps) - 1
        for step_idx, (t_curr, t_prev) in enumerate(
            tqdm(
                zip(timesteps[:-1], timesteps[1:]),
                desc="Sampling",
                total=total_steps,
                disable=not use_tqdm,
            )
        ):
            if callback:
                callback(step_idx, total_steps)
            dt = t_prev - t_curr
            t_vec = torch.full((batch_size,), t_curr, dtype=dtype, device=device)

            pred = self.tryon_model.forward_for_cfg(images, t_vec, **model_kwargs)
            v_c, v_u = pred["v_c"], pred["v_u"]

            # Skip CFG at final steps to prevent color saturation
            if skip_cfg_last_n_steps > 0 and step_idx >= num_timesteps - skip_cfg_last_n_steps:
                v_guided = v_c
            else:
                v_guided = v_u + guidance_scale * (v_c - v_u)

            images = images + dt * v_guided

        images = images.to(dtype=torch.float).clamp_(-1.0, 1.0)
        return [tensor_to_pil(img, unnormalize=True) for img in images]

    @torch.inference_mode()
    def __call__(
        self,
        person_image: Image.Image,
        garment_image: Image.Image,
        category: Literal["tops", "bottoms", "one-pieces"],
        garment_photo_type: Literal["model", "flat-lay"] = "model",
        num_samples: int = 1,
        num_timesteps: int = 30,
        guidance_scale: float = 1.5,
        skip_cfg_last_n_steps: int = 1,
        seed: int = 42,
        segmentation_free: bool = True,
        pose_source: Literal["auto", "internal_dwpose", "external_pose_keypoints"] = "auto",
        person_pose_keypoints: Optional[dict] = None,
        garment_pose_keypoints: Optional[dict] = None,
        parser_backend: Literal["fashn_human_parser", "external_fashn_labelmap"] = "fashn_human_parser",
        person_segmentation_image: Optional[Image.Image] = None,
        garment_segmentation_image: Optional[Image.Image] = None,
        callback: Optional[callable] = None,
    ) -> PipelineOutput:
        """
        Run virtual try-on inference.

        Args:
            person_image: RGB image of the person to dress.
            garment_image: RGB image of the garment (model photo or flat-lay).
            category: Garment category - "tops", "bottoms", or "one-pieces".
            garment_photo_type: "model" if garment is worn by a person,
                "flat-lay" for product shots on plain backgrounds.
            num_samples: Number of output images to generate (1-4).
            num_timesteps: Diffusion sampling steps. Higher = better quality, slower.
            guidance_scale: Classifier-free guidance strength.
            skip_cfg_last_n_steps: Skip CFG for final N steps to prevent color saturation.
            seed: Random seed for reproducibility.
            segmentation_free: If True, generate without masking the person image.
            pose_source: Pose source routing strategy.
            person_pose_keypoints: Optional external keypoints payload for person.
            garment_pose_keypoints: Optional external keypoints payload for garment.
            parser_backend: Parser backend strategy.
            person_segmentation_image: Optional external person segmentation map.
            garment_segmentation_image: Optional external garment segmentation map.

        Returns:
            PipelineOutput with `images` list containing generated PIL Images.
        """

        # Set seed
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        # Pre-resize for detection/parsing quality
        person_image = self.pre_resize(person_image, allow_upsampling=False)
        garment_image = self.pre_resize(garment_image, allow_upsampling=False)
        nearest_resample = Image.Resampling if hasattr(Image, "Resampling") else Image

        if person_segmentation_image is not None:
            person_segmentation_image = self.pre_resize(
                person_segmentation_image, allow_upsampling=False, interpolation=nearest_resample.NEAREST
            )
        if garment_segmentation_image is not None:
            garment_segmentation_image = self.pre_resize(
                garment_segmentation_image, allow_upsampling=False, interpolation=nearest_resample.NEAREST
            )

        person_image_np = np.array(person_image)
        garment_image_np = np.array(garment_image)

        person_pose_img, garment_pose_img = self._get_pose_images(
            person_image_np=person_image_np,
            garment_image_np=garment_image_np,
            garment_photo_type=garment_photo_type,
            pose_source=pose_source,
            person_pose_keypoints=person_pose_keypoints,
            garment_pose_keypoints=garment_pose_keypoints,
        )

        person_seg_pred, garment_seg_pred = self._get_segmentation_maps(
            person_image_np=person_image_np,
            garment_image_np=garment_image_np,
            parser_backend=parser_backend,
            person_segmentation_image=person_segmentation_image,
            garment_segmentation_image=garment_segmentation_image,
        )

        # Get labels to segment based on category
        body_coverage = CATEGORY_TO_BODY_COVERAGE.get(category)
        labels_to_segment = BODY_COVERAGE_TO_FASHN_LABELS.get(body_coverage)
        labels_to_segment_indices = [FASHN_LABELS_TO_IDS[label] for label in labels_to_segment]

        # Create clothing-agnostic and garment images
        ca_image = create_clothing_agnostic_image(
            img_np=person_image_np.copy(),
            seg_pred=person_seg_pred.copy(),
            labels_to_segment_indices=labels_to_segment_indices.copy(),
            body_coverage=body_coverage,
            disable_masking=segmentation_free,
            logger=self.logger,
        )

        garment_image_processed = create_garment_image(
            img_np=garment_image_np,
            seg_pred=garment_seg_pred,
            labels_to_segment_indices=labels_to_segment_indices.copy(),
            disable_masking=garment_photo_type == "flat-lay",
        )

        # Resize/pad for model input
        ca_image = self.resize_pad_fn(ca_image, mem_padding=True)
        garment_image_processed = self.resize_pad_fn(garment_image_processed)
        person_pose_img = self.resize_pad_fn(person_pose_img, interpolation=cv2.INTER_NEAREST_EXACT)
        garment_pose_img = self.resize_pad_fn(garment_pose_img, interpolation=cv2.INTER_NEAREST_EXACT)

        # Prepare tensors
        def prepare_tensor(img: np.ndarray) -> torch.Tensor:
            t = numpy_to_torch(img).unsqueeze(0)
            t = normalize_uint8_to_neg1_1(t)
            t = t.to(self.device).repeat(num_samples, 1, 1, 1)
            return t

        ca_tensor = prepare_tensor(ca_image)
        garment_tensor = prepare_tensor(garment_image_processed)
        person_pose_tensor = prepare_tensor(person_pose_img)
        garment_pose_tensor = prepare_tensor(garment_pose_img)

        garment_categories = (
            torch.tensor(self.CATEGORY_TO_LABEL[category]).unsqueeze(0).repeat(num_samples).to(self.device)
        )

        # Cast to inference dtype
        ca_tensor = ca_tensor.to(dtype=self.inference_dtype)
        garment_tensor = garment_tensor.to(dtype=self.inference_dtype)
        person_pose_tensor = person_pose_tensor.to(dtype=self.inference_dtype)
        garment_pose_tensor = garment_pose_tensor.to(dtype=self.inference_dtype)

        # Run sampling
        self.logger.info(f"Running inference with {num_timesteps} timesteps...")
        images = self._sample(
            ca_images=ca_tensor,
            garment_images=garment_tensor,
            person_poses=person_pose_tensor,
            garment_poses=garment_pose_tensor,
            garment_categories=garment_categories,
            num_timesteps=num_timesteps,
            guidance_scale=guidance_scale,
            skip_cfg_last_n_steps=skip_cfg_last_n_steps,
            callback=callback,
        )

        # Unpad outputs
        images = [self.resize_pad_fn.unpad(img) for img in images]

        self.logger.info(f"Generated {len(images)} images")

        return PipelineOutput(images=images)
