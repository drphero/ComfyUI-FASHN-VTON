import os

import numpy as np
import torch
from PIL import Image

import comfy.model_management
import comfy.utils
import folder_paths

from .fashn_vton import TryOnPipeline
from .fashn_vton.pose_keypoints import convert_pose_keypoints_to_dwpose
from .fashn_vton.preprocessing import FASHN_LABELS_TO_IDS

MODEL_LIST = ["fashn-ai/fashn-vton-1.5"]
MAX_FASHN_LABEL_ID = max(FASHN_LABELS_TO_IDS.values())
DEFAULT_CATEGORY_LABEL_ID = {
    "tops": FASHN_LABELS_TO_IDS["top"],
    "bottoms": FASHN_LABELS_TO_IDS["pants"],
    "one-pieces": FASHN_LABELS_TO_IDS["dress"],
}


def _tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    """Convert a ComfyUI IMAGE tensor (B,H,W,C in [0,1]) to PIL for pipeline input."""
    img = image_tensor[0].detach().cpu().numpy()
    img = np.clip(img, 0.0, 1.0)

    if img.ndim == 2:
        return Image.fromarray((img * 255.0).astype(np.uint8), mode="L")

    channels = img.shape[2]

    if channels == 1:
        return Image.fromarray((img[..., 0] * 255.0).astype(np.uint8), mode="L")

    if channels == 2:
        return Image.fromarray((img[..., 0] * 255.0).astype(np.uint8), mode="L")

    if channels > 3:
        img = img[..., :3]

    return Image.fromarray((img * 255.0).astype(np.uint8))


class FashnVtonLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (MODEL_LIST, {"default": "fashn-ai/fashn-vton-1.5"}),
            }
        }

    RETURN_TYPES = ("FASHN_VTON_PIPELINE",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "load_pipeline"
    CATEGORY = "FashnAI"

    def load_pipeline(self, model):
        weights_name = "fashn-vton"
        base_weights_dir = os.path.join(folder_paths.models_dir, weights_name)

        os.makedirs(base_weights_dir, exist_ok=True)

        from huggingface_hub import hf_hub_download

        # Download TryOnModel
        tryon_path = os.path.join(base_weights_dir, "model.safetensors")
        if not os.path.exists(tryon_path):
            print(f"FashnVTON: Downloading TryOnModel weights to {tryon_path}...")
            hf_hub_download(
                repo_id=model,
                filename="model.safetensors",
                local_dir=base_weights_dir,
            )

        # Download DWPose
        dwpose_dir = os.path.join(base_weights_dir, "dwpose")
        os.makedirs(dwpose_dir, exist_ok=True)
        for filename in ["yolox_l.onnx", "dw-ll_ucoco_384.onnx"]:
            if not os.path.exists(os.path.join(dwpose_dir, filename)):
                print(f"FashnVTON: Downloading DWPose/{filename} to {dwpose_dir}...")
                hf_hub_download(
                    repo_id="fashn-ai/DWPose",
                    filename=filename,
                    local_dir=dwpose_dir,
                )

        # Initialize Pipeline
        print(f"FashnVTON: Loading pipeline from {base_weights_dir}...")
        pipeline = TryOnPipeline(weights_dir=base_weights_dir)

        return (pipeline,)


class FashnVtonInference:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("FASHN_VTON_PIPELINE",),
                "person_image": ("IMAGE",),
                "garment_image": ("IMAGE",),
                "garment_photo_type": (
                    ["model", "flat-lay"],
                    {
                        "default": "model",
                        "tooltip": "'model' if garment is worn by a person, 'flat-lay' for product shots on plain backgrounds",
                    },
                ),
                "category": (
                    ["tops", "bottoms", "one-pieces"],
                    {"default": "tops", "tooltip": "Garment category - 'tops', 'bottoms', or 'one-pieces'"},
                ),
                "skip_cfg_last_n_steps": (
                    "INT",
                    {"default": 1, "tooltip": "Skip CFG for final N steps to prevent color saturation"},
                ),
                "segmentation_free": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "If True, skip person clothing-agnostic masking. Set to False to use person segmentation for person mask control.",
                    },
                ),
                "steps": (
                    "INT",
                    {
                        "default": 30,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                        "tooltip": "Recommended: 20 (fast), 30 (balanced), 50 (quality)",
                    },
                ),
                "cfg": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 10.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
                "pose_source": (
                    ["auto", "internal_dwpose", "external_pose_keypoints"],
                    {
                        "default": "auto",
                        "tooltip": "auto: use external keypoints when connected, otherwise internal DWPose fallback.",
                    },
                ),
                "parser_backend": (
                    ["fashn_human_parser", "external_fashn_labelmap"],
                    {
                        "default": "fashn_human_parser",
                        "tooltip": "Use external_fashn_labelmap only when providing segmentation images encoded with FASHN label IDs (0-17).",
                    },
                ),
            },
            "optional": {
                "person_pose_keypoints": (
                    "FASHN_DWPOSE_KEYPOINTS",
                    {"tooltip": "Optional external keypoints for person pose. Used when pose_source is auto/external_pose_keypoints."},
                ),
                "garment_pose_keypoints": (
                    "FASHN_DWPOSE_KEYPOINTS",
                    {"tooltip": "Optional external keypoints for garment pose. Used when pose_source is auto/external_pose_keypoints."},
                ),
                "person_segmentation_image": (
                    "IMAGE",
                    {
                        "tooltip": "Optional external person labelmap. Only affects output when segmentation_free is False and parser_backend is external_fashn_labelmap.",
                    },
                ),
                "garment_segmentation_image": (
                    "IMAGE",
                    {
                        "tooltip": "Optional external garment labelmap. Used when parser_backend is external_fashn_labelmap.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "FashnAI"

    def process(
        self,
        pipeline,
        person_image,
        garment_image,
        garment_photo_type,
        category,
        skip_cfg_last_n_steps,
        segmentation_free,
        steps,
        cfg,
        seed,
        keep_model_loaded,
        pose_source="auto",
        parser_backend="fashn_human_parser",
        person_pose_keypoints=None,
        garment_pose_keypoints=None,
        person_segmentation_image=None,
        garment_segmentation_image=None,
    ):
        device = comfy.model_management.get_torch_device()

        print(f"FashnVTON: Moving models to {device}...")
        if hasattr(pipeline, "tryon_model"):
            pipeline.tryon_model.to(device)
        if hasattr(pipeline, "hp_model"):
            if hasattr(pipeline.hp_model, "model"):
                pipeline.hp_model.model.to(device)
            pipeline.hp_model.device = device

        pbar = comfy.utils.ProgressBar(steps)

        def progress_callback(step, total_steps):
            pbar.update_absolute(step + 1, total_steps)

        person_pil = _tensor_to_pil(person_image)
        garment_pil = _tensor_to_pil(garment_image)

        person_segmentation_pil = (
            _tensor_to_pil(person_segmentation_image) if person_segmentation_image is not None else None
        )
        garment_segmentation_pil = (
            _tensor_to_pil(garment_segmentation_image) if garment_segmentation_image is not None else None
        )

        seed = seed % (2**32)

        try:
            result = pipeline(
                person_image=person_pil,
                garment_image=garment_pil,
                category=category,
                garment_photo_type=garment_photo_type,
                segmentation_free=segmentation_free,
                skip_cfg_last_n_steps=skip_cfg_last_n_steps,
                num_timesteps=steps,
                guidance_scale=cfg,
                seed=seed,
                pose_source=pose_source,
                person_pose_keypoints=person_pose_keypoints,
                garment_pose_keypoints=garment_pose_keypoints,
                parser_backend=parser_backend,
                person_segmentation_image=person_segmentation_pil,
                garment_segmentation_image=garment_segmentation_pil,
                callback=progress_callback,
            )
        finally:
            # Handle Offloading
            if not keep_model_loaded:
                print("FashnVTON: Unloading models from VRAM...")

                if hasattr(pipeline, "tryon_model"):
                    pipeline.tryon_model.to("cpu")
                if hasattr(pipeline, "hp_model"):
                    if hasattr(pipeline.hp_model, "model"):
                        pipeline.hp_model.model.to("cpu")
                    pipeline.hp_model.device = torch.device("cpu")
                if hasattr(pipeline, "pose_model"):
                    del pipeline.pose_model
                if hasattr(pipeline, "pose_provider"):
                    pipeline.pose_provider = None

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

                comfy.model_management.soft_empty_cache()

        # Convert back to ComfyUI format (B, H, W, C)
        output_img = np.array(result.images[0]).astype(np.float32) / 255.0
        output_tensor = torch.from_numpy(output_img).unsqueeze(0)

        return (output_tensor,)


class FashnPoseKeypointsAdapter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_keypoints": ("POSE_KEYPOINT",),
                "single_person": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("FASHN_DWPOSE_KEYPOINTS",)
    FUNCTION = "adapt"
    CATEGORY = "FashnAI/Adapters"

    def adapt(self, pose_keypoints, single_person):
        converted = convert_pose_keypoints_to_dwpose(pose_keypoints, single_person=single_person)
        if converted is None:
            raise ValueError(
                "Unsupported POSE_KEYPOINT payload for Fashn adapter. "
                "Expected OpenPose-like `people` data or DWPose-style bodies."
            )
        return (converted,)


class FashnMaskToLabelmap:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_image": ("IMAGE",),
                "category": (["tops", "bottoms", "one-pieces"], {"default": "tops"}),
                "mask_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "label_id_override": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": MAX_FASHN_LABEL_ID,
                        "step": 1,
                        "tooltip": "-1 uses category default label id.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "convert"
    CATEGORY = "FashnAI/Adapters"

    def convert(self, mask_image, category, mask_threshold, label_id_override):
        arr = mask_image.detach().cpu().numpy().astype(np.float32)
        if arr.shape[-1] == 1:
            mask = arr[..., 0]
        elif arr.shape[-1] >= 3:
            mask = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
        else:
            mask = np.mean(arr, axis=-1)

        mask = np.clip(mask, 0.0, 1.0)
        fg = mask >= float(mask_threshold)

        label_id = int(label_id_override)
        if label_id < 0:
            label_id = DEFAULT_CATEGORY_LABEL_ID[category]

        label_map = np.zeros(mask.shape, dtype=np.uint8)
        label_map[fg] = label_id

        out = np.repeat(label_map[..., None], 3, axis=-1).astype(np.float32) / 255.0
        return (torch.from_numpy(out),)


NODE_CLASS_MAPPINGS = {
    "FashnVtonLoader": FashnVtonLoader,
    "FashnVtonInference": FashnVtonInference,
    "FashnPoseKeypointsAdapter": FashnPoseKeypointsAdapter,
    "FashnMaskToLabelmap": FashnMaskToLabelmap,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FashnVtonLoader": "(Down)load Fashn VTON",
    "FashnVtonInference": "Fashn VTON Inference",
    "FashnPoseKeypointsAdapter": "Fashn Pose Keypoints Adapter",
    "FashnMaskToLabelmap": "Fashn Mask to Labelmap",
}
