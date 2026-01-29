import os
import torch
import numpy as np
from PIL import Image
import folder_paths
import comfy.utils
import comfy.model_management
from .fashn_vton import TryOnPipeline

model_list = [
    'fashn-ai/fashn-vton-1.5'
]

class FashnVtonLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (model_list, {"default": 'fashn-ai/fashn-vton-1.5'})
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
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("FASHN_VTON_PIPELINE",),
                "person_image": ("IMAGE",),
                "garment_image": ("IMAGE",),
                "category": (["tops", "bottoms", "one-pieces"], {"default": "tops"}),
                "num_timesteps": ("INT", {"default": 30, "min": 1, "max": 100, "step": 1}),
                "guidance_scale": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 10.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "FashnAI"

    def process(self, pipeline, person_image, garment_image, category, num_timesteps, guidance_scale, seed, keep_model_loaded):
        
        device = comfy.model_management.get_torch_device()
        
        print(f"FashnVTON: Moving models to {device}...")
        if hasattr(pipeline, "tryon_model"):
            pipeline.tryon_model.to(device)
        if hasattr(pipeline, "hp_model"):
            if hasattr(pipeline.hp_model, "model"):
                pipeline.hp_model.model.to(device)

        pbar = comfy.utils.ProgressBar(num_timesteps)
        
        def progress_callback(step, total_steps):
            pbar.update_absolute(step + 1, total_steps)

        # ComfyUI images are (B, H, W, C) tensors in [0, 1]
        def tensor_to_pil(tensor):
            img = tensor[0].cpu().numpy()
            img = (img * 255).astype(np.uint8)
            return Image.fromarray(img)

        person_pil = tensor_to_pil(person_image)
        garment_pil = tensor_to_pil(garment_image)

        seed = seed % (2**32)
        
        try:
            result = pipeline(
                person_image=person_pil,
                garment_image=garment_pil,
                category=category,
                num_timesteps=num_timesteps,
                guidance_scale=guidance_scale,
                seed=seed,
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
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                
                comfy.model_management.soft_empty_cache()

        # Convert back to ComfyUI format (B, H, W, C)
        output_img = np.array(result.images[0]).astype(np.float32) / 255.0
        output_tensor = torch.from_numpy(output_img).unsqueeze(0)

        return (output_tensor,)

NODE_CLASS_MAPPINGS = {
    "FashnVtonLoader": FashnVtonLoader,
    "FashnVtonInference": FashnVtonInference,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FashnVtonLoader": "(Down)load Fashn VTON",
    "FashnVtonInference": "Fashn VTON Inference",
}