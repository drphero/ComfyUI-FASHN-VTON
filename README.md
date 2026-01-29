# ComfyUI FASHN VTON v1.5 Custom Nodes

This custom node set implements the [FASHN VTON v1.5](https://github.com/fashn-AI/fashn-vton-1.5) model for virtual try-on in ComfyUI.

## Installation

**Via ComfyUI-Manager:**  
Search for "ComfyUI-FASHN-VTON" in the manager and install it directly.

**Manually:**
1.  Clone this repo into `custom_nodes` folder.
2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    Note: If you are using a portable version of ComfyUI, use the corresponding python executable.

## How to Use

<img width="980" height="496" alt="example" src="https://github.com/user-attachments/assets/b82241d9-4976-4a89-8922-b347c367a335" />

### 1. (Down)load FASHN VTON

Use the **(Down)load FASHN VTON** node.

The model **downloads automatically on first use**. In most cases, **no manual download is required**.

When this node runs, it will:
- Download the **FASHN VTON v1.5** model weights
- Download the required **DWPose** models
- Store everything under `ComfyUI/models/fashn-vton/`
- Load the pipeline automatically

If the files already exist locally, the download step is skipped.

#### Automatic Downloads

The following files are fetched automatically from Hugging Face:

- **FASHN VTON model**
  - `model.safetensors`
  - Source: https://huggingface.co/fashn-ai/fashn-vton-1.5

- **DWPose models**
  - `yolox_l.onnx`
  - `dw-ll_ucoco_384.onnx`
  - Source: https://huggingface.co/fashn-ai/DWPose

#### Manual Download (Optional)

If you prefer to download the models manually (e.g. for offline use), place the files in the following directory structure:

```
ComfyUI/
└── models/
    └── fashn-vton/
        ├── model.safetensors
        └── dwpose/
            ├── yolox_l.onnx
            └── dw-ll_ucoco_384.onnx
```

### 2. Inference

Use the **FASHN VTON Inference** node:
- **pipeline**: Connect from the Loader node.
- **person_image**: The image of the person.
- **garment_image**: The image of the garment.
- **category**: `tops`, `bottoms`, or `one-pieces`.
- **num_timesteps**: Recommended 30–50.
- **guidance_scale**: Recommended 1.5–3.0.
- **keep_model_loaded**: If set to `false`, the model will be moved to CPU after each inference to save VRAM.

## Progress Bar

The inference node supports the ComfyUI native progress bar to show the status of the sampling process.

## Credits

Model by [FASHN AI](https://fashn.ai/). Implementation based on their open-source repository.
