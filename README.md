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

Existing workflows remain valid. If you do not connect new optional inputs, behavior stays the same as before.

New optional controls:
- **pose_source**:
  - `auto` (default): use external keypoints if connected, otherwise internal DWPose
  - `internal_dwpose`: always use built-in DWPose
  - `external_pose_keypoints`: prefer external keypoints payloads
- **person_pose_keypoints / garment_pose_keypoints** (optional): keypoint payloads converted by **Fashn Pose Keypoints Adapter**.
- **parser_backend**:
  - `fashn_human_parser` (default)
  - `external_fashn_labelmap` (uses external segmentation maps encoded with FASHN label IDs)
- **person_segmentation_image / garment_segmentation_image** (optional): external segmentation maps.

Fallback behavior:
- If external keypoint payloads are missing/invalid, inference falls back to internal DWPose.
- For `flat-lay` garments without external garment pose, the pipeline uses the built-in dummy garment pose.
- If external segmentation maps are missing/invalid, inference falls back to `fashn_human_parser`.

### 3. Adapter Nodes

Two utility nodes are included to make third-party node outputs easier to connect:
- **Fashn Pose Keypoints Adapter**:
  - Converts `POSE_KEYPOINT` payloads into internal DWPose-style keypoints.
  - Supports single-person selection to match internal DWPose behavior.
- **Fashn Mask to Labelmap**:
  - Converts a single merged mask into a valid FASHN labelmap image.
  - Uses category defaults (`tops->top`, `bottoms->pants`, `one-pieces->dress`) with optional label ID override.

Typical external keypoint workflow (closest match to internal rendering):
1. Third-party DWPose/OpenPose node -> **Fashn Pose Keypoints Adapter**
2. Connect adapter output to `person_pose_keypoints` (and/or `garment_pose_keypoints`)
3. Set **pose_source** to `auto` or `external_pose_keypoints`

Typical external garment-mask workflow:
1. Third-party segmentation node -> merged garment mask
2. Mask -> **Fashn Mask to Labelmap** (choose category, optionally override label ID)
3. Connect output to `garment_segmentation_image`
4. Set `parser_backend=external_fashn_labelmap`

## Credits

Model by [FASHN AI](https://fashn.ai/). Implementation based on their open-source repository.
