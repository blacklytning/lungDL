from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import io
import os
import base64
from typing import List

from PIL import Image
import numpy as np

import torch
from torchvision import transforms as T
import timm

app = FastAPI()


origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Grad-CAM API"}


# ---- Model + GradCAM utilities ----
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pth")


def load_model(num_classes: int = None):
    """Load a timm rexnet_150 model. If a local model checkpoint exists, load state_dict.
    If num_classes is provided, adjust the final layer to that output size.
    """
    model = timm.create_model("rexnet_150", pretrained=True)
    if num_classes is not None:
        # replace classifier if needed
        try:
            in_features = model.get_classifier().in_features
            model.reset_classifier(num_classes=num_classes)
        except Exception:
            # best-effort: try common attribute
            pass

    if os.path.exists(MODEL_PATH):
        try:
            state = torch.load(MODEL_PATH, map_location=DEVICE)
            # state may be state_dict or full model
            if isinstance(state, dict) and any(k.startswith("module.") or k in model.state_dict() for k in state.keys()):
                model.load_state_dict(state)
            else:
                # loaded as full model object
                model = state
        except Exception:
            # ignore and continue with pretrained
            pass

    model.to(DEVICE)
    model.eval()
    return model


class GradCAM:
    def __init__(self, model, target_layer_name: str):
        self.model = model
        self.target_layer_name = target_layer_name
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        modules = dict(self.model.named_modules())
        if self.target_layer_name not in modules:
            # pick last conv-like module
            for name, m in reversed(list(modules.items())):
                if hasattr(m, "weight") and m.weight.ndim == 4:
                    self.target_layer_name = name
                    break
        layer = modules.get(self.target_layer_name)
        if layer is None:
            raise RuntimeError(f"Cannot find layer {self.target_layer_name} in model")

        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            # grad_out is a tuple
            self.gradients = grad_out[0].detach()

        # use full backward hook to avoid deprecation issues
        layer.register_forward_hook(forward_hook)
        try:
            layer.register_full_backward_hook(backward_hook)
        except Exception:
            # fallback
            layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """Return a HxW cam normalized to [0,1]."""
        self.model.zero_grad()
        out = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(out.argmax(dim=1).item())
        score = out[:, class_idx]
        score.backward(retain_graph=True)

        grads = self.gradients
        acts = self.activations
        # global average pooling on gradients
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze(0).squeeze(0)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        cam_np = cam.cpu().numpy()
        return cam_np


# one global model instance
_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = load_model()
    return _MODEL


def transform_image(pil_img: Image.Image):
    tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return tf(pil_img).unsqueeze(0).to(DEVICE)


def make_overlay_image(pil_img: Image.Image, cam: np.ndarray, alpha: float = 0.4) -> str:
    """Return a base64-encoded PNG of the image with jet-colormap CAM overlay."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = pil_img.convert("RGB").resize((cam.shape[1], cam.shape[0]))
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    ax.imshow(img)
    ax.imshow(cam, cmap="jet", alpha=alpha)
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return b64


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # read image
    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image")

    model = get_model()
    inp = transform_image(img)

    # prediction
    with torch.no_grad():
        out = model(inp)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0].tolist()
        pred_idx = int(np.argmax(probs))

    # Grad-CAM (requires gradients so do a non-no-grad pass)
    try:
        gradcam = GradCAM(model, target_layer_name="features.16.conv")
        cam = gradcam.generate_cam(inp, class_idx=pred_idx)
        # cam is HxW in 0..1 range; resize to original image size
        cam_img = Image.fromarray(np.uint8(255 * cam)).resize(img.size)
        cam_arr = np.array(cam_img) / 255.0
        overlay_b64 = make_overlay_image(img, cam_arr, alpha=0.4)
    except Exception:
        overlay_b64 = None

    return JSONResponse({
        "filename": file.filename,
        "prediction_index": pred_idx,
        "probabilities": probs,
        "gradcam_overlay_base64": overlay_b64,
    })
