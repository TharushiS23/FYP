"""
Brain Tumor MRI Classification Dashboard
Ensemble: ResNet50 + EfficientNet-B0 (Logit Average, Class-wise Weighted)
Classes: Glioma | Meningioma | No Tumor | Pituitary
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights
import cv2
import io
import os

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroScan AI — Brain Tumor Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CUSTOM CSS  (dark clinical theme)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0d1117;
    color: #e6edf3;
}

/* ── Header ── */
.ns-header {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
    border-bottom: 1px solid #21262d;
    padding: 2rem 2.5rem 1.5rem;
    margin: -1rem -1rem 2rem -1rem;
}
.ns-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.1rem;
    font-weight: 600;
    letter-spacing: -0.5px;
    color: #58a6ff;
    margin: 0 0 0.25rem 0;
}
.ns-subtitle {
    font-size: 0.9rem;
    color: #8b949e;
    font-weight: 300;
    letter-spacing: 0.6px;
    text-transform: uppercase;
}

/* ── Cards ── */
.ns-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
}
.ns-card-title {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #8b949e;
    margin-bottom: 0.8rem;
}

/* ── Result badges ── */
.badge-tumor {
    display: inline-block;
    background: rgba(248,81,73,0.15);
    border: 1px solid rgba(248,81,73,0.4);
    color: #f85149;
    border-radius: 6px;
    padding: 0.25rem 0.8rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.5px;
}
.badge-healthy {
    display: inline-block;
    background: rgba(63,185,80,0.15);
    border: 1px solid rgba(63,185,80,0.4);
    color: #3fb950;
    border-radius: 6px;
    padding: 0.25rem 0.8rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.5px;
}
.badge-unknown {
    display: inline-block;
    background: rgba(210,153,34,0.15);
    border: 1px solid rgba(210,153,34,0.4);
    color: #d29922;
    border-radius: 6px;
    padding: 0.25rem 0.8rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.5px;
}

/* ── Confidence bar ── */
.conf-row { margin-bottom: 0.5rem; }
.conf-label {
    font-size: 0.78rem;
    color: #8b949e;
    font-family: 'IBM Plex Mono', monospace;
    margin-bottom: 2px;
}
.conf-bar-bg {
    background: #21262d;
    border-radius: 4px;
    height: 8px;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.6s ease;
}

/* ── Warning banner ── */
.ns-warning {
    background: rgba(210,153,34,0.08);
    border: 1px solid rgba(210,153,34,0.3);
    border-radius: 8px;
    padding: 1rem 1.4rem;
    margin-top: 2.5rem;
    font-size: 0.84rem;
    color: #d29922;
    line-height: 1.6;
}
.ns-warning strong { color: #e3b341; }

/* ── Error box ── */
.ns-error {
    background: rgba(248,81,73,0.08);
    border: 1px solid rgba(248,81,73,0.3);
    border-radius: 8px;
    padding: 1rem 1.4rem;
    color: #f85149;
    font-size: 0.88rem;
}

/* ── Upload zone ── */
.uploadedFile { background: #161b22 !important; }
[data-testid="stFileUploadDropzone"] {
    background: #161b22 !important;
    border: 1px dashed #30363d !important;
    border-radius: 8px !important;
}

/* ── Step label ── */
.step-label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #58a6ff;
    margin-bottom: 0.3rem;
}

/* ── Divider ── */
.ns-divider { border: none; border-top: 1px solid #21262d; margin: 1.8rem 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
CLASS_NAMES   = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
NUM_CLASSES   = 4
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_UPLOADS   = None   # No limit — user may upload any number of slices

# Class-wise ensemble weights (must sum to 1.0 per class)
RESNET_WEIGHTS     = torch.tensor([0.4, 0.4, 0.6, 0.6])  # per class
EFFICIENTNET_WEIGHTS = torch.tensor([0.6, 0.6, 0.4, 0.4])  # per class

WEIGHT_FILES = {
    "resnet":      "resnet50_finetuned_optuna.pth",
    "efficientnet": "efficientnetB0_finetuned_optuna.pth",
}

PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ─────────────────────────────────────────────
# MODEL BUILDERS
# ─────────────────────────────────────────────

def build_resnet(num_classes=NUM_CLASSES):
    """ResNet-50: freeze all conv layers except layer4 + fc."""
    model = models.resnet50(weights=None)
    for name, param in model.named_parameters():
        if not (name.startswith("layer4") or name.startswith("fc")):
            param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_efficientnet(num_classes=NUM_CLASSES):
    """EfficientNet-B0: freeze all blocks except the last one + classifier."""
    model = models.efficientnet_b0(weights=None)
    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze last MBConv block (index 7) + classifier
    for param in model.features[7].parameters():
        param.requires_grad = True
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


@st.cache_resource(show_spinner=False)
def load_models():
    """Load both models from disk. Returns (resnet, efficientnet) or raises."""
    missing = [v for v in WEIGHT_FILES.values() if not os.path.exists(v)]
    if missing:
        raise FileNotFoundError(
            f"Model weight file(s) not found: {missing}\n"
            "Place them in the same directory as app.py."
        )

    resnet = build_resnet()
    resnet.load_state_dict(torch.load(WEIGHT_FILES["resnet"],
                                      map_location=DEVICE))
    resnet.to(DEVICE).eval()

    effnet = build_efficientnet()
    effnet.load_state_dict(torch.load(WEIGHT_FILES["efficientnet"],
                                      map_location=DEVICE))
    effnet.to(DEVICE).eval()

    return resnet, effnet

# ─────────────────────────────────────────────
# MRI VERIFICATION
# ─────────────────────────────────────────────

def is_valid_mri(pil_image: Image.Image) -> tuple[bool, str]:
    """
    Heuristic verification that the image looks like a brain MRI scan.
    Checks:
      1. Not near-blank / all-black
      2. Grayscale-ish (MRI scans have low colour saturation)
      3. Reasonable dynamic range
    Returns (is_valid, reason_if_invalid)
    """
    img_np = np.array(pil_image.convert("RGB")).astype(np.float32)

    # 1. Brightness check — reject near-black images
    mean_brightness = img_np.mean()
    if mean_brightness < 5.0:
        return False, "Image appears to be blank or completely black."

    # 2. Colour saturation check — MRI scans are grayscale (R≈G≈B)
    r, g, b = img_np[:,:,0], img_np[:,:,1], img_np[:,:,2]
    rg_diff = np.abs(r - g).mean()
    rb_diff = np.abs(r - b).mean()
    gb_diff = np.abs(g - b).mean()
    avg_color_diff = (rg_diff + rb_diff + gb_diff) / 3
    if avg_color_diff > 30:
        return False, (
            "Image appears highly colourful. MRI scans are typically "
            "grayscale. Please upload a valid MRI scan."
        )

    # 3. Dynamic range — reject images with almost no contrast
    gray = img_np.mean(axis=2)
    if gray.std() < 8.0:
        return False, (
            "Image has very low contrast. This does not appear to be a "
            "valid MRI scan."
        )

    return True, ""

# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────

def predict_ensemble(tensor: torch.Tensor,
                     resnet: nn.Module,
                     effnet: nn.Module) -> tuple[np.ndarray, int, float]:
    """
    Class-wise logit-weighted ensemble.
    Returns (softmax_probs, predicted_class_idx, confidence).
    """
    inp = tensor.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits_r = resnet(inp).squeeze(0).cpu()   # [4]
        logits_e = effnet(inp).squeeze(0).cpu()   # [4]

    # Class-wise weighted average of logits
    combined = RESNET_WEIGHTS * logits_r + EFFICIENTNET_WEIGHTS * logits_e
    probs = torch.softmax(combined, dim=0).numpy()
    pred  = int(np.argmax(probs))
    conf  = float(probs[pred])
    return probs, pred, conf

# ─────────────────────────────────────────────
# GRAD-CAM
# ─────────────────────────────────────────────

class GradCAM:
    """Simple GradCAM hook for a single target layer."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model        = model
        self.gradients    = None
        self.activations  = None
        self._hooks       = []

        self._hooks.append(
            target_layer.register_forward_hook(self._save_activation)
        )
        self._hooks.append(
            target_layer.register_full_backward_hook(self._save_gradient)
        )

    def _save_activation(self, _, __, output):
        self.activations = output.detach()

    def _save_gradient(self, _, __, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        inp = tensor.unsqueeze(0).to(DEVICE)
        inp.requires_grad_(True)

        self.model.zero_grad()
        logits = self.model(inp)
        logits[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [1,C,1,1]
        cam     = (weights * self.activations).sum(dim=1).squeeze()
        cam     = torch.relu(cam).cpu().numpy()

        # Normalise
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        # Resize to 224×224
        cam = cv2.resize(cam, (224, 224))
        return cam

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()


def get_target_layers(resnet: nn.Module, effnet: nn.Module):
    """Return the last conv layers for GradCAM."""
    # ResNet-50: last layer of layer4
    resnet_layer = resnet.layer4[-1].conv3
    # EfficientNet-B0: last conv in last MBConv block
    effnet_layer  = effnet.features[7][-1].block[-1][0]
    return resnet_layer, effnet_layer


def ensemble_gradcam(tensor: torch.Tensor,
                     resnet: nn.Module,
                     effnet: nn.Module,
                     class_idx: int) -> np.ndarray:
    """
    Generate GradCAM from both models and average them.
    Uses same class-wise weights as the ensemble.
    """
    r_layer, e_layer = get_target_layers(resnet, effnet)

    gcam_r = GradCAM(resnet, r_layer)
    gcam_e = GradCAM(effnet,  e_layer)

    cam_r = gcam_r.generate(tensor, class_idx)
    cam_e = gcam_e.generate(tensor, class_idx)

    gcam_r.remove_hooks()
    gcam_e.remove_hooks()

    # Weight the CAMs the same way we weight the logits
    w_r = float(RESNET_WEIGHTS[class_idx])
    w_e = float(EFFICIENTNET_WEIGHTS[class_idx])
    combined = (w_r * cam_r + w_e * cam_e) / (w_r + w_e)
    return combined


def overlay_gradcam(pil_image: Image.Image,
                    cam: np.ndarray,
                    alpha: float = 0.45) -> np.ndarray:
    """Overlay a GradCAM heatmap on the original image."""
    orig   = np.array(pil_image.convert("RGB"))
    orig   = cv2.resize(orig, (224, 224))
    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam), cv2.COLORMAP_JET
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    blended = (alpha * heatmap + (1 - alpha) * orig).astype(np.uint8)
    return blended

# ─────────────────────────────────────────────
# VISUALISATION HELPERS
# ─────────────────────────────────────────────

CLASS_COLORS = {
    "Glioma":     "#f85149",
    "Meningioma": "#f85149",
    "Pituitary":  "#f85149",
    "No Tumor":   "#3fb950",
}

def conf_bar_html(class_name: str, prob: float) -> str:
    pct   = prob * 100
    color = CLASS_COLORS.get(class_name, "#58a6ff")
    return f"""
    <div class="conf-row">
      <div class="conf-label">{class_name} &nbsp;<span style="color:{color}">{pct:.1f}%</span></div>
      <div class="conf-bar-bg">
        <div class="conf-bar-fill" style="width:{pct:.1f}%;background:{color};"></div>
      </div>
    </div>
    """


def prediction_badge_html(class_name: str) -> str:
    if class_name == "No Tumor":
        return f'<span class="badge-healthy">✓ {class_name}</span>'
    return f'<span class="badge-tumor">⚠ {class_name}</span>'


def fig_to_pil(fig) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor="#0d1117", dpi=120)
    buf.seek(0)
    return Image.open(buf)

# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────

def main():
    # ── Header ──
    st.markdown("""
    <div class="ns-header">
      <div class="ns-title">🧠 NeuroScan AI</div>
      <div class="ns-subtitle">Brain Tumour Classification · Ensemble ResNet-50 + EfficientNet-B0 · GradCAM Explainability</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Load models ──
    with st.spinner("Loading ensemble model weights…"):
        try:
            resnet, effnet = load_models()
        except FileNotFoundError as e:
            st.markdown(f'<div class="ns-error">🔴 <strong>Model weights not found.</strong><br>{e}</div>',
                        unsafe_allow_html=True)
            st.stop()

    # ── Upload ──
    st.markdown('<div class="step-label">Step 1 — Upload MRI Slices</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        label="Upload MRI brain scan slices (PNG / JPG / JPEG / WEBP) — any number of slices",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        key="mri_upload",
    )

    if not uploaded_files:
        st.markdown("""
        <div class="ns-card" style="text-align:center; padding: 2.5rem 1.6rem; color:#8b949e;">
          <div style="font-size:2.5rem;margin-bottom:0.5rem;">🔬</div>
          <div style="font-weight:600; margin-bottom:0.3rem;">No images uploaded yet</div>
          <div style="font-size:0.85rem;">Upload between 1 and 5 MRI brain scan slices to begin analysis.</div>
        </div>
        """, unsafe_allow_html=True)
        _render_disclaimer()
        return

    # ── Process each image ──
    st.markdown('<hr class="ns-divider">', unsafe_allow_html=True)
    st.markdown('<div class="step-label">Step 2 — Verification · Prediction · Explainability</div>',
                unsafe_allow_html=True)

    # ── Pass 1: verify all images first, collect valid ones ──
    valid_slices   = []   # list of (idx, file, pil_img, tensor)
    invalid_slices = []   # list of (idx, file, reason)

    for idx, file in enumerate(uploaded_files):
        pil_img = Image.open(file).convert("RGB")
        ok, reason = is_valid_mri(pil_img)
        if ok:
            tensor = PREPROCESS(pil_img)
            valid_slices.append((idx, file, pil_img, tensor))
        else:
            invalid_slices.append((idx, file, reason))

    # ── Show blocked images immediately ──
    for idx, file, reason in invalid_slices:
        st.markdown(f"""
        <div class="ns-card">
          <div class="ns-card-title">Slice {idx + 1} — {file.name}</div>
          <div class="ns-error">
            🔴 <strong>This image could not be verified as an MRI scan.</strong><br><br>
            {reason}<br><br>
            Please upload a valid brain MRI scan image (grayscale, proper contrast, not blank).
            If you believe this is a valid scan, check that the file has not been corrupted or
            incorrectly exported.
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<hr class="ns-divider">', unsafe_allow_html=True)

    if not valid_slices:
        st.markdown("""
        <div class="ns-card" style="text-align:center; padding:2rem; color:#8b949e;">
          <div style="font-size:2rem;margin-bottom:0.5rem;">🚫</div>
          <div style="font-weight:600;">No valid MRI scans to analyse.</div>
          <div style="font-size:0.85rem; margin-top:0.4rem;">
            None of the uploaded images passed MRI verification. Please upload valid brain MRI scans.
          </div>
        </div>
        """, unsafe_allow_html=True)
        _render_disclaimer()
        return

    # ── Pass 2: run predictions on all valid slices ──
    results = []   # list of (idx, file, pil_img, tensor, probs, pred_idx, conf)
    with st.spinner("Running ensemble model…"):
        for idx, file, pil_img, tensor in valid_slices:
            probs, pred_idx, conf = predict_ensemble(tensor, resnet, effnet)
            results.append((idx, file, pil_img, tensor, probs, pred_idx, conf))

    # ── Identify the highest-confidence slice for GradCAM ──
    best_result = max(results, key=lambda x: x[6])   # x[6] = conf
    best_idx    = best_result[0]

    # ── Generate GradCAM only for the best slice ──
    with st.spinner("Generating GradCAM heatmap for highest-confidence slice…"):
        _, _, best_pil, best_tensor, _, best_pred_idx, _ = best_result
        best_cam     = ensemble_gradcam(best_tensor, resnet, effnet, best_pred_idx)
        best_overlay = overlay_gradcam(best_pil, best_cam)

    # ── Render all valid slices ──
    for idx, file, pil_img, tensor, probs, pred_idx, conf in results:
        pred_class = CLASS_NAMES[pred_idx]
        is_best    = (idx == best_idx)

        with st.container():
            label = f"Slice {idx + 1} — {file.name}"
            if is_best:
                label += " &nbsp;<span style='color:#d29922;font-size:0.7rem;'>★ HIGHEST CONFIDENCE</span>"

            st.markdown(f"""
            <div class="ns-card">
              <div class="ns-card-title">{label}</div>
            </div>
            """, unsafe_allow_html=True)

            if is_best:
                col_orig, col_heatmap, col_results = st.columns([1, 1, 1.4], gap="medium")
            else:
                col_orig, col_results = st.columns([1, 1.8], gap="medium")

            # ── Original image ──
            with col_orig:
                st.markdown('<div class="step-label">Original Scan</div>',
                            unsafe_allow_html=True)
                st.image(pil_img, use_container_width=True)

            # ── GradCAM (best slice only) ──
            if is_best:
                with col_heatmap:
                    st.markdown('<div class="step-label">GradCAM Heatmap</div>',
                                unsafe_allow_html=True)
                    st.image(best_overlay, use_container_width=True,
                             caption="Region that most influenced the prediction")

            # ── Results ──
            with col_results:
                st.markdown('<div class="step-label">Prediction</div>',
                            unsafe_allow_html=True)

                st.markdown(
                    f'<div style="margin-bottom:1rem;">{prediction_badge_html(pred_class)}'
                    f'<span style="font-size:0.8rem; color:#8b949e; margin-left:0.6rem;">'
                    f'Confidence {conf*100:.1f}%</span></div>',
                    unsafe_allow_html=True,
                )

                st.markdown('<div class="ns-card-title">Class Probabilities</div>',
                            unsafe_allow_html=True)
                bars_html = "".join(
                    conf_bar_html(CLASS_NAMES[i], float(probs[i]))
                    for i in range(NUM_CLASSES)
                )
                st.markdown(bars_html, unsafe_allow_html=True)

                if pred_class != "No Tumor":
                    st.markdown(
                        f'<div style="margin-top:0.8rem;font-size:0.78rem;color:#8b949e;">'
                        f'Detected tumour type: <span style="color:#f85149;font-weight:600;">{pred_class}</span>. '
                        f'Please refer to a qualified radiologist for clinical confirmation.</div>',
                        unsafe_allow_html=True,
                    )

                if is_best and not (idx == results[0][0] and len(results) == 1):
                    st.markdown(
                        '<div style="margin-top:0.6rem;font-size:0.75rem;color:#d29922;">'
                        '★ GradCAM shown for this slice — it had the highest prediction confidence.</div>',
                        unsafe_allow_html=True,
                    )

        st.markdown('<hr class="ns-divider">', unsafe_allow_html=True)

    _render_disclaimer()


def _render_disclaimer():
    st.markdown("""
    <div class="ns-warning">
      <strong>⚠ Clinical Disclaimer</strong><br>
      This tool is powered by an AI model and is intended for <strong>research and educational purposes only</strong>.
      It is <strong>not a certified medical device</strong> and must not be used as a substitute for professional
      medical advice, diagnosis, or treatment. AI models can and do make errors — including false positives and
      false negatives. All outputs should be reviewed and confirmed by a qualified radiologist or medical professional
      before any clinical decision is made. If you have concerns about a scan, please consult a healthcare provider immediately.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()