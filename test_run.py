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

/* ── Fact card (persistent, dismissible) ── */
.ns-fact-card {
    position: relative;
    background: #0e1825;
    border: 1px solid #1f3a5c;
    border-left: 3px solid #58a6ff;
    border-radius: 10px;
    padding: 1.1rem 3rem 1.1rem 1.4rem;
    margin: 1rem 0 1.4rem 0;
}
.ns-fact-title {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #58a6ff;
    margin-bottom: 0.45rem;
}
.ns-fact-body {
    font-size: 0.87rem;
    color: #c9d1d9;
    line-height: 1.7;
}
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
# BRAIN TUMOUR FACTS
# ─────────────────────────────────────────────
BRAIN_TUMOUR_FACTS = [
    ("🧠 Did you know?",
     "Gliomas are the most common primary brain tumours in adults, accounting for about 80% of all malignant brain tumours. They arise from glial cells — the supportive tissue that protects and nourishes neurons."),
    ("🔬 About Meningiomas",
     "Meningiomas grow from the meninges, the protective membranes surrounding the brain and spinal cord. The vast majority are benign and slow-growing — many are discovered incidentally during scans done for entirely unrelated reasons."),
    ("💡 Pituitary Tumours",
     "Pituitary adenomas are almost always benign. Because the pituitary gland regulates key hormones, even a small tumour here can trigger significant hormonal changes — affecting growth, metabolism, fertility, and blood pressure."),
    ("📊 Early Detection Saves Lives",
     "For low-grade gliomas detected early, 5-year survival rates can exceed 70%. Routine check-ups and prompt investigation of neurological symptoms are among the most important steps for improving outcomes."),
    ("⚠️ Warning Signs to Know",
     "Common symptoms warranting prompt evaluation include: new-onset seizures, persistent morning headaches, unexplained nausea or vomiting, gradual changes in vision or speech, memory difficulties, and personality changes."),
    ("🩺 Headaches & Brain Tumours",
     "Not every headache signals a tumour — fewer than 1 in 1,000 headache patients have one. However, headaches that are worst in the morning, wake you from sleep, or escalate rapidly over days should be clinically evaluated without delay."),
    ("🏥 Modern Treatment Options",
     "Treatment depends on tumour type, grade, size, and location. Options include surgical resection, stereotactic radiosurgery, fractionated radiotherapy, chemotherapy, targeted molecular therapy, and immunotherapy — often used in combination."),
    ("🧬 Genetics & Risk Factors",
     "The majority of brain tumours are not inherited. However, conditions like Neurofibromatosis types 1 and 2, Li-Fraumeni syndrome, and Von Hippel-Lindau disease do carry elevated risk. Genetic counselling is advisable for families with multiple affected members."),
    ("🌍 Global Incidence",
     "Brain and CNS tumours account for approximately 3% of all cancers worldwide, with around 308,000 new cases diagnosed annually. They affect people of all ages, though incidence peaks in childhood and again in older adulthood."),
    ("👶 Brain Tumours in Children",
     "Brain tumours are the most common solid tumour in children and the leading cause of cancer-related death in paediatric patients. Medulloblastoma and pilocytic astrocytoma are the most frequent types, with many children achieving long-term remission."),
    ("🔍 Why MRI is the Gold Standard",
     "MRI provides exceptional soft-tissue contrast without ionising radiation. It allows clinicians to assess tumour size, exact location, involvement of adjacent structures, and vascularity — critical information for surgical planning and monitoring treatment response."),
    ("💊 Watch-and-Wait Approach",
     "For many benign or low-grade tumours, active surveillance (watch-and-wait) is an appropriate first step. Regular MRI monitoring allows clinicians to detect any growth while avoiding unnecessary intervention and preserving quality of life."),
    ("🧘 Reducing Modifiable Risk",
     "While most brain tumour risk factors are not controllable, general brain health practices matter: avoiding unnecessary ionising radiation exposure, not smoking, maintaining a healthy body weight, and protecting your head from repeated trauma are all advisable."),
    ("📱 AI in Neuro-Oncology",
     "Deep learning models trained on large MRI datasets can now assist radiologists in detecting and classifying brain tumours faster and with increasing precision. AI is a tool to support clinical judgement — not to replace the expertise of qualified clinicians."),
    ("🕐 Time Matters for High-Grade Tumours",
     "For aggressive tumours like Glioblastoma Multiforme (GBM), time from diagnosis to treatment initiation directly affects survival. The current standard of care — surgery followed by concurrent radiotherapy and temozolomide chemotherapy — should begin as soon as the patient is fit."),
    ("🧪 Tumour Biomarkers",
     "Molecular markers like IDH1/2 mutation status, MGMT promoter methylation, and 1p/19q co-deletion have transformed brain tumour classification. These markers guide treatment decisions and are strong predictors of prognosis independent of tumour grade."),
    ("🫀 Blood-Brain Barrier Challenges",
     "The blood-brain barrier (BBB), which protects the brain from harmful substances, also limits the delivery of many chemotherapy drugs to brain tumours. Overcoming the BBB is one of the most active areas of research in neuro-oncology today."),
]

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
    for param in model.parameters():
        param.requires_grad = False
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
    img_np = np.array(pil_image.convert("RGB")).astype(np.float32)

    mean_brightness = img_np.mean()
    if mean_brightness < 5.0:
        return False, "Image appears to be blank or completely black."

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
    inp = tensor.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits_r = resnet(inp).squeeze(0).cpu()
        logits_e = effnet(inp).squeeze(0).cpu()

    combined = RESNET_WEIGHTS * logits_r + EFFICIENTNET_WEIGHTS * logits_e
    probs = torch.softmax(combined, dim=0).numpy()
    pred  = int(np.argmax(probs))
    conf  = float(probs[pred])
    return probs, pred, conf

# ─────────────────────────────────────────────
# GRAD-CAM
# ─────────────────────────────────────────────

class GradCAM:
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

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam     = (weights * self.activations).sum(dim=1).squeeze()
        cam     = torch.relu(cam).cpu().numpy()

        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        cam = cv2.resize(cam, (224, 224))
        return cam

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()


def get_target_layers(resnet: nn.Module, effnet: nn.Module):
    resnet_layer = resnet.layer4[-1].conv3
    effnet_layer  = effnet.features[7][-1].block[-1][0]
    return resnet_layer, effnet_layer


def ensemble_gradcam(tensor: torch.Tensor,
                     resnet: nn.Module,
                     effnet: nn.Module,
                     class_idx: int) -> np.ndarray:
    r_layer, e_layer = get_target_layers(resnet, effnet)

    gcam_r = GradCAM(resnet, r_layer)
    gcam_e = GradCAM(effnet,  e_layer)

    cam_r = gcam_r.generate(tensor, class_idx)
    cam_e = gcam_e.generate(tensor, class_idx)

    gcam_r.remove_hooks()
    gcam_e.remove_hooks()

    w_r = float(RESNET_WEIGHTS[class_idx])
    w_e = float(EFFICIENTNET_WEIGHTS[class_idx])
    combined = (w_r * cam_r + w_e * cam_e) / (w_r + w_e)
    return combined


def overlay_gradcam(pil_image: Image.Image,
                    cam: np.ndarray,
                    alpha: float = 0.45) -> np.ndarray:
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
# PERSISTENT FACT CARD  (NEW)
# ─────────────────────────────────────────────

def _init_fact_state():
    """
    Pick a random fact once per analysis run and store it in session state.
    Call this before the spinner blocks so the same fact persists across reruns.
    """
    import random
    if "fact_dismissed" not in st.session_state:
        st.session_state.fact_dismissed = False
    if "current_fact" not in st.session_state:
        st.session_state.current_fact = random.choice(BRAIN_TUMOUR_FACTS)


def _reset_fact():
    """Pick a fresh fact for the next analysis run."""
    import random
    st.session_state.fact_dismissed = False
    st.session_state.current_fact = random.choice(BRAIN_TUMOUR_FACTS)


def _render_fact_card():
    """
    Render the persistent, dismissible fact card.
    The card stays visible until the user presses ✕.
    Nothing is rendered once dismissed.
    """
    if st.session_state.get("fact_dismissed", False):
        return

    title, body = st.session_state.current_fact

    st.markdown(f"""
    <div class="ns-fact-card">
      <div class="ns-fact-title">💡 Did you know? &nbsp;·&nbsp; {title}</div>
      <div class="ns-fact-body">{body}</div>
    </div>
    """, unsafe_allow_html=True)

    # Dismiss button — sits below the card so it's always visible
    if st.button("✕ Dismiss this fact", key="dismiss_fact"):
        st.session_state.fact_dismissed = True
        st.rerun()

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

    # ─────────────────────────────────────────────
    # STEP 1 — Upload
    # ─────────────────────────────────────────────
    st.markdown('<div class="step-label">Step 1 — Upload MRI Slices</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        label="Upload one or more brain MRI scan slices (PNG / JPG / JPEG / WEBP)",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        key="mri_upload",
    )

    if not uploaded_files:
        st.markdown("""
        <div class="ns-card" style="text-align:center; padding: 2.5rem 1.6rem; color:#8b949e;">
          <div style="font-size:2.5rem;margin-bottom:0.5rem;">🔬</div>
          <div style="font-weight:600; margin-bottom:0.3rem;">No images uploaded yet</div>
          <div style="font-size:0.85rem;">
            Upload your MRI brain scan slices above, then press <strong style="color:#e6edf3;">Analyse Scans</strong> when you are ready.
          </div>
        </div>
        """, unsafe_allow_html=True)
        _render_disclaimer()
        return

    # Show thumbnails
    st.markdown(f"""
    <div style="font-size:0.82rem; color:#8b949e; margin: 0.3rem 0 0.8rem 0;">
      {len(uploaded_files)} slice(s) loaded. Review them below, then press <strong style="color:#e6edf3;">Analyse Scans</strong> when ready.
    </div>
    """, unsafe_allow_html=True)

    thumb_cols = st.columns(min(len(uploaded_files), 6))
    for i, f in enumerate(uploaded_files):
        with thumb_cols[i % 6]:
            st.image(Image.open(f).convert("RGB"), caption=f.name, use_container_width=True)

    st.markdown('<hr class="ns-divider">', unsafe_allow_html=True)

    # ─────────────────────────────────────────────
    # STEP 2 — Predict button
    # ─────────────────────────────────────────────
    st.markdown('<div class="step-label">Step 2 — Run Analysis</div>', unsafe_allow_html=True)

    col_btn, col_note = st.columns([1, 4], gap="small")
    with col_btn:
        run_analysis = st.button(
            "🔍 Analyse Scans",
            type="primary",
            use_container_width=True,
        )
    with col_note:
        st.markdown(
            '<div style="font-size:0.8rem; color:#8b949e; padding-top:0.55rem;">'
            'Verification → Ensemble Prediction → GradCAM Heatmaps</div>',
            unsafe_allow_html=True,
        )

    if not run_analysis:
        _render_disclaimer()
        return

    # ─────────────────────────────────────────────
    # STEP 3 — Verify & Predict
    # ─────────────────────────────────────────────
    st.markdown('<hr class="ns-divider">', unsafe_allow_html=True)
    st.markdown('<div class="step-label">Step 3 — Results</div>', unsafe_allow_html=True)

    # Initialise / refresh the fact for this run
    _reset_fact()

    # Show the persistent fact card ONCE — stays until user dismisses it
    _render_fact_card()

    # ── Pass 1: verify all images ──
    valid_slices   = []
    invalid_slices = []

    with st.spinner("Verifying uploaded images…"):
        for idx, file in enumerate(uploaded_files):
            pil_img = Image.open(file).convert("RGB")
            ok, reason = is_valid_mri(pil_img)
            if ok:
                tensor = PREPROCESS(pil_img)
                valid_slices.append((idx, file, pil_img, tensor))
            else:
                invalid_slices.append((idx, file, reason))

    # ── Show blocked images ──
    for idx, file, reason in invalid_slices:
        st.markdown(f"""
        <div class="ns-card">
          <div class="ns-card-title">Slice {idx + 1} — {file.name}</div>
          <div class="ns-error">
            🔴 <strong>This image could not be verified as a valid MRI scan.</strong><br><br>
            {reason}<br><br>
            Please remove this file and upload a valid brain MRI scan image. Valid MRI scans are
            grayscale, have visible brain structure, and are not blank or corrupted. If you believe
            this is a valid scan, check that the file has not been incorrectly exported or compressed.
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
            None of the uploaded images passed MRI verification. Please upload valid brain MRI scans and try again.
          </div>
        </div>
        """, unsafe_allow_html=True)
        _render_disclaimer()
        return

    # ── Pass 2: ensemble predictions ──
    results = []
    with st.spinner(f"Running ensemble model on {len(valid_slices)} valid slice(s)…"):
        for idx, file, pil_img, tensor in valid_slices:
            probs, pred_idx, conf = predict_ensemble(tensor, resnet, effnet)
            results.append((idx, file, pil_img, tensor, probs, pred_idx, conf))

    # ── Pass 3: GradCAM ──
    gradcam_overlays = {}
    with st.spinner(f"Generating GradCAM heatmaps for {len(results)} slice(s)…"):
        for idx, file, pil_img, tensor, probs, pred_idx, conf in results:
            cam     = ensemble_gradcam(tensor, resnet, effnet, pred_idx)
            overlay = overlay_gradcam(pil_img, cam)
            gradcam_overlays[idx] = overlay

    # ─────────────────────────────────────────────
    # RENDER RESULTS
    # ─────────────────────────────────────────────
    for idx, file, pil_img, tensor, probs, pred_idx, conf in results:
        pred_class = CLASS_NAMES[pred_idx]
        overlay    = gradcam_overlays[idx]

        with st.container():
            st.markdown(f"""
            <div class="ns-card">
              <div class="ns-card-title">Slice {idx + 1} — {file.name}</div>
            </div>
            """, unsafe_allow_html=True)

            col_orig, col_heatmap, col_results = st.columns([1, 1, 1.4], gap="medium")

            with col_orig:
                st.markdown('<div class="step-label">Original Scan</div>', unsafe_allow_html=True)
                st.image(pil_img, use_container_width=True)

            with col_heatmap:
                st.markdown('<div class="step-label">GradCAM Heatmap</div>', unsafe_allow_html=True)
                st.image(overlay, use_container_width=True,
                         caption="Regions that most influenced the prediction")

            with col_results:
                st.markdown('<div class="step-label">Prediction</div>', unsafe_allow_html=True)

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
                        f'<div style="margin-top:0.9rem; font-size:0.78rem; color:#8b949e;">'
                        f'Detected tumour type: <span style="color:#f85149; font-weight:600;">{pred_class}</span>. '
                        f'Please refer to a qualified radiologist for clinical confirmation.</div>',
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