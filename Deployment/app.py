import os
from typing import Dict, Tuple

import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import transforms
from PIL import Image

# ── Constants ─────────────────────────────────────────────────────────────────
SPECIES_LABELS = {0: "Guava", 1: "Mango", 2: "Papaya"}
HEALTH_LABELS  = {0: "Healthy", 1: "Anthracnose"}
IMG_SIZE       = 224
NUM_SPECIES    = 3
NUM_HEALTH     = 2
DROPOUT        = 0.4
FPN_CHANNELS   = 256
GCA_REDUCTION  = 16

NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD  = [0.229, 0.224, 0.225]

preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Model architecture (matches training notebook exactly) ────────────────────

class GlobalContextAttention(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid(),
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, 1, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.global_context = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.norm = nn.LayerNorm([in_channels])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        ch_w = self.channel_attention(x).view(b, c, 1, 1)
        sp_w = self.spatial_attention(x)
        combined = x * ch_w + x * sp_w
        return x + self.global_context(combined)


class MultiHeadGCA(nn.Module):
    def __init__(self, in_channels: int, num_heads: int = 4, reduction: int = 16):
        super().__init__()
        assert in_channels % num_heads == 0
        self.num_heads = num_heads
        head_dim = in_channels // num_heads
        self.heads = nn.ModuleList(
            [GlobalContextAttention(head_dim, reduction) for _ in range(num_heads)]
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunks = x.chunk(self.num_heads, dim=1)
        attended = [h(c) for h, c in zip(self.heads, chunks)]
        return self.fusion(torch.cat(attended, dim=1))


class SwinLargeFPNBackbone(nn.Module):
    STAGE_DIMS = [192, 384, 768, 1536]

    def __init__(self, swin_model, fpn_channels: int = 256):
        super().__init__()
        self.swin = swin_model
        self.fpn_channels = fpn_channels
        self._feature_maps: Dict[int, torch.Tensor] = {}
        self._hooks = []
        self._register_hooks()

        self.lateral1 = nn.Conv2d(self.STAGE_DIMS[0], fpn_channels, 1)
        self.lateral2 = nn.Conv2d(self.STAGE_DIMS[1], fpn_channels, 1)
        self.lateral3 = nn.Conv2d(self.STAGE_DIMS[2], fpn_channels, 1)
        self.lateral4 = nn.Conv2d(self.STAGE_DIMS[3], fpn_channels, 1)
        self.smooth1  = nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1)
        self.smooth2  = nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1)
        self.smooth3  = nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1)
        self.smooth4  = nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1)
        self.out_channels = fpn_channels

    def _get_stages(self):
        if hasattr(self.swin, "layers"):
            return list(self.swin.layers)
        elif hasattr(self.swin, "features"):
            f = self.swin.features
            return [f[1], f[3], f[5], f[7]]
        return None

    def _register_hooks(self):
        stages = self._get_stages()
        if stages is None:
            raise RuntimeError("Cannot find Swin stages.")
        for idx, stage in enumerate(stages):
            h = stage.register_forward_hook(
                lambda m, inp, out, i=idx: self._feature_maps.update({i: out})
            )
            self._hooks.append(h)

    @staticmethod
    def _to_nchw(feat: torch.Tensor) -> torch.Tensor:
        if feat.dim() == 4 and feat.shape[-1] > feat.shape[1]:
            return feat.permute(0, 3, 1, 2).contiguous()
        return feat

    @staticmethod
    def _upsample_add(x, y):
        _, _, h, w = y.size()
        return F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False) + y

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        self._feature_maps = {}
        _ = self.swin(x)

        c1 = self._to_nchw(self._feature_maps[0])
        c2 = self._to_nchw(self._feature_maps[1])
        c3 = self._to_nchw(self._feature_maps[2])
        c4 = self._to_nchw(self._feature_maps[3])

        p4 = self.lateral4(c4)
        p3 = self._upsample_add(p4, self.lateral3(c3))
        p2 = self._upsample_add(p3, self.lateral2(c2))
        p1 = self._upsample_add(p2, self.lateral1(c1))

        return {
            "p1": self.smooth1(p1),
            "p2": self.smooth2(p2),
            "p3": self.smooth3(p3),
            "p4": self.smooth4(p4),
        }


class MultiTaskSwinLarge(nn.Module):
    def __init__(
        self,
        num_species: int = NUM_SPECIES,
        num_health:  int = NUM_HEALTH,
        pretrained:  bool = False,
        dropout:     float = DROPOUT,
        fpn_channels: int = FPN_CHANNELS,
        gca_reduction: int = GCA_REDUCTION,
    ):
        super().__init__()
        model_name = (
            "swin_large_patch4_window7_224.ms_in22k_ft_in1k"
            if pretrained
            else "swin_large_patch4_window7_224"
        )
        swin_model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.backbone = SwinLargeFPNBackbone(swin_model, fpn_channels=fpn_channels)
        self.gca = MultiHeadGCA(fpn_channels, num_heads=4, reduction=gca_reduction)
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(fpn_channels * 4, fpn_channels * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(fpn_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_channels * 2, fpn_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True),
        )
        self.global_pool  = nn.AdaptiveAvgPool2d(1)
        self.dropout      = nn.Dropout(dropout)
        self.head_species = nn.Linear(fpn_channels, num_species)
        self.head_health  = nn.Linear(fpn_channels, num_health)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.backbone(x)
        p1 = self.gca(feats["p1"])
        p2 = self.gca(feats["p2"])
        p3 = self.gca(feats["p3"])
        p4 = self.gca(feats["p4"])

        sz = p4.shape[-2:]
        ms = torch.cat(
            [F.adaptive_avg_pool2d(p1, sz),
             F.adaptive_avg_pool2d(p2, sz),
             F.adaptive_avg_pool2d(p3, sz),
             p4],
            dim=1,
        )
        pooled = self.global_pool(self.feature_fusion(ms)).flatten(1)
        pooled = self.dropout(pooled)
        return self.head_species(pooled), self.head_health(pooled)


# ── Load model once at startup ────────────────────────────────────────────────

def _load_model() -> MultiTaskSwinLarge:
    model = MultiTaskSwinLarge(pretrained=False)
    ckpt_path = os.path.join(os.path.dirname(__file__), "best_model.pt")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model.to(DEVICE)


print("Loading model…")
model = _load_model()
print(f"Model loaded on {DEVICE}")


# ── Inference ─────────────────────────────────────────────────────────────────

def predict(image: Image.Image) -> Tuple[Dict[str, float], Dict[str, float]]:
    if image is None:
        return {}, {}
    if image.mode != "RGB":
        image = image.convert("RGB")
    tensor = preprocess(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits_sp, logits_h = model(tensor)
    prob_sp = F.softmax(logits_sp, dim=1)[0].cpu()
    prob_h  = F.softmax(logits_h,  dim=1)[0].cpu()
    species_out = {SPECIES_LABELS[i]: float(prob_sp[i]) for i in range(NUM_SPECIES)}
    health_out  = {HEALTH_LABELS[i]:  float(prob_h[i])  for i in range(NUM_HEALTH)}
    return species_out, health_out


# ── Gradio UI helpers ─────────────────────────────────────────────────────────

CSS = """
/* layout */
.gradio-container { max-width: 1100px !important; margin: 0 auto !important; }
footer { display: none !important; }

/* app header */
.app-header {
    text-align: center;
    padding: 2.2rem 1.25rem 1.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    margin-bottom: 1.2rem;
    box-sizing: border-box;
    width: 100%;
}
.app-header h1 {
    font-size: clamp(1.35rem, 4vw, 1.9rem);
    font-weight: 800; margin: 0 0 0.5rem;
    letter-spacing: -0.015em;
    font-family: 'Inter', system-ui, sans-serif;
    line-height: 1.25;
    word-break: break-word;
}
.app-header p {
    font-size: clamp(0.82rem, 2vw, 0.92rem);
    line-height: 1.7;
    max-width: 560px;
    margin: 0 auto;
    opacity: 0.62;
    font-family: 'Inter', system-ui, sans-serif;
    text-wrap: balance;
}
@media (max-width: 600px) {
    .app-header { padding: 1.6rem 1rem 1.2rem; }
}

/* info / how-to cards */
.info-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1rem 1.35rem;
    margin-bottom: 0.85rem;
    font-family: 'Inter', system-ui, sans-serif;
}
.card-lbl {
    font-size: 0.66rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.12em;
    opacity: 0.38; margin-bottom: 0.65rem;
}
.steps ol { margin: 0; padding-left: 1.25rem; }
.steps li { font-size: 0.865rem; line-height: 1.88; opacity: 0.62; }
.steps li strong { opacity: 1; font-weight: 600; }

/* result panels */
.result-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1.1rem 1.35rem;
    margin-bottom: 0.85rem;
    font-family: 'Inter', system-ui, sans-serif;
}
.result-empty {
    text-align: center; padding: 2.4rem 1rem;
    opacity: 0.28; font-size: 0.88rem;
    font-family: 'Inter', system-ui, sans-serif;
}
.r-lbl {
    font-size: 0.66rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.12em;
    opacity: 0.38; margin-bottom: 0.48rem;
}
.r-top { font-size: 1.48rem; font-weight: 700; margin-bottom: 0.9rem; letter-spacing: -0.01em; }
.conf-row { display: flex; align-items: center; gap: 0.65rem; margin-bottom: 0.48rem; }
.conf-name { width: 90px; font-size: 0.8rem; opacity: 0.62; flex-shrink: 0; }
.conf-track { flex: 1; height: 5px; background: rgba(255,255,255,0.07); border-radius: 999px; overflow: hidden; }
.conf-fill  { height: 100%; border-radius: 999px; }
.conf-pct   { width: 44px; text-align: right; font-size: 0.8rem; font-variant-numeric: tabular-nums; opacity: 0.85; }

/* anthracnose action panel */
.action-card {
    background: rgba(239,68,68,0.06);
    border: 1px solid rgba(239,68,68,0.28);
    border-radius: 12px;
    padding: 1.1rem 1.35rem;
    margin-bottom: 0.85rem;
    font-family: 'Inter', system-ui, sans-serif;
}
.action-card .a-lbl {
    font-size: 0.66rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.12em;
    color: #ef4444; opacity: 0.8; margin-bottom: 0.55rem;
}
.action-card .a-title {
    font-size: 0.95rem; font-weight: 700; color: #ef4444;
    margin-bottom: 0.8rem;
}
.action-steps { list-style: none; padding: 0; margin: 0; }
.action-steps li {
    display: flex; gap: 0.65rem; align-items: flex-start;
    font-size: 0.845rem; line-height: 1.55; margin-bottom: 0.6rem; opacity: 0.88;
}
.action-steps li .si {
    flex-shrink: 0; min-width: 22px; height: 22px; border-radius: 50%;
    background: rgba(239,68,68,0.18); color: #ef4444;
    font-size: 0.65rem; font-weight: 700;
    display: flex; align-items: center; justify-content: center;
    margin-top: 0.12rem; padding: 0 3px;
}
.action-steps li strong { font-weight: 600; }

/* predict button */
#predict-btn {
    background: linear-gradient(135deg, #e07b39, #c86524) !important;
    color: #fff !important; border: none !important;
    border-radius: 10px !important; font-weight: 700 !important;
    font-size: 0.96rem !important; letter-spacing: 0.05em !important;
    transition: opacity .18s !important;
}
#predict-btn:hover { opacity: 0.82 !important; }

/* footer */
.app-footer {
    text-align: center; font-size: 0.76rem; opacity: 0.28;
    padding-top: 0.9rem; border-top: 1px solid rgba(255,255,255,0.06);
    margin-top: 0.5rem; font-family: 'Inter', system-ui, sans-serif;
}
"""

_EMPTY_RESULT = ""

_ANTHRACNOSE_ACTIONS = """
<div class="action-card">
  <div class="a-lbl">⚠ Disease Detected</div>
  <div class="a-title">Recommended Actions for Anthracnose</div>
  <ul class="action-steps">
    <li><span class="si">1</span><span>Remove infected leaves and fallen debris.</span></li>
    <li><span class="si">2</span><span>Do not compost diseased plant parts.</span></li>
    <li><span class="si">3</span><span>Avoid overhead watering; keep leaves dry.</span></li>
    <li><span class="si">4</span><span>Improve air circulation by pruning crowded branches.</span></li>
    <li><span class="si">5</span><span>Clean tools after touching infected plants.</span></li>
    <li><span class="si">6</span><span>Monitor nearby plants after rain or humid weather.</span></li>
    <li><span class="si">7</span><span>Remove rotten fruit or severely affected parts.</span></li>
    <li><span class="si">8</span><span>Use healthy, disease-free planting material.</span></li>
    <li><span class="si">9</span><span>Use fungicide only with expert/local label guidance.</span></li>
    <li><span class="si">10</span><span>Consult an agriculture expert if infection spreads.</span></li>
  </ul>
</div>
"""


def _conf_bar(name: str, prob: float, fill: str) -> str:
    pct = prob * 100
    return (
        f'<div class="conf-row">'
        f'<span class="conf-name">{name}</span>'
        f'<div class="conf-track">'
        f'<div class="conf-fill" style="width:{pct:.1f}%;background:{fill}"></div>'
        f'</div>'
        f'<span class="conf-pct">{pct:.1f}%</span>'
        f'</div>'
    )


def predict_html(image: Image.Image) -> Tuple[str, str, str]:
    species_dict, health_dict = predict(image)
    if not species_dict:
        return _EMPTY_RESULT, _EMPTY_RESULT, ""

    top_sp = max(species_dict, key=species_dict.get)
    sp_bars = "".join(
        _conf_bar(lbl, p, "#e07b39")
        for lbl, p in sorted(species_dict.items(), key=lambda x: -x[1])
    )
    species_html = (
        f'<div class="result-card">'
        f'<div class="r-lbl">Plant Species</div>'
        f'<div class="r-top">{top_sp}</div>'
        f'{sp_bars}'
        f'</div>'
    )

    top_h = max(health_dict, key=health_dict.get)
    h_color = "#22c55e" if top_h == "Healthy" else "#ef4444"
    h_bars = "".join(
        _conf_bar(lbl, p, "#22c55e" if lbl == "Healthy" else "#ef4444")
        for lbl, p in sorted(health_dict.items(), key=lambda x: -x[1])
    )
    health_html = (
        f'<div class="result-card">'
        f'<div class="r-lbl">Health Status</div>'
        f'<div class="r-top" style="color:{h_color}">{top_h}</div>'
        f'{h_bars}'
        f'</div>'
    )

    action_html = _ANTHRACNOSE_ACTIONS if top_h == "Anthracnose" else ""

    return species_html, health_html, action_html


# ── Gradio interface ──────────────────────────────────────────────────────────

_theme = gr.themes.Base(primary_hue="orange", neutral_hue="slate")

_SAMPLES = [
    ["Sample_Images/Guava_Healthy_leaf.jpg"],
    ["Sample_Images/Guava_Anthracnose_leaf.jpg"],
    ["Sample_Images/Mango_Healthy_leaf.jpeg"],
    ["Sample_Images/Mango_Anthracnose_leaf.jpg"],
    ["Sample_Images/Papaya_Healthy_leaf.jpg"],
    ["Sample_Images/Papaya_Anthracnose_leaf.jpg"],
]

with gr.Blocks(title="Anthracnose Detector", css=CSS, theme=_theme) as demo:

    # ── Header ───────────────────────────────────────────────────────────────
    gr.HTML("""
    <div class="app-header">
      <h1>🌿 Anthracnose Detector</h1>
      <p>
        A multi-task deep learning system for simultaneous
        <strong>plant species identification</strong> and
        <strong>anthracnose disease detection</strong> from leaf photographs.
        Supports Guava, Mango, and Papaya.
      </p>
    </div>
    """)

    # ── Main two-column layout ────────────────────────────────────────────────
    with gr.Row(equal_height=False):

        # Left column: image input + predict button + how to use
        with gr.Column(scale=1, min_width=320):
            gr.HTML("""
            <p style="
                font-family:'Inter',system-ui,sans-serif;
                font-size:0.82rem; font-weight:600;
                text-transform:uppercase; letter-spacing:0.1em;
                opacity:0.45; margin:0 0 0.45rem 0.1rem;
            ">Capture or upload a leaf image to see the results</p>
            """)
            img_input = gr.Image(type="pil", label="Leaf Image", height=290)
            predict_btn = gr.Button("Predict", elem_id="predict-btn", variant="primary")
            gr.HTML("""
            <div class="info-card" style="margin-top:0.85rem">
              <div class="card-lbl">How to Use</div>
              <div class="steps">
                <ol>
                  <li><strong>Upload</strong> a clear, well-lit photo of a single leaf, or pick a sample below.</li>
                  <li>Click <strong>Predict</strong>, or results appear automatically on upload.</li>
                  <li>Read the <strong>Plant Species</strong> and <strong>Health Status</strong> panels on the right.</li>
                  <li>If Anthracnose is detected, a list of <strong>recommended actions</strong> will appear.</li>
                </ol>
              </div>
            </div>
            """)

        # Right column: result panels + conditional action panel
        with gr.Column(scale=1, min_width=320):
            species_out = gr.HTML(_EMPTY_RESULT)
            health_out  = gr.HTML(_EMPTY_RESULT)
            action_out  = gr.HTML("")

    # ── Sample images ─────────────────────────────────────────────────────────
    gr.Examples(
        examples=_SAMPLES,
        inputs=img_input,
        label="Sample Images — click any to load",
    )

    # ── Footer ────────────────────────────────────────────────────────────────
    gr.HTML("""
    <div class="app-footer">
      AnthraNet &nbsp;·&nbsp; Leaf Species &amp; Anthracnose Classification
      &nbsp;·&nbsp; Guava · Mango · Papaya
    </div>
    """)

    predict_btn.click(fn=predict_html, inputs=img_input, outputs=[species_out, health_out, action_out])
    img_input.change(fn=predict_html, inputs=img_input, outputs=[species_out, health_out, action_out])

if __name__ == "__main__":
    demo.launch()
