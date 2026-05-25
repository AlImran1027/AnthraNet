# AnthraNet — A Segmentation-Independent Deep Learning Framework for Multi-Species Anthracnose Diagnosis

> Code, notebooks, and checkpoints accompanying the paper
> **"A Segmentation-Independent Deep Learning Framework for Multi-Species Anthracnose Diagnosis"**
> Al Imran, Md. Mazharul Islam Nibir, Md. Ashrif Rahman Arian, Safayet Ullah Neyam, Jawad Ibn Ahad, Sifat Momen.
> Department of Electrical & Computer Engineering, North South University, Dhaka, Bangladesh.

**AnthraNet** is a leaf-agnostic, segmentation-independent framework for diagnosing **anthracnose** disease across **Guava, Mango, and Papaya** from raw RGB leaf images. It combines a **Swin-Transformer-Large** backbone with a **Feature Pyramid Network (FPN)** for multi-scale lesion modelling and a **Global Context Attention (GCA)** module for background suppression, trained under a shared multi-task objective that jointly predicts host species and disease status. On the combined three-species dataset it reaches **99.74 ± 0.06 % accuracy** (macro-F1 99.72 %, ROC-AUC 99.95 %) — improving the standalone Swin-Large baseline by 1.95 pp and the strongest related prior work by 1.32 pp.

---

## Table of contents
- [Key results](#key-results)
- [Architecture (AnthraNet)](#architecture-anthranet)
- [Dataset](#dataset)
- [Repository layout](#repository-layout)
- [Getting started](#getting-started)
- [Running the notebooks](#running-the-notebooks)
- [Configuration](#configuration)
- [Best checkpoint](#best-checkpoint)
- [Cross-dataset (leave-one-species-out) evaluation](#cross-dataset-leave-one-species-out-evaluation)
- [Feature importance analysis](#feature-importance-analysis)
- [Limitations](#limitations)

---

## Key results

### Combined three-species dataset (Mango + Guava + Papaya)

| Metric | Value |
|---|---:|
| Accuracy | **99.74 ± 0.06 %** |
| Macro F1 | **99.72 ± 0.06 %** |
| Macro precision | 99.79 ± 0.07 % |
| Macro recall | 99.65 ± 0.07 % |
| ROC-AUC | 99.95 ± 0.31 % |

### Per-species

| Leaf | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Mango only | 99.34 | 99.35 | 99.35 | 99.34 |
| Guava only | 99.46 | 99.68 | 98.31 | 98.98 |
| Papaya only | 99.28 | 99.58 | 98.10 | 98.58 |

### Component ablation (Swin-Large fixed backbone)

| Configuration | FPN ch. | GCA red. | Accuracy | F1 |
|---|---:|---:|---:|---:|
| Baseline Swin-Large | – | – | 97.79 ± 0.32 | 98.20 ± 0.38 |
| + GCA only | – | 16 | 99.47 ± 0.67 | 99.44 ± 0.98 |
| + FPN only | 256 | – | 99.64 ± 0.04 | 99.62 ± 0.98 |
| + FPN + GCA | 256 | 16 | 99.51 ± 0.76 | 99.58 ± 0.44 |
| + FPN + GCA | 512 | 32 | 99.61 ± 0.78 | 99.58 ± 0.62 |
| **+ FPN + GCA (best — final AnthraNet)** | **128** | **8** | **99.74 ± 0.06** | **99.72 ± 0.06** |

Larger FPN channel widths did not improve performance — excessive feature dimensionality introduces redundant or less stable representations on this dataset.

### Computational efficiency (AnthraNet)

| | |
|---|---|
| Total parameters | 196.14 M |
| FLOPs | 42.7 G |
| Mean inference latency | 15.00 ms (σ = 2.18 ms, CUDA) |
| Throughput | 66.7 images / s |
| Model size on disk | 786.76 MB |

---

## Architecture (AnthraNet)

`AnthraNet = Swin-Transformer-Large + FPN + GCA`, with two task-specific heads sharing one representation.

```
                       ┌────────────────────────────────────────────┐
   x (3×224×224)  ──▶  │  Swin-Transformer-Large (timm, pretrained) │  → {C2, C3, C4, C5}
                       └────────────────────────────────────────────┘
                                            │
                                            ▼
                       ┌────────────────────────────────────────────┐
                       │  Feature Pyramid Network                    │  → {P2, P3, P4, P5}
                       │   Pℓ = Conv1×1(Cℓ) + Upsample(Pℓ₊₁)        │
                       └────────────────────────────────────────────┘
                                            │
                                            ▼
                       ┌────────────────────────────────────────────┐
                       │  Global Context Attention (multi-head)      │
                       │   Y = X + Wv₂ · ReLU( LN( Wv₁ · Σⱼ αⱼxⱼ ))  │
                       └────────────────────────────────────────────┘
                                            │
                                            ▼
                       Feature fusion ─▶ GlobalAvgPool ─▶ Dropout ─▶ ┐
                                                                      │
                                       ┌──────────────────────────────┤
                                       ▼                              ▼
                            Species head (3 classes)        Disease head (2 classes)
                                 softmax                         softmax / σ
```

**Multi-task objective:** `L_total = λ_s · L_species + λ_d · L_disease`, with `λ_s = 0.4`, `λ_d = 0.6`. `L_species` is categorical cross-entropy with label smoothing 0.1; `L_disease` is class-weighted binary cross-entropy. The species head acts as an auxiliary constraint that encourages the shared representation to remain host-aware.

**Why this combination?** Anthracnose lesions appear at many scales (small dark spots through to large necrotic patches) and against highly variable backgrounds (soil, shadows, overlapping leaves). The hierarchical Swin backbone supplies multi-stage feature maps that connect naturally to an FPN; the FPN preserves fine lesion cues alongside high-level disease semantics; and the GCA module recalibrates the fused representation using global spatial context so the model does not learn shortcuts from background patterns.

---

## Dataset

5,072 RGB images compiled from **five public sources**, covering three tropical fruit crops × {Healthy, Anthracnose}. Stratified 70 / 15 / 15 split (joint Species×Disease label) → 3,550 train / 761 val / 761 test, with a fixed seed for reproducibility.

| Class | Species | Health | N | Median resolution |
|---|---|---|---:|---|
| Guava_Anthracnose | Guava | Anthracnose | 237 | 4000×4000 |
| Guava_Healthy | Guava | Healthy | 1,248 | 3468×4032 |
| Mango_Anthracnose | Mango | Anthracnose | 1,100 | 512×512 |
| Mango_Healthy | Mango | Healthy | 1,100 | 512×512 |
| Papaya_Anthracnose | Papaya | Anthracnose | 585 | 640×480 |
| Papaya_Healthy | Papaya | Healthy | 802 | 640×480 |
| **Total** | — | — | **5,072** | — |

**Class-imbalance handling** (used together): `WeightedRandomSampler`, class-weighted loss functions, and Mixup / CutMix augmentation (`MIXUP_ALPHA=0.2`, `CUTMIX_ALPHA=1.0`, `MIXUP_PROB=0.3`). The Guava subset is the most imbalanced (5.27× ratio between Guava_Healthy and Guava_Anthracnose).

**Quality controls applied during dataset construction.**
- All images resized to 224×224 and normalised with ImageNet mean / std.
- 1,410 blurry images (Laplacian variance < 50, mostly Mango field captures) and 103 over-exposed images flagged but retained for robustness.
- 436 exact duplicates (8.6 %) identified via difference-hashing (dHash) — concentrated in Mango_Anthracnose where filenames contained "Copy" / "Copy - Copy" suffixes — were de-duplicated and stratified splitting was used to suppress data-leakage risk.

Source data lives in `Dataset_raw/`, organised as `Dataset_raw/{Species}_{Class}/image.jpg`. Full EDA (resolution, sharpness, brightness, HSV / LAB colour distributions, statistical separability, split verification) is in [EDA_Anthracnose_Dataset.ipynb](EDA_Anthracnose_Dataset.ipynb); the underlying figures live in [EDA_Figures/](EDA_Figures/) and the summary table in [EDA_Figures/dataset_summary_table.csv](EDA_Figures/dataset_summary_table.csv).

---

## Repository layout

```
.
├── Dataset_raw/                  Raw images, grouped as {Species}_{Class}/
├── EDA_Anthracnose_Dataset.ipynb Exploratory data analysis
├── EDA_Figures/                  EDA outputs (PDF + PNG figures, CSV table)
│
├── Done/                         Initial CNN ablation notebooks (canonical template:
│                                 anthracnose_resnet50.ipynb) — ResNet50, ResNet101,
│                                 DenseNet121, EfficientNetV2-Small, ShuffleNetV2.
├── CNN_models+FPN+GCA/           Final CNN+FPN+GCA training notebooks:
│                                   - DenseNet121 / DenseNet201
│                                   - EfficientNet-B0 / EfficientNet-B4
│                                   - MobileNet-V2 / MobileNet-V3
│                                   - ResNet50
│
├── Vit_Models/                   ViT / DeiT / Swin baselines (multi-task, no FPN/GCA):
│                                 ViT-Base / Vit-large, DeiT-Small / DeiT-base / DeiT-Large,
│                                 Swin-Tiny / Swin-Base / Swin-Large.
├── ViT_with_FPN+GCA/             First-pass combination of the ViT family with FPN + GCA.
├── Refined_ViT_wit_FPN+GCA/      Refined ViT family with FPN + GCA used in the main study.
│
├── Ablation_fIles/               Swin-Large ablation pieces:
│                                   - Swin-Large_Ablation.ipynb     (full FPN+GCA)
│                                   - Swin-Large_backbone.ipynb     (backbone only)
│                                   - Swin-Large_onlyFPN.ipynb      (FPN only)
│                                   - Swin-Large_onlyGCA.ipynb      (GCA only)
│                                   - Swin-Large_PerSpecies_Test.ipynb
├── Ablation_results/             Outputs at three FPN-channels / GCA-reduction settings:
│                                   - 128,8/   (paper's winning configuration)
│                                   - 256,16/  (full XAI outputs + best_model.pt arch.)
│                                   - 512,32/
│
├── Cross-dataset_evaluation/     Leave-one-species-out generalisation tests:
│                                   - Swin-Large_cross-test_M+G_test-P.ipynb
│                                   - Swin-Large_cross-test_M+P_test-G.ipynb
│                                   - Swin-Large_cross-test_P+G_test-M.ipynb
├── Cross_dataset_results/        Per-fold result notebooks.
│
├── FPN+GCA models/               Intended output directory for saved checkpoints.
├── best_model.pt                 Swin-Large + FPN + GCA checkpoint (~795 MB).
│                                 Architecture: 256-channel FPN, GCA reduction 16
│                                 (i.e. the 256/16 ablation variant; the paper's
│                                 final 128/8 model achieves marginally higher accuracy).
│
├── Augmentation_Visualization.ipynb  Visualises individual + composite augmentations.
├── Feature_Importance_Analysis.ipynb Feature importance for best_model.pt (see below).
├── feature_importance_outputs/   Figures from the feature-importance notebook.
│
├── Anthracnose_final_draft.pdf   Manuscript draft.
├── requirements.txt
├── CLAUDE.md                     Contributor guidance.
└── README.md
```

---

## Getting started

The project uses a local virtual environment at `.venv/`.

```bash
# Create the venv (only needed the first time)
python3 -m venv .venv

# Install dependencies
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

# Launch Jupyter
.venv/bin/jupyter notebook
```

Dependencies (see [requirements.txt](requirements.txt)) cover PyTorch + torchvision, `timm` (Swin / DeiT / ViT backbones), `albumentations`, `opencv-python-headless`, the scientific Python stack (numpy / pandas / scikit-learn / scikit-image), matplotlib / seaborn, tqdm, lime, and the Jupyter runtime. Tested on Python 3.13 with PyTorch 2.12 (CUDA 11.8 / 12.x on RunPod; also runs on Apple Silicon via MPS).

---

## Running the notebooks

Each notebook is self-contained — it imports its dependencies, builds the model, loads data, trains, evaluates, and writes outputs into its own directory. There is no shared library code.

```bash
# Interactive
.venv/bin/jupyter notebook Vit_Models/Swin-Large.ipynb
.venv/bin/jupyter notebook "Ablation_results/128,8/Swin-Large_128_and_8.ipynb"
.venv/bin/jupyter notebook Cross-dataset_evaluation/Swin-Large_cross-test_M+G_test-P.ipynb

# Headless (run end-to-end and save outputs back into the notebook)
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
    "Ablation_results/128,8/Swin-Large_128_and_8.ipynb"
```

Adjust `DATA_ROOT` (default `/workspace/Dataset_raw` on the original RunPod training host) at the top of any notebook to point at the local `Dataset_raw/` folder before running.

---

## Configuration

Every notebook starts with a `Config` class that centralises hyperparameters. Final values reported in the paper (Table 3):

| Parameter | Value |
|---|---|
| `IMG_SIZE` | 224 |
| `BATCH_SIZE` | 24 |
| `LEARNING_RATE` | 1e-4 (searched over {1e-3, 1e-4, 1e-5}) |
| `WEIGHT_DECAY` | 1e-5 (searched over {1e-4, 1e-5, 1e-6}) |
| `OPTIMIZER` | AdamW (searched over {Adam, AdamW}) |
| `WEIGHT_INIT` | Orthogonal (searched over {Xavier, Xavier-normal, Orthogonal, Kaiming}) |
| `WARMUP_EPOCHS` | 5 (linear warm-up + cosine annealing) |
| `EPOCHS` | 100 (max) |
| `PATIENCE` (early stopping) | 15 |
| `LABEL_SMOOTHING` | 0.1 |
| `FPN_CHANNELS` | 128 (best — paper's final AnthraNet) |
| `GCA_REDUCTION` | 8 (best — paper's final AnthraNet) |
| `DROPOUT_RATE` | 0.3 (CNN) · 0.4 (Swin-Large) |
| `MIXUP_ALPHA` / `CUTMIX_ALPHA` / `MIXUP_PROB` | 0.2 / 1.0 / 0.3 |
| Loss weights (λ_s, λ_d) | 0.4, 0.6 |
| Split | 70 / 15 / 15 (stratified on Species × Disease) |
| Normalisation | ImageNet mean / std |

**Augmentation pipeline** (Albumentations, training only): Random Rotation ±30° (reflective padding), HFlip (p=0.5), VFlip (p=0.25), Random Resized Crop (scale 0.8–1.0, ratio 0.9–1.1), Brightness ±0.2, Contrast ±0.2, Colour Jitter (B 0.1, C 0.1, S 0.2, H 0.1), Gaussian Blur (kernel 3–7, p=0.15), Random Shadow (1–2 shadows), CLAHE (clip 4.0, tile 8×8), Coarse Dropout (≤ 8 holes), plus Mixup and CutMix as composite two-sample augmentations.

Each ablation cell in [Ablation_results/](Ablation_results/) sweeps `FPN_CHANNELS` and `GCA_REDUCTION` over three settings: **128 / 8 (winning)**, 256 / 16, and 512 / 32.

---

## Best checkpoint

`best_model.pt` (~795 MB) holds Swin-Large + FPN + GCA weights from the **256 / 16** ablation variant. It is a `dict` containing the `model` state-dict and loads directly with the model class defined in [Ablation_results/256,16/Swin-Large_Ablation (1).ipynb](Ablation_results/256,16/Swin-Large_Ablation%20%281%29.ipynb):

```python
import torch
model = MultiTaskSwinLarge(num_species=3, num_health=2, pretrained=True, dropout=0.4).to(device)
ckpt  = torch.load("best_model.pt", map_location=device, weights_only=False)
model.load_state_dict(ckpt["model"])
model.eval()
```

The paper's headline 99.74 % result corresponds to the **128 / 8** configuration ([Ablation_results/128,8/Swin-Large_128_and_8.ipynb](Ablation_results/128,8/Swin-Large_128_and_8.ipynb)); the 256 / 16 variant — which the bundled `best_model.pt` realises — sits about 0.2 pp behind on macro accuracy (99.51 ± 0.76 %) but retains essentially identical qualitative behaviour for explainability and feature-importance analyses.

CNN notebooks additionally implement resumable training via `find_and_load_checkpoint()` and `resume_training_from_checkpoint()`; checkpoints are written every epoch alongside a JSON config file.

---

## Cross-dataset (leave-one-species-out) evaluation

To probe whether AnthraNet learns truly species-agnostic representations, three leave-one-out folds were trained on two species and tested on the third without any target-species fine-tuning:

| Train | Test | Acc. (%) | Prec. (%) | Rec. (%) | F1 (%) | Healthy acc. | Anthracnose acc. |
|---|---|---:|---:|---:|---:|---:|---:|
| Mango + Guava | Papaya | 75.78 | 83.75 | 75.78 | 75.36 | 59.10 | 98.63 |
| Mango + Papaya | Guava | 94.34 | 94.49 | 94.34 | 93.90 | 99.60 | 66.67 |
| Papaya + Guava | Mango | 86.82 | 89.57 | 86.82 | 86.59 | 100.00 | 73.64 |
| **Mean** | — | **85.65** | **89.27** | **85.65** | **85.28** | 86.23 | 79.65 |

A clear asymmetry across folds — Guava / Mango folds preserve healthy-leaf classification almost perfectly but lose anthracnose recall, while the Papaya fold loses healthy-leaf accuracy instead — indicates that **transferability of healthy vs. diseased visual signatures is species-dependent**, and is the primary source of cross-dataset variance.

Notebooks: [Cross-dataset_evaluation/](Cross-dataset_evaluation/) (training) and [Cross_dataset_results/](Cross_dataset_results/) (analysis).

---

## Feature importance analysis

[Feature_Importance_Analysis.ipynb](Feature_Importance_Analysis.ipynb) provides three complementary views of what the deployed model actually relies on, computed against the held-out test set with `best_model.pt`:

1. **Permutation importance** on the 256-D pooled FPN embedding — per-channel **ΔNLL** and **Δtrue-class probability** for both the species and health heads.
2. **FPN level importance** — performance drop when each of `P1` (56×56), `P2` (28×28), `P3` (14×14), and `P4` (7×7) is zeroed before fusion. Reports ΔNLL, Δtrue-class probability, and Δaccuracy side-by-side.
3. **Integrated Gradients** — pixel-level attributions per `species × health` combination plus an aggregate test-set map per head. Implemented from scratch with Gaussian smoothing (σ ≈ 6 px) to suppress the Swin window-grid artifact, computed against the softmax probability (not raw logits) to avoid signal loss from the model's near-saturated logits.

> **Note on metrics.** Because the model is essentially saturated on this dataset (Species 100 %, Health ≈ 99.7 % on the bundled checkpoint), accuracy alone is uninformative under most interventions — single-channel permutations and single-level ablations almost never flip an argmax. The notebook therefore reports NLL and true-class probability, which give a continuous signal even when predictions don't change.

All four resulting figures are written to [feature_importance_outputs/](feature_importance_outputs/). For the paper's own GradCAM++ / LIME explainability outputs, see [Ablation_results/256,16/xai_gradcam_lime.png](Ablation_results/256,16/xai_gradcam_lime.png) and Figure 14 of the manuscript.

---

## Limitations

Reproduced from the manuscript discussion (Section 7.2) so users can evaluate the framework's intended scope:

- The dataset was assembled from public sources, not a single controlled field campaign — image-level stratified splits do not fully prove **source-independent** generalisation, since images from the same source may share visual patterns and allow the model to benefit from distributional similarity between training and test.
- The leaf-agnostic claim has so far been validated only on **Mango, Guava, and Papaya**. Other anthracnose-affected hosts (strawberry, citrus, avocado) are planned extensions.
- The model performs **image-level classification only** — no lesion segmentation, no severity / infection-area estimation.
- Swin-Transformer-Large is computationally heavy (~196 M parameters, 786 MB on disk). Direct mobile / edge deployment will need pruning, quantisation, knowledge distillation, or a lighter backbone.

A prototype interface is available as a Hugging Face Space at `AnthraNet.net`, supporting accessibility beyond offline experimentation.

---
