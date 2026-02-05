# 🌿 Anthracnose Disease Classification Pipeline

## CSE499 Senior Design Project

A comprehensive deep learning pipeline for classifying Anthracnose disease in Guava, Mango, and Papaya leaves using:
- **DenseNet201** backbone with **Feature Pyramid Networks (FPN)**
- **Global Context Attention (GCA)** for feature refinement


---

## 📁 Repository Structure

```
project/
├── README.md                           # Project documentation
├── dataset/
│   └── Raw_dataset/
│       ├── Guava_Anthracnose/
│       ├── Guava_Healthy/
│       ├── Mango_Anthracnose/
│       ├── Mango_Healthy/
│       ├── Papaya_Anthracnose/
│       └── Papaya_Healthy/
└── Custom_model/
    ├── anthracnose_training.ipynb      # Training notebook
    ├── anthracnose_evaluation.ipynb    # Evaluation notebook
    └── outputs/                        # Generated during training/evaluation
        ├── checkpoints/
        │   └── best_model.pth
        ├── logs/
        │   └── training_history.json
        └── evaluation_results/
            ├── confusion_matrix_*.png
            ├── training_curves.png
            └── evaluation_report.txt
```

---

## 🔧 Installation

### Required Packages

```bash
# Core deep learning
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Image augmentation and processing
pip install albumentations opencv-python pillow

# Visualization
pip install matplotlib seaborn

# ML utilities
pip install scikit-learn pandas numpy tqdm

# Pretrained models
pip install timm

# Mask R-CNN (optional)
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

### Quick Install (All Packages)

```bash
pip install torch torchvision albumentations opencv-python matplotlib seaborn scikit-learn pillow tqdm pandas numpy timm
```

---

## 🚀 Usage Instructions

### Step 1: Prepare Dataset

Organize your dataset in the following structure:
```
dataset/Raw_dataset/
├── Guava_Anthracnose/    # Guava leaves with anthracnose
├── Guava_Healthy/        # Healthy guava leaves
├── Mango_Anthracnose/    # Mango leaves with anthracnose
├── Mango_Healthy/        # Healthy mango leaves
├── Papaya_Anthracnose/   # Papaya leaves with anthracnose
└── Papaya_Healthy/       # Healthy papaya leaves
```

### Step 2: Create Training Notebook

1. Create a new Jupyter notebook named `training.ipynb`
2. Open `anthracnose_training_pipeline.py`
3. Copy each section marked with `# %%` into separate cells
4. Sections marked with `# %% [markdown]` are markdown cells

### Step 3: Create Evaluation Notebook

1. Create a new Jupyter notebook named `evaluation.ipynb`
2. Open `anthracnose_evaluation_pipeline.py`
3. Copy each section marked with `# %%` into separate cells

### Step 4: Run Training

1. Open `training.ipynb`
2. Run all cells in order
3. When you reach Cell 21, uncomment the training line:
   ```python
   trained_model, metrics = train_model(model, train_loader, val_loader, Config, class_weights, DEVICE)
   ```
4. Training will run for up to 100 epochs (with early stopping)

### Step 5: Run Evaluation

1. Open `evaluation.ipynb`
2. Run all cells in order
3. Results will be saved to `outputs/evaluation_results/`

---

## 🏗️ Architecture Overview

### Model Components

```
Input Image (224×224×3)
         │
         ▼
┌─────────────────────────┐
│   DenseNet201 Backbone  │  ← Pretrained on ImageNet
│   (Feature Extraction)  │
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Feature Pyramid Network │  ← Multi-scale features
│  (FPN with 4 levels)     │     P1, P2, P3, P4
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Global Context Attention│  ← Channel + Spatial attention
│  (Multi-head GCA)        │     for disease regions
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│   Feature Aggregation    │  ← Concatenate + Fuse
│   + Global Pooling       │
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Classification Head     │  ← FC layers with dropout
│  (2 classes)             │
└─────────────────────────┘
         │
         ▼
    [Healthy, Anthracnose]
```

### Key Features

| Feature | Description |
|---------|-------------|
| **DenseNet201** | Dense connections for feature reuse |
| **FPN** | Multi-scale features for varying lesion sizes |
| **Multi-head GCA** | 4-head attention for comprehensive feature refinement |
| **Label Smoothing** | Prevents overconfidence (0.1 smoothing) |
| **Mixup/Cutmix** | Data augmentation for regularization |
| **Class Weighting** | Handles class imbalance |
| **Warmup + Cosine LR** | Stable training with proper LR scheduling |
| **Early Stopping** | Prevents overfitting (patience=15) |

---

## 📊 Training Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `IMG_SIZE` | 224 | Input image size |
| `BATCH_SIZE` | 16 | Batch size |
| `LEARNING_RATE` | 1e-4 | Initial learning rate |
| `NUM_EPOCHS` | 100 | Maximum epochs |
| `WARMUP_EPOCHS` | 5 | LR warmup period |
| `PATIENCE` | 15 | Early stopping patience |
| `LABEL_SMOOTHING` | 0.1 | Label smoothing factor |
| `MIXUP_ALPHA` | 0.2 | Mixup alpha parameter |
| `DROPOUT_RATE` | 0.3 | Dropout probability |

---

## 🔄 Augmentation Pipeline

Applied **only during training**:

1. **Random Rotation** (±30°)
2. **Horizontal/Vertical Flip**
3. **Brightness/Contrast Jitter**
4. **Random Resized Crop** (0.8-1.0 scale)
5. **Color Jitter** (Hue, Saturation)
6. **Gaussian Blur** (optional)
7. **Coarse Dropout** (Cutout)
8. **CLAHE** (Contrast enhancement)
9. **Random Shadow** (Outdoor simulation)

---



## 📁 Output Files

### After Training

```
outputs/
├── checkpoints/
│   ├── best_model.pth          # Best model checkpoint
│   └── checkpoint_epoch_*.pth  # Periodic checkpoints
├── logs/
│   └── training_history.json   # Training metrics
├── train_split.csv             # Training set indices
├── val_split.csv               # Validation set indices
├── test_split.csv              # Test set indices
├── class_distribution.png      # Dataset visualization
├── augmentation_visualization.png
└── individual_augmentations.png
```

### After Evaluation

```
outputs/evaluation_results/
├── evaluation_metrics.json     # All metrics as JSON
├── detailed_predictions.csv    # Per-sample predictions
├── error_analysis.csv          # Misclassified samples
├── evaluation_report.txt       # Text summary
├── confusion_matrix_counts.png
├── confusion_matrix_normalized.png
├── confusion_matrices_per_plant.png
├── roc_pr_curves.png
├── training_curves.png
├── error_analysis.png
├── misclassified_samples.png
└── attention_maps.png
```

---

## 🔍 Evaluation Metrics

### Confusion Matrix
- Raw counts
- Normalized (percentages)
- Per-plant breakdown

### Classification Report
- Precision, Recall, F1 per class
- Support (sample counts)
- Macro/Weighted averages

### Curves
- ROC curve with AUC
- Precision-Recall curve with AP
- Training/Validation loss curves
- F1 score curves
- Learning rate schedule

### Error Analysis
- Misclassified sample visualization
- Error distribution by plant
- Error distribution by class
- Confidence distribution of errors

---

## 🛠️ Customization

### Change Number of Classes

```python
# In Config class
NUM_CLASSES = 3  # For multi-class
CLASSES = ['Healthy', 'Mild', 'Severe']
```

### Adjust Model Complexity

```python
# In Config class
FPN_CHANNELS = 128  # Lighter model
GCA_REDUCTION = 8   # More attention capacity
DROPOUT_RATE = 0.5  # More regularization
```

### Modify Augmentation

```python
# In AugmentationPipeline.get_train_transforms()
A.Rotate(limit=45, p=0.7),  # Stronger rotation
A.GaussNoise(var_limit=(10, 50), p=0.3),  # Add noise
```

---

## ⚠️ Troubleshooting

### CUDA Out of Memory
```python
# Reduce batch size
Config.BATCH_SIZE = 8

# Or use gradient accumulation
accumulation_steps = 2
```

### Slow Training
```python
# Reduce workers if on Windows
num_workers = 0  # In DataLoader

# Or use smaller image size
Config.IMG_SIZE = 192
```

### Model Not Converging
```python
# Try different learning rate
Config.LEARNING_RATE = 3e-4

# Or use more warmup
Config.WARMUP_EPOCHS = 10
```

---

## 📚 References

- DenseNet: [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)
- FPN: [Feature Pyramid Networks](https://arxiv.org/abs/1612.03144)
- Attention: [CBAM](https://arxiv.org/abs/1807.06521), [SE-Net](https://arxiv.org/abs/1709.01507)
- Mixup: [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412)
- CutMix: [CutMix: Regularization Strategy](https://arxiv.org/abs/1905.04899)

---

