# 🔍 Complete Deepfake Detection Suite

**3 State-of-the-Art Models + Ensemble Detector**

This package contains standalone inference scripts for three top-performing deepfake detection models, plus an ensemble detector that combines all three for maximum accuracy.

---

## 📦 **What's Included**

1. **fsfm_unified_detector.py** - FSFM-3C (4-class ViT)
2. **cemroot_detector.py** - CemRoot (EfficientNetB7 + Attention)
3. **vit_detector.py** - ViT-v2 (prithivMLmods)
4. **ensemble_detector.py** - Combines all 3 models
5. **models_vit.py** - Required for FSFM-3C
6. **README.md** - This file

---

## 🚀 **Quick Start Guide**

### **1. Install Dependencies**
```bash
# PyTorch models (FSFM, ViT)
pip install torch torchvision transformers timm pillow

# TensorFlow model (CemRoot)
pip install tensorflow keras opencv-python numpy
```

### **2. Download Model Files**

**Option A: Easiest (ViT-v2 only)**
```bash
python vit_detector.py --image test.jpg  # Auto-downloads!
```

**Option B: Full Setup (All models)**
Download manually from HuggingFace or use the download script below.

---

## 💻 **Individual Model Usage**

### **Model 1: ViT-v2 (EASIEST!)**
✅ No downloads needed - auto-downloads from HuggingFace

```bash
python vit_detector.py --image test.jpg
```

### **Model 2: FSFM-3C (4-class detection)**
Files needed: checkpoint-min_train_loss.pth + pretrain_ds_mean_std.txt

```bash
python fsfm_unified_detector.py \
  --image test.jpg \
  --checkpoint checkpoint-min_train_loss.pth \
  --mean_std pretrain_ds_mean_std.txt
```

### **Model 3: CemRoot (95% accuracy)**
File needed: best_model_effatt.h5

```bash
python cemroot_detector.py \
  --image test.jpg \
  --model best_model_effatt.h5 \
  --method training_match  # CRITICAL for 95% accuracy!
```

### **Ensemble (Best overall)**
Needs all model files

```bash
python ensemble_detector.py \
  --image test.jpg \
  --fsfm_checkpoint checkpoint-min_train_loss.pth \
  --fsfm_mean_std pretrain_ds_mean_std.txt \
  --cemroot_model best_model_effatt.h5 \
  --voting weighted
```

---

## 📊 **Which Model Should I Use?**

| Use Case | Recommended Model |
|----------|------------------|
| **Easy setup** | ViT-v2 (auto-downloads) |
| **Maximum accuracy** | Ensemble (weighted voting) |
| **Detailed classification** | FSFM-3C (4 classes) |
| **Production deployment** | CemRoot (training_match) |
| **Fast inference** | ViT-v2 |

---

## 🎯 **Model Details**

### **FSFM-3C: 4-Class Detection**
- **Classes:** Real, Deepfake, Diffusion/AIGC, Spoofing
- **Best for:** Identifying specific threat types
- **Speed:** Fast (~50ms)

### **CemRoot: 95% Accuracy**
- **Classes:** Real, Fake
- **Best for:** Maximum accuracy (with correct preprocessing!)
- **⚠️ CRITICAL:** Use `--method training_match` for 95% accuracy
    - training_match: 95% ✅
    - simple_norm: 58% ❌
    - efficientnet: 72% ⚠️

### **ViT-v2: Easiest Setup**
- **Classes:** Realism, Deepfake
- **Best for:** Quick deployment
- **✨ Feature:** Auto-downloads, no manual setup

### **Ensemble: Maximum Robustness**
- **Strategy:** Combines all 3 models
- **Voting:** Majority, Weighted (recommended), Unanimous
- **Best for:** Production systems

---

## 📥 **Downloading Model Files**

### **Quick Download Script**

```python
from huggingface_hub import hf_hub_download

# FSFM-3C files
hf_hub_download(
    repo_id="Wolowolo/fsfm-3c",
    filename="finetuned_models/Unified-detector/v1_Fine-tuned_on_4_classes/checkpoint-min_train_loss.pth",
    local_dir="./models/fsfm"
)
hf_hub_download(
    repo_id="Wolowolo/fsfm-3c",
    filename="finetuned_models/Unified-detector/v1_Fine-tuned_on_4_classes/pretrain_ds_mean_std.txt",
    local_dir="./models/fsfm"
)

# CemRoot file
hf_hub_download(
    repo_id="CemRoot/deepfake-detection-model",
    filename="best_model_effatt.h5",
    local_dir="./models/cemroot"
)

# ViT-v2: No download needed (auto-downloads)
```

---

## 🔧 **Common Issues & Solutions**

### **Issue:** CemRoot has low accuracy (~58%)
**Solution:** Use `training_match` preprocessing!
```bash
--method training_match  # 95% accuracy ✅
```

### **Issue:** `ModuleNotFoundError: No module named 'models_vit'`
**Solution:** Ensure models_vit.py is in the same directory
```bash
ls models_vit.py  # Should exist
```

### **Issue:** First run is slow
**Solution:** ViT-v2 downloads ~350MB on first run (then cached)

### **Issue:** Out of memory
**Solution:** Use CPU instead
```bash
--device cpu
```

---

## 📈 **Performance Comparison**

| Model | Accuracy | Speed | Memory | Setup Difficulty |
|-------|----------|-------|--------|-----------------|
| FSFM-3C | High | 50ms | 2GB | Medium |
| CemRoot | 95%* | 100ms | 4GB | Medium |
| ViT-v2 | 92% | 40ms | 1.5GB | **Easy** |
| Ensemble | Highest | 190ms | 7.5GB | Hard |

*With training_match preprocessing

---

## 💡 **Usage Examples in Python**

### **Quick Test (ViT-v2)**
```python
from vit_detector import DeepFakeDetectorV2

detector = DeepFakeDetectorV2()  # Auto-downloads
result = detector.predict('image.jpg')
print(f"{result['predicted_label']}: {result['confidence']*100:.1f}%")
```

### **4-Class Detection (FSFM-3C)**
```python
from fsfm_unified_detector import FSFM_UnifiedDetector

detector = FSFM_UnifiedDetector(
    checkpoint_path='checkpoint-min_train_loss.pth',
    mean_std_path='pretrain_ds_mean_std.txt'
)

result = detector.predict('image.jpg', return_all_probs=True)

# Shows: Real, Deepfake, Diffusion, or Spoofing
print(result['predicted_label'])
print(result['all_probabilities'])
```

### **Maximum Accuracy (CemRoot)**
```python
from cemroot_detector import CemRootDetector

detector = CemRootDetector(model_path='best_model_effatt.h5')

# CRITICAL: Use training_match for 95% accuracy!
result = detector.predict('image.jpg', method='training_match')
print(f"{result['predicted_label']}: {result['confidence']*100:.1f}%")
```

### **Ensemble (Best Overall)**
```python
from ensemble_detector import EnsembleDeepfakeDetector

ensemble = EnsembleDeepfakeDetector(
    fsfm_config={'checkpoint': '...', 'mean_std': '...'},
    cemroot_config={'model_path': '...'},
    vit_config={}
)

result = ensemble.predict('image.jpg', voting_strategy='weighted')
print(f"Ensemble: {result['ensemble_prediction']}")
print(f"Confidence: {result['ensemble_confidence']*100:.1f}%")
print(f"All agree: {result['agreement']['all_agree']}")
```

---

## 📝 **Model Sources**

- **FSFM-3C:** https://huggingface.co/Wolowolo/fsfm-3c
- **CemRoot:** https://huggingface.co/CemRoot/deepfake-detection-model
- **ViT-v2:** https://huggingface.co/prithivMLmods/Deep-Fake-Detector-v2-Model

---

## ✨ **Key Takeaways**

1. **Start simple:** Try ViT-v2 first (easiest setup)
2. **Need accuracy:** Use CemRoot with `training_match`
3. **Need details:** Use FSFM-3C for 4-class classification
4. **Production use:** Use Ensemble with weighted voting
5. **⚠️ CRITICAL:** CemRoot preprocessing matters (95% vs 58%)!

---

**Ready to detect deepfakes? Pick a model and start testing! 🚀**