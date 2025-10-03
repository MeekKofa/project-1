# Dataset-Specific Configuration Recommendations

## Based on Deep Analysis Results

---

## 📊 cattlebody (raw/processed)

### Dataset Characteristics

- **Classes:** 1 (Cattlebody)
- **Objects per image:** ~1 (mean: 1.01)
- **Original resolution:** 720×480 to 4000×3000 (mostly 2048×1363)
- **Aspect ratios:** 0.75 to 2.19 (large variation!)
- **Bbox size:** Mean area ~10% of image (relatively large)

### ⚠️ Issues

- **train split:** Image/label mismatch (3424 images, 3432 labels)
- Large aspect ratio variation

### ✅ Recommended Config

```yaml
dataset:
  name: cattlebody
  split: raw # Use preprocessing to fix mismatch!

preprocess:
  enabled: true
  target_size: [640, 640] # ✅ 640 is fine - single large object
  maintain_aspect: true # ✅ CRITICAL - large aspect ratio variation

augmentation:
  enabled: true
  horizontal_flip: 0.5 # ✅ Cattle can face either direction
  vertical_flip: 0.0 # ❌ Don't flip vertically
  rotation: 10 # ✅ Slight rotation okay
  brightness: 0.2 # ✅ Recommended by analysis
  contrast: 0.2 # ✅ Recommended by analysis
  mosaic: true # ✅ Even for single object, helps generalization

loss:
  type: standard # ✅ Single class, no imbalance

train:
  batch_size: 8 # ✅ Can use larger batch with single object
  learning_rate: 0.001 # ✅ Standard
```

### 🔧 Action Required

**MUST preprocess first to fix mismatch:**

```bash
python preprocess_dataset.py --dataset cattlebody --split raw
```

---

## 📊 cattle (raw/processed)

### Dataset Characteristics

- **Classes:** 2 (heavy imbalance 10.40:1)
- **Objects per image:** ~5 (mean: 4.91), up to 10
- **Original resolution:** 886×592 to 2048×1810 (mostly 2560×1440)
- **Aspect ratios:** 0.75 to 1.91
- **Bbox size:** Mean area ~3% (SMALL objects!)

### ⚠️ Issues

- **Class imbalance:** 10.40:1 ratio
- **Small objects:** Many tiny bboxes
- Large aspect ratio variation

### ✅ Recommended Config

```yaml
dataset:
  name: cattle
  split: processed  # or raw, both work

preprocess:
  enabled: true
  target_size: [1280, 1280]      # 🔥 HIGHER for small objects!
  maintain_aspect: true          # ✅ CRITICAL

augmentation:
  enabled: true
  horizontal_flip: 0.5
  vertical_flip: 0.0
  rotation: 10
  brightness: 0.2
  contrast: 0.2
  mosaic: true                   # 🔥 CRITICAL for multi-object scenes!
  scale: [0.8, 1.2]             # ✅ Scale augmentation helps

loss:
  type: focal                    # 🔥 CRITICAL for class imbalance!
  focal_alpha: 0.25
  focal_gamma: 2.0
  # OR
  type: weighted
  class_weights: auto            # Computes from dataset

train:
  batch_size: 4                  # ⬇️ Reduce for 1280 resolution
  learning_rate: 0.001
```

### 🎯 Why Higher Resolution?

Your analysis explicitly recommends:

> "Use small anchor sizes for tiny objects"
> "Consider using higher resolution (e.g., 1280x1280)"

With 1280×1280:

- Small objects become 4× larger (area)
- Better bbox precision
- More features for detection

### 💡 Advanced: Two-Stage Training

```bash
# Stage 1: Train at 640 for speed
# config: target_size: [640, 640], epochs: 50

# Stage 2: Fine-tune at 1280 for accuracy
# config: target_size: [1280, 1280], epochs: 30, resume: stage1_best.pt
```

---

## 📊 cattleface (processed)

### Dataset Characteristics

- **Classes:** ❌ 0 - NO LABELS!
- **Original resolution:** All 224×224 (already preprocessed)
- **Objects per image:** 0 (no labels)

### ⚠️ Issues

- **CRITICAL:** No label files found!
  - train: 4565 images, 0 labels
  - val: 979 images, 0 labels
  - test: 984 images, 0 labels

### ❌ Cannot Use for Training

**Solutions:**

1. Find original raw dataset with labels
2. Check if labels are in different format/location
3. Re-annotate using tools like CVAT, LabelImg
4. Use for inference only (if you have trained model)

---

## 🎯 Quick Reference Table

| Dataset    | Resolution | Batch Size | Loss Type      | Priority Augmentation |
| ---------- | ---------- | ---------- | -------------- | --------------------- |
| cattlebody | 640×640    | 8-16       | standard       | mosaic, flip          |
| cattle     | 1280×1280  | 4-8        | focal/weighted | mosaic, scale, flip   |
| cattleface | ❌ N/A     | N/A        | N/A            | Cannot use            |

---

## 🔬 Why NOT 224×224?

### Classification vs Detection

```
Classification (224×224):
Input: Full image
Output: Single label
Example: "This is a cow" ✅

Detection (640+ ×640+):
Input: Full image
Output: Multiple [bbox, class, conf]
Example:
  - [120, 45, 300, 200] → "cow", 0.95
  - [350, 100, 450, 250] → "cattle_tag", 0.88
  - [100, 300, 250, 450] → "cow", 0.92
```

### Impact of Resolution

| Resolution | Use Case              | Pros           | Cons                     |
| ---------- | --------------------- | -------------- | ------------------------ |
| 224×224    | ❌ Detection          | Fast           | Too small, loses details |
| 640×640    | ✅ Standard Detection | Good balance   | May miss small objects   |
| 1280×1280  | ✅ Small Objects      | Best accuracy  | Slower, more memory      |
| 1920×1920  | ✅ Tiny Objects       | Maximum detail | Very slow, high memory   |

### Real Example from Your Data

**Original:** 2560×1440 image with bbox area 3% (~110×43 pixels)

| Target Size | Bbox Size After Resize | Detection Quality      |
| ----------- | ---------------------- | ---------------------- |
| 224×224     | ~10×4 pixels           | ❌ Too small to detect |
| 640×640     | ~28×11 pixels          | ⚠️ Challenging         |
| 1280×1280   | ~55×22 pixels          | ✅ Good                |

---

## 🚀 Recommended Workflow

### For cattlebody:

```bash
# 1. Preprocess (fixes mismatch)
python preprocess_dataset.py --dataset cattlebody --split raw

# 2. Update config.yaml
dataset:
  name: cattlebody_preprocessed

preprocess:
  target_size: [640, 640]

# 3. Train
python train.py --config config.yaml
```

### For cattle:

```bash
# 1. Update config.yaml for small objects
preprocess:
  target_size: [1280, 1280]  # Higher resolution!

loss:
  type: focal  # Handle class imbalance

train:
  batch_size: 4  # Reduce for higher resolution

# 2. Train
python train.py --config config.yaml
```

---

## 📝 Notes

1. **Always use `maintain_aspect: true`** - Your datasets have large aspect ratio variations
2. **Mosaic is critical** - Especially for multi-object scenes (cattle dataset)
3. **Start with 640, increase if needed** - Balance between speed and accuracy
4. **Monitor GPU memory** - 1280 uses ~4× memory of 640
5. **Batch size inversely proportional to resolution** - Reduce batch size for higher resolution

---

**Generated from:** `analyze_datasets_deep.py` results  
**Last updated:** October 3, 2025
