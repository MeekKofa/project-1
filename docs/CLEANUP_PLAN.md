# 🧹 PROJECT CLEANUP & CONSOLIDATION PLAN

## Current Issues

### 1. Multiple Config Systems (REDUNDANT)

```
configs/
├── yolov8_cattlebody.yaml     # Old format with hardcoded num_classes
├── high_performance.yaml      # Old format with hardcoded num_classes
└── quick_test.yaml            # Old format

config.yaml                     # NEW dynamic config (correct!)
dataset_profiles.yaml           # NEW dataset profiles (correct!)
src/config/cattle.yaml          # OLD/redundant
```

### 2. Multiple Config Loaders (REDUNDANT)

```
src/config/
├── config_loader.py           # OLD loader
├── dataset_config.py          # OLD loader
├── training_config.py         # OLD loader
└── dynamic_config_loader.py   # NEW loader (correct!)
```

---

## 🎯 CLEANUP ACTIONS

### Phase 1: Archive Old Configs

Move old configs to an archive folder:

```bash
mkdir -p archive/old_configs
mv configs/*.yaml archive/old_configs/
mv src/config/cattle.yaml archive/old_configs/
```

### Phase 2: Consolidate New System

Keep ONLY:

- `config.yaml` - Main training configuration (dynamic)
- `dataset_profiles.yaml` - Dataset-specific profiles
- `src/config/dynamic_config_loader.py` - Dynamic loader

### Phase 3: Create Unified Preset System

Instead of multiple config files, use **config presets** within main config.yaml:

```yaml
# config.yaml can have presets section
presets:
  quick_test:
    epochs: 20
    batch_size: 16
    image_size: 416

  high_performance:
    epochs: 300
    batch_size: 16
    image_size: 640

  standard:
    epochs: 100
    batch_size: 8
    image_size: 640
```

Then use: `python workflow_manager.py --preset quick_test`

---

## 📁 FINAL CLEAN STRUCTURE

```
project1/
├── config.yaml                      # ✅ Main config (dynamic, no hardcoding)
├── dataset_profiles.yaml            # ✅ Dataset-specific profiles
├── workflow_manager.py              # ✅ Main entry point
├── preprocess_dataset.py            # ✅ Preprocessing script
│
├── src/
│   └── config/
│       ├── __init__.py
│       └── dynamic_config_loader.py # ✅ ONLY loader we need
│
├── dataset_analysis_results/        # ✅ Analysis outputs
├── processed_data/                  # ✅ Preprocessed data
│
├── archive/                         # 📦 Old/deprecated files
│   └── old_configs/
│       ├── yolov8_cattlebody.yaml
│       ├── high_performance.yaml
│       ├── quick_test.yaml
│       └── cattle.yaml
│
└── docs/                            # 📚 Documentation
    ├── CONFIG_SYSTEM_README.md
    └── WORKFLOW_GUIDE.md
```

---

## 🚀 EXECUTION PLAN

### Step 1: Backup Everything

```bash
git add .
git commit -m "Backup before cleanup"
```

### Step 2: Archive Old Files

```bash
mkdir -p archive/old_configs
mkdir -p archive/old_loaders

# Move old config files
mv configs/yolov8_cattlebody.yaml archive/old_configs/
mv configs/high_performance.yaml archive/old_configs/
mv configs/quick_test.yaml archive/old_configs/
mv src/config/cattle.yaml archive/old_configs/

# Move old loader files (keep only dynamic_config_loader.py)
mv src/config/config_loader.py archive/old_loaders/
mv src/config/dataset_config.py archive/old_loaders/
mv src/config/training_config.py archive/old_loaders/
```

### Step 3: Update config.yaml with Presets

Add a presets section to config.yaml for different training modes.

### Step 4: Create Documentation

Organize all README files in docs/ folder.

### Step 5: Test

```bash
# Test workflow manager
python workflow_manager.py --dataset cattlebody --stage check

# Test config loading
python src/config/dynamic_config_loader.py
```

---

## 🎨 ENHANCED CONFIG.YAML STRUCTURE

```yaml
# =============================================================================
# ACTIVE CONFIGURATION
# =============================================================================
# Set which preset to use (overrides defaults)
active_preset: standard # Options: quick_test, standard, high_performance, custom

# =============================================================================
# PRESETS - Quick switching between training modes
# =============================================================================
presets:
  quick_test:
    train:
      epochs: 20
      batch_size: 16
    preprocess:
      target_size: [416, 416]
    strategy:
      early_stopping: false
    output:
      save_predictions: false
      save_visualizations: false

  standard:
    train:
      epochs: 100
      batch_size: 8
    preprocess:
      target_size: [640, 640]
    strategy:
      early_stopping: true
      patience: 20

  high_performance:
    train:
      epochs: 300
      batch_size: 16
      learning_rate: 0.002
    preprocess:
      target_size: [640, 640]
    augmentation:
      rotation: 15
      brightness: 0.3
    strategy:
      early_stopping: true
      patience: 50

  custom:
    # Custom settings defined below
# =============================================================================
# BASE CONFIGURATION (Can be overridden by presets)
# =============================================================================
# ... rest of config ...
```

---

## 📊 BENEFITS OF CLEANUP

### Before (Messy):

- ❌ 3+ config files with hardcoded values
- ❌ 4 different config loaders
- ❌ Confusion about which to use
- ❌ Duplicated settings
- ❌ Inconsistent formats

### After (Clean):

- ✅ 1 main config with presets
- ✅ 1 dataset profile file
- ✅ 1 dynamic loader
- ✅ No hardcoded values
- ✅ Easy preset switching
- ✅ Clear documentation
- ✅ Archive of old files (not deleted)

---

## 🔄 MIGRATION GUIDE

If you were using old configs:

### Old way:

```bash
python train.py --config configs/yolov8_cattlebody.yaml
```

### New way:

```bash
# Option 1: Use preset
python workflow_manager.py --dataset cattlebody --stage train --preset standard

# Option 2: Edit config.yaml and set active_preset
# Then just run:
python workflow_manager.py --dataset cattlebody --stage train
```

---

## ✅ VERIFICATION CHECKLIST

After cleanup, verify:

- [ ] config.yaml has presets section
- [ ] dataset_profiles.yaml exists
- [ ] dynamic_config_loader.py works
- [ ] Old files in archive/ folder
- [ ] workflow_manager.py works with new structure
- [ ] Documentation updated
- [ ] Git committed

---

## 🎯 NEXT STEPS AFTER CLEANUP

1. Run analysis on all datasets
2. Update dataset_profiles.yaml with analysis results
3. Preprocess datasets
4. Test training with different presets
5. Document results

---

**Status:** Ready to execute cleanup! 🚀
