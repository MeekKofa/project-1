# 🎯 QUICK REFERENCE CARD

## 🚀 Common Commands

### Workflow Manager (Main Tool)

```bash
# Help
python workflow_manager.py --help

# Check dataset
python workflow_manager.py --dataset cattlebody --stage check

# Run analysis
python workflow_manager.py --dataset cattlebody --stage analyze

# Preprocess
python workflow_manager.py --dataset cattlebody --stage preprocess

# Train
python workflow_manager.py --dataset cattlebody --stage train

# Full pipeline
python workflow_manager.py --dataset cattlebody --stage all
```

### Preprocessing (Standalone)

```bash
# Preprocess dataset
python preprocess_dataset.py --dataset cattlebody --split raw

# Force overwrite
python preprocess_dataset.py --dataset cattlebody --split raw --force
```

### Configuration Testing

```bash
# Test dynamic config loader
python src/config/dynamic_config_loader.py
```

---

## ⚙️ Configuration Quick Edits

### Switch Training Preset

```yaml
# Edit config.yaml
active_preset: quick_test      # Fast (20 epochs)
active_preset: standard        # Balanced (100 epochs)
active_preset: high_performance # Maximum (300 epochs)
```

### Change Dataset

```yaml
# Edit config.yaml
dataset:
  name: cattlebody # Or: cattle, cattleface
  split: raw # Or: processed
```

### Adjust Resolution

```yaml
# Edit config.yaml
preprocess:
  target_size: [640, 640]     # Standard
  target_size: [1280, 1280]   # For small objects
  target_size: [416, 416]     # Quick tests
```

---

## 📊 Dataset Quick Facts

| Dataset    | Classes        | Images | Status                 | Resolution |
| ---------- | -------------- | ------ | ---------------------- | ---------- |
| cattlebody | 1              | 4,852  | ⚠️ Needs preprocessing | 640x640    |
| cattle     | 2 (imbalanced) | 11,369 | ✅ Ready               | 1280x1280  |
| cattleface | 0              | 6,528  | ❌ No labels           | N/A        |

---

## 🔧 Common Issues & Fixes

| Issue                | Solution                           |
| -------------------- | ---------------------------------- |
| Image/label mismatch | Run preprocessing                  |
| Class imbalance      | Use `loss.type: auto` (focal loss) |
| Small objects        | Increase resolution to 1280        |
| Too slow             | Use `quick_test` preset            |
| Missing labels       | Check dataset or re-annotate       |

---

## 📁 Important Files

| File                                  | Purpose                         |
| ------------------------------------- | ------------------------------- |
| `config.yaml`                         | Main configuration (edit this!) |
| `dataset_profiles.yaml`               | Dataset-specific settings       |
| `workflow_manager.py`                 | Main entry point                |
| `preprocess_dataset.py`               | Preprocessing script            |
| `src/config/dynamic_config_loader.py` | Runtime detection               |

---

## 🎯 Training Presets

| Preset           | Epochs        | Batch | Resolution | Use Case          |
| ---------------- | ------------- | ----- | ---------- | ----------------- |
| quick_test       | 20            | 16    | 416        | Fast iteration    |
| standard         | 100           | 8     | 640        | Balanced training |
| high_performance | 300           | 16    | 640        | Maximum accuracy  |
| custom           | Your settings | -     | -          | Full control      |

---

## 📈 Workflow Stages

| Stage      | Description              | Command              |
| ---------- | ------------------------ | -------------------- |
| check      | Validate dataset health  | `--stage check`      |
| analyze    | Deep analysis with stats | `--stage analyze`    |
| preprocess | Clean and normalize data | `--stage preprocess` |
| train      | Train detection model    | `--stage train`      |
| validate   | Validate performance     | `--stage validate`   |
| test       | Test on test set         | `--stage test`       |
| visualize  | Generate visualizations  | `--stage visualize`  |
| all        | Run entire pipeline      | `--stage all`        |

---

## 🧹 Project Cleanup

```bash
# Run cleanup (one time)
chmod +x cleanup.sh
./cleanup.sh
```

**What it does:**

- Archives old config files
- Organizes documentation
- Creates clean structure
- Keeps old files (not deleted)

---

## ✅ Quick Health Check

```bash
# 1. Check file exists
ls config.yaml dataset_profiles.yaml workflow_manager.py

# 2. Test workflow manager
python workflow_manager.py --help

# 3. Test config loader
python src/config/dynamic_config_loader.py

# 4. Check dataset
python workflow_manager.py --dataset cattlebody --stage check
```

---

## 🎓 Remember

1. **Never hardcode** dataset properties (num_classes, class_names)
2. **Always run analysis** before preprocessing
3. **Use presets** for quick switching
4. **Check stage** before training
5. **Run preprocessing** to fix data issues

---

## 📞 Getting Help

```bash
# Workflow manager help
python workflow_manager.py --help

# Preprocessing help
python preprocess_dataset.py --help

# View summary
python workflow_manager.py --dataset cattlebody --summary
```

---

## 🚀 Typical Workflow

```bash
# 1. Cleanup (one time)
./cleanup.sh

# 2. Analyze dataset
python workflow_manager.py --dataset cattlebody --stage analyze

# 3. Preprocess dataset
python workflow_manager.py --dataset cattlebody --stage preprocess

# 4. Quick test (edit config.yaml: active_preset: quick_test)
python workflow_manager.py --dataset cattlebody --stage train

# 5. Full training (edit config.yaml: active_preset: standard)
python workflow_manager.py --dataset cattlebody --stage train

# 6. Validate
python workflow_manager.py --dataset cattlebody --stage validate
```

---

**Print this card or keep it open for quick reference! 📋**
