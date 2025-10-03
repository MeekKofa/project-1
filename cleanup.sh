#!/bin/bash

# Project Cleanup and Reorganization Script
# This script archives old/redundant files and organizes the project structure

set -e  # Exit on error

echo "🧹 Starting Project Cleanup..."
echo "================================"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo ""
echo "📁 Current directory: $SCRIPT_DIR"
echo ""

# Step 1: Create archive directories
echo "📦 Step 1: Creating archive directories..."
mkdir -p archive/old_configs
mkdir -p archive/old_loaders
mkdir -p archive/old_docs
mkdir -p docs

echo -e "${GREEN}✓${NC} Archive directories created"

# Step 2: Archive old config files
echo ""
echo "📦 Step 2: Archiving old configuration files..."

# Check and move old configs
if [ -d "configs" ]; then
    echo "  Moving configs/*.yaml to archive..."
    if [ -f "configs/yolov8_cattlebody.yaml" ]; then
        mv configs/yolov8_cattlebody.yaml archive/old_configs/
        echo -e "    ${GREEN}✓${NC} Moved yolov8_cattlebody.yaml"
    fi
    if [ -f "configs/high_performance.yaml" ]; then
        mv configs/high_performance.yaml archive/old_configs/
        echo -e "    ${GREEN}✓${NC} Moved high_performance.yaml"
    fi
    if [ -f "configs/quick_test.yaml" ]; then
        mv configs/quick_test.yaml archive/old_configs/
        echo -e "    ${GREEN}✓${NC} Moved quick_test.yaml"
    fi
    
    # Remove empty configs directory
    if [ -z "$(ls -A configs)" ]; then
        rmdir configs
        echo -e "    ${GREEN}✓${NC} Removed empty configs/ directory"
    fi
fi

# Archive old cattle.yaml if exists
if [ -f "src/config/cattle.yaml" ]; then
    mv src/config/cattle.yaml archive/old_configs/
    echo -e "    ${GREEN}✓${NC} Moved src/config/cattle.yaml"
fi

# Step 3: Archive old loader files
echo ""
echo "📦 Step 3: Archiving old loader files..."

if [ -f "src/config/config_loader.py" ]; then
    mv src/config/config_loader.py archive/old_loaders/
    echo -e "    ${GREEN}✓${NC} Moved config_loader.py"
fi

if [ -f "src/config/dataset_config.py" ]; then
    mv src/config/dataset_config.py archive/old_loaders/
    echo -e "    ${GREEN}✓${NC} Moved dataset_config.py"
fi

if [ -f "src/config/training_config.py" ]; then
    mv src/config/training_config.py archive/old_loaders/
    echo -e "    ${GREEN}✓${NC} Moved training_config.py"
fi

# Step 4: Organize documentation
echo ""
echo "📚 Step 4: Organizing documentation..."

# Move documentation files to docs/
DOC_FILES=(
    "CONFIG_SYSTEM_README.md"
    "CLEANUP_PLAN.md"
    "ARCHITECTURE_DIAGRAM.md"
    "NEW_ARCHITECTURE.md"
    "NEW_SYSTEM_README.md"
    "TRAINING_GUIDE.md"
    "BOX_SANITIZATION_FIX.md"
    "GRADIENT_GRAPH_FIX.md"
    "OBJECTNESS_FIX.md"
    "CRITICAL_FIXES_APPLIED.md"
    "CRITICAL_FIXES_OCT2.md"
    "PERFORMANCE_ANALYSIS.md"
    "REBUILD_SUMMARY.md"
    "MIGRATION_CHECKLIST.md"
    "COMPARISON.md"
    "COMPLETE.md"
    "SUMMARY.md"
    "CHECKLIST.md"
)

for file in "${DOC_FILES[@]}"; do
    if [ -f "$file" ]; then
        mv "$file" docs/
        echo -e "    ${GREEN}✓${NC} Moved $file to docs/"
    fi
done

# Keep main README in root
if [ ! -f "README.md" ]; then
    echo -e "    ${YELLOW}⚠${NC} No README.md in root - consider creating one"
fi

# Step 5: Create archive README
echo ""
echo "📝 Step 5: Creating archive documentation..."

cat > archive/README.md << 'EOF'
# Archived Files

This directory contains old/deprecated files that have been replaced by the new robust system.

## Why Archived?

These files were part of the old configuration system that had:
- Hardcoded values (num_classes, class_names, image counts)
- Multiple redundant config files
- Inconsistent formats
- No dynamic detection

## New System

The new system uses:
- **config.yaml** - Single dynamic config with presets
- **dataset_profiles.yaml** - Dataset-specific profiles
- **dynamic_config_loader.py** - Runtime property detection
- **workflow_manager.py** - Unified workflow management

## Contents

### old_configs/
Old configuration files with hardcoded values

### old_loaders/
Old configuration loaders replaced by dynamic_config_loader.py

### old_docs/
Old documentation files (kept for reference)

## Can I Delete These?

These files are kept for reference and rollback purposes. 
After verifying the new system works correctly, you can safely delete this archive folder.

**Date Archived:** $(date)
EOF

echo -e "${GREEN}✓${NC} Created archive/README.md"

# Step 6: Create .gitignore updates
echo ""
echo "📝 Step 6: Updating .gitignore..."

# Add common ignore patterns if not already present
GITIGNORE_ENTRIES=(
    "# Python"
    "__pycache__/"
    "*.py[cod]"
    "*$py.class"
    "*.so"
    ".Python"
    "env/"
    "venv/"
    ".venv/"
    ""
    "# Project specific"
    "outputs/"
    "processed_data/*_preprocessed/"
    "*.log"
    "*.png"
    "*.jpg"
    "*.jpeg"
    ".DS_Store"
    ""
    "# IDE"
    ".vscode/"
    ".idea/"
    "*.swp"
    "*.swo"
)

if [ ! -f ".gitignore" ]; then
    echo "Creating .gitignore..."
    printf '%s\n' "${GITIGNORE_ENTRIES[@]}" > .gitignore
    echo -e "${GREEN}✓${NC} Created .gitignore"
else
    echo -e "${YELLOW}ℹ${NC} .gitignore already exists (not modified)"
fi

# Step 7: Summary
echo ""
echo "================================"
echo "🎉 Cleanup Complete!"
echo "================================"
echo ""
echo "📊 Summary:"
echo "  ✅ Old configs archived to archive/old_configs/"
echo "  ✅ Old loaders archived to archive/old_loaders/"
echo "  ✅ Documentation organized in docs/"
echo "  ✅ Archive README created"
echo ""
echo "📁 Current structure:"
echo "  - config.yaml (main config with presets)"
echo "  - dataset_profiles.yaml (dataset-specific)"
echo "  - workflow_manager.py (main entry point)"
echo "  - src/config/dynamic_config_loader.py (runtime detection)"
echo "  - docs/ (all documentation)"
echo "  - archive/ (old files)"
echo ""
echo "🔍 Verification:"
echo "  Run: python workflow_manager.py --dataset cattlebody --stage check"
echo "  Run: python src/config/dynamic_config_loader.py"
echo ""
echo -e "${GREEN}Ready to start clean workflow! 🚀${NC}"
