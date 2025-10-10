#!/usr/bin/env python3
"""
Comprehensive system check for cattle detection project.
Validates syntax, imports, and code completeness.

Features:
- Automatic discovery of all Python files in project
- Recursive directory scanning with smart filtering
- Syntax validation using AST parsing
- Import checking with package detection
- Incomplete code detection (TODOs, placeholders)
"""

import ast
import sys
import fnmatch
from pathlib import Path
from typing import List, Tuple, Set

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))


# Directories to exclude from scanning
EXCLUDE_DIRS = {
    '.git', '.idea', '.venv', 'venv', 'env', '__pycache__',
    '.pytest_cache', '.mypy_cache', 'node_modules',
    'archive',  # Archived old code
    'docs',  # Documentation only
    'dataset',  # Data files
    'outputs',  # Training outputs
    'processed_data',  # Processed datasets
    'dataset_analysis_results',  # Analysis results
    'train_logs',  # Log files
}

# File patterns to exclude
EXCLUDE_PATTERNS = [
    '*/archive/*', '*/docs/*', '*/dataset/*', '*/outputs/*',
    '*/processed_data/*', '*/dataset_analysis_results/*',
    '*/train_logs/*', '*/__pycache__/*', '*/.venv/*',
    '*/venv/*', '*/.git/*', '*/.DS_Store',
]

# Packages that are expected to be missing locally but available on server
OPTIONAL_PACKAGES = {
    'torch', 'torchvision', 'numpy', 'PIL', 'pillow',
    'cv2', 'matplotlib', 'seaborn', 'yaml', 'pyyaml',
}


def discover_python_files(root_dir: Path = Path('.')):
    """
    Recursively discover all Python files in the project.

    Args:
        root_dir: Root directory to start scanning from

    Returns:
        List of Python file paths
    """
    python_files = []

    def should_exclude(path: Path):
        """Check if path should be excluded."""
        # Check if any parent directory is in exclude list
        for parent in path.parents:
            if parent.name in EXCLUDE_DIRS:
                return True

        # Check against exclude patterns
        path_str = str(path)
        for pattern in EXCLUDE_PATTERNS:
            if fnmatch.fnmatch(path_str, pattern):
                return True

        return False

    # Recursively find all .py files
    for py_file in root_dir.rglob('*.py'):
        if not should_exclude(py_file):
            python_files.append(py_file)

    return sorted(python_files)


def check_syntax():
    """
    Check syntax of all Python files using AST parsing.

    Returns:
        Tuple of (all_pass, error_messages, checked_files)
    """
    errors = []
    checked_files = []

    # Discover all Python files
    python_files = discover_python_files()

    if not python_files:
        errors.append("⚠️  No Python files found!")
        return False, errors, checked_files

    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
            ast.parse(source)
            print(f"✅ {file_path}")
            checked_files.append(file_path)
        except FileNotFoundError:
            errors.append(f"❌ {file_path}: File not found")
        except SyntaxError as e:
            errors.append(
                f"❌ {file_path}: Syntax error at line {e.lineno}: {e.msg}")
        except Exception as e:
            errors.append(f"❌ {file_path}: {str(e)}")

    return len(errors) == 0, errors, checked_files


def check_imports(checked_files):
    """
    Check if modules can be imported.

    Args:
        checked_files: List of files that passed syntax check

    Returns:
        Tuple of (all_pass, error_messages, missing_packages)
    """
    errors = []
    missing_packages = set()

    # Convert file paths to module names for import testing
    test_modules = []
    for file_path in checked_files:
        # Skip test files, debug scripts, and verification scripts
        if any(x in str(file_path) for x in ['test_', 'debug_', 'verify_', 'check_']):
            continue

        # Convert path to module name (e.g., src/models/registry.py -> src.models.registry)
        if file_path.suffix == '.py' and file_path.stem != '__init__':
            module_parts = file_path.with_suffix('').parts
            # Skip root-level utility scripts
            if len(module_parts) == 1:
                continue
            module_name = '.'.join(module_parts)
            display_name = str(file_path)
            test_modules.append((module_name, display_name))

    for module_name, display_name in test_modules:
        try:
            __import__(module_name)
            print(f"✅ {display_name}")
        except ModuleNotFoundError as e:
            # Check if it's a missing external package
            missing_module = str(e).split("'")[1] if "'" in str(e) else str(e)
            base_package = missing_module.split('.')[0]

            if base_package in OPTIONAL_PACKAGES:
                missing_packages.add(base_package)
                print(f"⚠️  {display_name}: Missing package '{base_package}'")
            else:
                errors.append(f"❌ {display_name}: {e}")
        except ImportError as e:
            errors.append(f"❌ {display_name}: {e}")
        except Exception as e:
            errors.append(f"❌ {display_name}: {e}")

    return len(errors) == 0, errors, missing_packages


def check_incomplete_code(checked_files):
    """
    Check for incomplete implementations, TODOs, and placeholders.

    Args:
        checked_files: List of files that passed syntax check

    Returns:
        Tuple of (all_pass, warning_messages)
    """
    warnings = []

    for file_path in checked_files:
        # Skip test and debug files
        if any(x in str(file_path) for x in ['test_', 'debug_', 'verify_']):
            continue

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for i, line in enumerate(lines, 1):
                line_lower = line.lower()
                line_stripped = line.strip()

                # Skip lines in check_system.py that are part of the checking logic itself
                if 'check_system.py' in str(file_path):
                    skip_keywords = [
                        'TODO found', 'FIXME found', 'Placeholder pass',
                        'Check for', 'Checking for', 'todo\' in', 'fixme\' in',
                        'placeholder\' in', '- Incomplete', 'TODOs, place'
                    ]
                    if any(keyword in line for keyword in skip_keywords):
                        continue

                # Check for TODOs
                if 'todo' in line_lower and not line_stripped.startswith('#'):
                    warnings.append(
                        f"⚠️  {file_path}:{i} - TODO found: {line_stripped}")

                # Check for FIXMEs
                if 'fixme' in line_lower:
                    warnings.append(
                        f"⚠️  {file_path}:{i} - FIXME found: {line_stripped}")

                # Check for suspicious placeholders
                if 'placeholder' in line_lower and 'pass' in line_lower:
                    warnings.append(
                        f"⚠️  {file_path}:{i} - Placeholder pass: {line_stripped}")

        except Exception as e:
            warnings.append(f"⚠️  {file_path}: Error reading file - {e}")

    return len(warnings) == 0, warnings


def main():
    """Run all checks."""
    print("="*70)
    print("COMPREHENSIVE SYSTEM CHECK - AUTO DISCOVERY MODE")
    print("="*70)

    # Discover files
    print("\n🔍 Discovering Python files...")
    discovered_files = discover_python_files()
    print(f"   Found {len(discovered_files)} Python files to check")

    # Syntax check
    print("\n" + "="*70)
    print("SYNTAX CHECK")
    print("="*70)
    syntax_pass, syntax_errors, checked_files = check_syntax()

    if syntax_errors:
        print(f"\n❌ Syntax errors found:")
        for error in syntax_errors:
            print(f"  {error}")
    else:
        print(f"\n✅ All {len(checked_files)} files passed syntax check")

    # Import check
    print("\n" + "="*70)
    print("IMPORT CHECK")
    print("="*70)
    import_pass, import_errors, missing_packages = check_imports(checked_files)

    if import_errors:
        print(f"\n❌ Import errors found:")
        for error in import_errors:
            print(f"  {error}")

    if missing_packages:
        print(f"\n📦 Missing external packages (install later):")
        for pkg in sorted(missing_packages):
            print(f"   - {pkg}")
        print(f"\nℹ️  Install with: pip install torch torchvision numpy pillow pyyaml")

    # Incomplete code check
    print("\n" + "="*70)
    print("INCOMPLETE CODE CHECK")
    print("="*70)
    print("Checking for TODOs, FIXMEs, and placeholders...")
    incomplete_pass, incomplete_warnings = check_incomplete_code(checked_files)

    if incomplete_warnings:
        print(f"\n⚠️  Potential issues found:")
        for warning in incomplete_warnings:
            print(f"  {warning}")
    else:
        print("✅ No incomplete code found")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    all_pass = syntax_pass and import_pass and incomplete_pass

    non_test_files = len([f for f in checked_files if not any(
        x in str(f) for x in ['test_', 'debug_', 'verify_', 'check_'])])

    print(f"{'✅ PASS' if syntax_pass else '❌ FAIL'}: Syntax ({len(checked_files)} files checked)")
    print(f"{'✅ PASS' if import_pass else '❌ FAIL'}: Imports ({non_test_files} modules tested)")
    print(f"{'✅ PASS' if incomplete_pass else '⚠️  WARNINGS'}: Incomplete Code")

    if all_pass:
        print(f"\n🎉 ALL CHECKS PASSED!")
        print(f"\nYour system is ready to use:")
        print(f"  python train.py train -m yolov8_resnet -d cattle -e 2 -b 4")
    else:
        print(f"\n⚠️  Some checks failed. Please fix the issues above.")

    print("="*70 + "\n")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
