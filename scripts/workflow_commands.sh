#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'EOF'
Usage: workflow_commands.sh <command> [options]

Core commands
  train [args...]              Run training via train.py (pass model/dataset flags)
  eval [args...]               Run evaluation via train.py (checkpoint, split, etc.)
  preprocess [args...]         Invoke scripts/workflow_manager.py for preprocessing
  workflow [args...]           Run scripts/workflow_manager.py with a custom stage

Maintenance & diagnostics
  verify-system                Execute verify_system.py checks
  verify-config                Run verify_training_config.py
  dataset-detect               Run test_dataset_detection.py helper
  check-system                 Execute check_system.py (environment sanity)
  analyze-dataset [args...]    Deep dataset analysis script

Artifacts helpers
  list-metrics --dataset D --model M        List metric files for a run
  list-visuals --dataset D --model M        List visualization assets for a run

Pass --help with a command for details in the underlying script.
Examples
  ./scripts/workflow_commands.sh train -m yolov8_resnet -d cattle -e 10 -b 4
  ./scripts/workflow_commands.sh eval -m yolov8_resnet -d cattle -p outputs/cattle/yolov8_resnet/checkpoints/best.pth --split test

EOF
}

need_args() {
  if [[ $1 -lt 1 ]]; then
    usage
    exit 1
  fi
}

list_artifacts() {
  local type="$1"; shift
  local dataset=""
  local model=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --dataset)
        dataset="$2"; shift 2 ;;
      --model)
        model="$2"; shift 2 ;;
      *)
        echo "Unknown option: $1" >&2
        exit 1 ;;
    esac
  done

  if [[ -z "$dataset" || -z "$model" ]]; then
    echo "Both --dataset and --model are required" >&2
    exit 1
  fi

  local target
  case "$type" in
    metrics)
      target="$PROJECT_ROOT/outputs/$dataset/$model/metrics"
      ;;
    visuals)
      target="$PROJECT_ROOT/outputs/$dataset/$model/visualizations"
      ;;
    *)
      echo "Unsupported artifact type: $type" >&2
      exit 1
      ;;
  esac

  if [[ -d "$target" ]]; then
    echo "Listing $type under $target"
    ls -R "$target"
  else
    echo "No $type directory found at $target" >&2
    exit 1
  fi
}

if [[ $# -lt 1 ]]; then
  usage
  exit 0
fi

cmd="$1"
shift || true

case "$cmd" in
  train)
    python "$PROJECT_ROOT/train.py" train "$@"
    ;;
  eval)
    python "$PROJECT_ROOT/train.py" eval "$@"
    ;;
  preprocess)
    python "$PROJECT_ROOT/scripts/workflow_manager.py" --stage preprocess "$@"
    ;;
  workflow)
    python "$PROJECT_ROOT/scripts/workflow_manager.py" "$@"
    ;;
  verify-system)
    python "$PROJECT_ROOT/verify_system.py" "$@"
    ;;
  verify-config)
    python "$PROJECT_ROOT/verify_training_config.py" "$@"
    ;;
  dataset-detect)
    python "$PROJECT_ROOT/test_dataset_detection.py" "$@"
    ;;
  check-system)
    python "$PROJECT_ROOT/check_system.py" "$@"
    ;;
  analyze-dataset)
    python "$PROJECT_ROOT/scripts/analyze_datasets_deep.py" "$@"
    ;;
  list-metrics)
    list_artifacts "metrics" "$@"
    ;;
  list-visuals)
    list_artifacts "visuals" "$@"
    ;;
  help|--help|-h)
    usage
    ;;
  *)
    echo "Unknown command: $cmd" >&2
    usage
    exit 1
    ;;
esac
