#!/usr/bin/env python3
"""
Performance Optimization Guide for Cattle Detection System

This script provides automated recommendations and tools to improve
your detection model performance based on current results.

Usage:
    python src/scripts/optimize_performance.py --current-map 0.6744 --target-map 0.8
    python src/scripts/optimize_performance.py --analyze-results outputs/cattlebody/faster_rcnn/results/
    python src/scripts/optimize_performance.py --suggest-hyperparams --model faster_rcnn
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.config.hyperparameters import (
        FASTER_RCNN_PARAMS, YOLOV8_PARAMS
    )
except ImportError:
    # Fallback if hyperparameters module not available
    FASTER_RCNN_PARAMS = {
        'learning_rate': 0.001,
        'batch_size': 4,
        'num_epochs': 100,
        'img_size': 512
    }
    YOLOV8_PARAMS = {
        'learning_rate': 0.01,
        'batch_size': 16,
        'num_epochs': 100,
        'img_size': 640
    }

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


# Performance improvement strategies
IMPROVEMENT_STRATEGIES = {
    "hyperparameter_tuning": {
        "description": "Optimize learning rate, batch size, and training schedule",
        "expected_improvement": 0.05,  # +5% mAP
        "effort": "Low",
        "time": "2-4 hours"
    },
    "data_augmentation": {
        "description": "Advanced augmentation: mixup, cutmix, mosaic",
        "expected_improvement": 0.03,  # +3% mAP
        "effort": "Medium",
        "time": "1-2 hours"
    },
    "architecture_optimization": {
        "description": "Tune anchor sizes, ROI settings, backbone features",
        "expected_improvement": 0.08,  # +8% mAP
        "effort": "High",
        "time": "4-8 hours"
    },
    "training_schedule": {
        "description": "Multi-stage training with different learning rates",
        "expected_improvement": 0.04,  # +4% mAP
        "effort": "Medium",
        "time": "2-3 hours"
    },
    "ensemble_methods": {
        "description": "Combine multiple models for better predictions",
        "expected_improvement": 0.06,  # +6% mAP
        "effort": "High",
        "time": "3-5 hours"
    },
    "data_quality": {
        "description": "Review and improve annotation quality",
        "expected_improvement": 0.10,  # +10% mAP
        "effort": "Very High",
        "time": "8-16 hours"
    }
}


class PerformanceAnalyzer:
    """Analyzes current performance and suggests improvements."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def analyze_current_performance(self, current_map: float) -> Dict:
        """Analyze current mAP and categorize performance level."""
        performance_levels = {
            (0.0, 0.3): {"level": "Poor", "color": "🔴", "priority": "Critical"},
            (0.3, 0.5): {"level": "Below Average", "color": "🟠", "priority": "High"},
            (0.5, 0.7): {"level": "Average", "color": "🟡", "priority": "Medium"},
            (0.7, 0.85): {"level": "Good", "color": "🟢", "priority": "Low"},
            (0.85, 1.0): {"level": "Excellent", "color": "🟢", "priority": "Optimization"}
        }

        for (low, high), details in performance_levels.items():
            if low <= current_map < high:
                return {
                    "current_map": current_map,
                    "level": details["level"],
                    "color": details["color"],
                    "priority": details["priority"]
                }

        return {"current_map": current_map, "level": "Unknown", "color": "⚪", "priority": "Medium"}

    def suggest_improvements(self, current_map: float, target_map: float) -> List[Dict]:
        """Suggest improvement strategies based on current and target mAP."""
        gap = target_map - current_map
        suggestions = []

        if gap <= 0:
            return [{"message": "🎉 You've already reached your target! Consider optimizing efficiency."}]

        # Rank strategies by expected improvement vs effort
        strategy_ranking = []
        for name, strategy in IMPROVEMENT_STRATEGIES.items():
            efficiency_score = strategy["expected_improvement"] / (
                {"Low": 1, "Medium": 2, "High": 3,
                    "Very High": 4}[strategy["effort"]]
            )
            strategy_ranking.append((name, strategy, efficiency_score))

        # Sort by efficiency (highest first)
        strategy_ranking.sort(key=lambda x: x[2], reverse=True)

        cumulative_improvement = 0
        for name, strategy, efficiency in strategy_ranking:
            if cumulative_improvement < gap:
                suggestions.append({
                    "name": name,
                    "description": strategy["description"],
                    "expected_improvement": strategy["expected_improvement"],
                    "effort": strategy["effort"],
                    "time": strategy["time"],
                    "efficiency_score": efficiency,
                    "priority": "High" if cumulative_improvement < gap * 0.7 else "Medium"
                })
                cumulative_improvement += strategy["expected_improvement"]

        return suggestions

    def generate_hyperparameter_recommendations(self, model: str, current_map: float) -> Dict:
        """Generate hyperparameter recommendations based on current performance."""
        recommendations = {}

        if model == "faster_rcnn":
            base_params = FASTER_RCNN_PARAMS.copy()

            if current_map < 0.5:
                # Poor performance - need major changes
                recommendations = {
                    "learning_rate": 0.005,  # Higher LR for faster learning
                    "batch_size": 2,         # Smaller batch for stability
                    "img_size": 512,         # Higher resolution
                    "num_epochs": 300,       # More training
                    "warmup_epochs": 10,     # Longer warmup
                    "weight_decay": 0.001,   # Stronger regularization
                    "reason": "Low performance detected - aggressive optimization needed"
                }
            elif current_map < 0.7:
                # Average performance - moderate changes
                recommendations = {
                    "learning_rate": 0.002,
                    "batch_size": 4,
                    "img_size": 512,
                    "num_epochs": 200,
                    "warmup_epochs": 5,
                    "weight_decay": 0.0005,
                    "reason": "Average performance - balanced optimization"
                }
            else:
                # Good performance - fine-tuning
                recommendations = {
                    "learning_rate": 0.001,
                    "batch_size": 8,
                    "img_size": 640,
                    "num_epochs": 150,
                    "warmup_epochs": 3,
                    "weight_decay": 0.0001,
                    "reason": "Good performance - fine-tuning for excellence"
                }

        return recommendations

    def create_training_plan(self, suggestions: List[Dict]) -> Dict:
        """Create a step-by-step training improvement plan."""
        plan = {
            "phases": [],
            "total_estimated_time": 0,
            "expected_final_improvement": 0
        }

        # Sort suggestions by priority and efficiency
        high_priority = [s for s in suggestions if s.get("priority") == "High"]
        medium_priority = [
            s for s in suggestions if s.get("priority") == "Medium"]

        phase_num = 1
        for phase_suggestions in [high_priority, medium_priority]:
            if phase_suggestions:
                phase = {
                    "phase": phase_num,
                    "name": f"Phase {phase_num}: {'Critical Improvements' if phase_num == 1 else 'Enhancement Optimizations'}",
                    "strategies": phase_suggestions,
                    "estimated_time": sum([self._parse_time(s["time"]) for s in phase_suggestions]),
                    "expected_improvement": sum([s["expected_improvement"] for s in phase_suggestions])
                }
                plan["phases"].append(phase)
                plan["total_estimated_time"] += phase["estimated_time"]
                plan["expected_final_improvement"] += phase["expected_improvement"]
                phase_num += 1

        return plan

    def _parse_time(self, time_str: str) -> float:
        """Parse time string like '2-4 hours' into average hours."""
        try:
            # Extract numbers from string like "2-4 hours"
            import re
            numbers = re.findall(r'\d+', time_str)
            if len(numbers) == 2:
                return (int(numbers[0]) + int(numbers[1])) / 2
            elif len(numbers) == 1:
                return int(numbers[0])
            else:
                return 4  # Default estimate
        except:
            return 4


def print_performance_report(current_map: float, target_map: Optional[float] = None):
    """Print comprehensive performance analysis report."""
    analyzer = PerformanceAnalyzer()

    print("=" * 80)
    print("🎯 CATTLE DETECTION PERFORMANCE ANALYSIS")
    print("=" * 80)

    # Current performance analysis
    analysis = analyzer.analyze_current_performance(current_map)
    print(f"\n📊 Current Performance:")
    print(f"   • mAP@0.5: {current_map:.4f}")
    print(f"   • Level: {analysis['color']} {analysis['level']}")
    print(f"   • Priority: {analysis['priority']}")

    # Target analysis
    if target_map:
        gap = target_map - current_map
        print(f"\n🎯 Target Performance:")
        print(f"   • Target mAP@0.5: {target_map:.4f}")
        print(f"   • Improvement needed: {gap:.4f} ({gap*100:.1f}%)")

        if gap > 0:
            suggestions = analyzer.suggest_improvements(
                current_map, target_map)

            print(f"\n🚀 Recommended Improvement Strategies:")
            print("-" * 60)
            # Top 5 suggestions
            for i, suggestion in enumerate(suggestions[:5], 1):
                if "message" in suggestion:
                    print(f"{suggestion['message']}")
                else:
                    print(
                        f"{i}. {suggestion['name'].replace('_', ' ').title()}")
                    print(f"   📝 {suggestion['description']}")
                    print(
                        f"   📈 Expected improvement: +{suggestion['expected_improvement']:.3f} mAP")
                    print(
                        f"   ⚡ Effort: {suggestion['effort']} | ⏱️  Time: {suggestion['time']}")
                    print(f"   🎯 Priority: {suggestion['priority']}")
                    print()

        # Create and display training plan
        if gap > 0 and "message" not in suggestions[0]:
            plan = analyzer.create_training_plan(suggestions)

            print(f"📋 STEP-BY-STEP IMPROVEMENT PLAN:")
            print("-" * 60)
            for phase in plan["phases"]:
                print(f"\n🔹 {phase['name']}")
                print(
                    f"   ⏱️  Estimated time: {phase['estimated_time']:.1f} hours")
                print(
                    f"   📈 Expected improvement: +{phase['expected_improvement']:.3f} mAP")

                for j, strategy in enumerate(phase["strategies"], 1):
                    print(
                        f"   {j}. {strategy['name'].replace('_', ' ').title()} ({strategy['effort']} effort)")

            print(f"\n📊 Overall Plan Summary:")
            print(
                f"   ⏱️  Total estimated time: {plan['total_estimated_time']:.1f} hours")
            print(
                f"   📈 Expected final improvement: +{plan['expected_final_improvement']:.3f} mAP")
            print(
                f"   🎯 Projected final mAP: {current_map + plan['expected_final_improvement']:.4f}")


def print_hyperparameter_suggestions(model: str, current_map: float):
    """Print hyperparameter optimization suggestions."""
    analyzer = PerformanceAnalyzer()
    recommendations = analyzer.generate_hyperparameter_recommendations(
        model, current_map)

    print("=" * 80)
    print("⚙️  HYPERPARAMETER OPTIMIZATION RECOMMENDATIONS")
    print("=" * 80)

    print(f"\n🎯 Based on current mAP: {current_map:.4f}")
    print(
        f"📝 Reasoning: {recommendations.pop('reason', 'Performance-based optimization')}")

    print(f"\n🔧 Recommended Parameters:")
    print("-" * 50)
    for param, value in recommendations.items():
        print(f"   {param:20}: {value}")

    print(f"\n🚀 Quick Start Commands:")
    print("-" * 50)
    print(f"# Standard optimization")
    print(
        f"python main.py train-advanced --model {model} --dataset cattlebody --profile high_precision --augmentation --early-stopping")
    print()
    print(f"# Hyperparameter optimization")
    print(
        f"python main.py optimize --model {model} --dataset cattlebody --profile high_precision --trials 10")

    print(f"\n📊 Performance Profiles Available:")
    print("-" * 50)
    print("   • default         : Balanced speed/accuracy")
    print("   • high_precision  : Maximum accuracy (slower)")
    print("   • fast_training   : Quick experiments")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Performance optimization recommendations")

    parser.add_argument('--current-map', type=float,
                        help='Current mAP@0.5 score')
    parser.add_argument('--target-map', type=float,
                        help='Target mAP@0.5 score')
    parser.add_argument('--model', choices=['faster_rcnn', 'yolov8'], default='faster_rcnn',
                        help='Model to optimize')
    parser.add_argument('--suggest-hyperparams', action='store_true',
                        help='Show hyperparameter suggestions')
    parser.add_argument('--analyze-results', type=str,
                        help='Path to results directory to analyze')

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    if args.current_map is not None:
        if args.suggest_hyperparams:
            print_hyperparameter_suggestions(args.model, args.current_map)
        else:
            print_performance_report(args.current_map, args.target_map)

    elif args.analyze_results:
        # Try to extract mAP from results files
        results_path = Path(args.analyze_results)
        if results_path.exists():
            # Look for metrics files
            for metrics_file in results_path.glob("*metrics*.txt"):
                try:
                    with open(metrics_file, 'r') as f:
                        content = f.read()
                        # Extract mAP@0.5 value
                        import re
                        match = re.search(r'mAP@0\.5\s*:\s*([\d\.]+)', content)
                        if match:
                            current_map = float(match.group(1))
                            print(
                                f"📂 Found mAP@0.5: {current_map:.4f} in {metrics_file.name}")
                            print_performance_report(current_map)
                            return
                except Exception as e:
                    continue

            print(f"❌ Could not find mAP metrics in {results_path}")
        else:
            print(f"❌ Results directory not found: {results_path}")

    else:
        parser.print_help()
        print("\n💡 Quick Examples:")
        print("python optimize_performance.py --current-map 0.6744 --target-map 0.8")
        print("python optimize_performance.py --current-map 0.6744 --suggest-hyperparams --model faster_rcnn")


if __name__ == "__main__":
    main()
