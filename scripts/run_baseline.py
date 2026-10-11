"""
Script to execute the conventional ResNet-18 baseline comparison.
"""
import os
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.config import ExperimentConfig, RAW_DATA_DIR, PreprocessConfig, EpisodeConfig, TrainingConfig
from ml.datasets.inspector import DatasetInspector
from ml.evaluation.evaluator import BaselineEvaluator


def main():
    print("=" * 60)
    print("MetaLearnHub - ResNet-18 Conventional Baseline Comparison")
    print("=" * 60)
    
    # Load or define experiment config
    config = ExperimentConfig(
        experiment_name="baseline_resnet18",
        held_out_class="Red Dane cattle",
        preprocess=PreprocessConfig(image_size=128),
        episode=EpisodeConfig(num_epochs=10), # 10 epochs for conventional training
        training=TrainingConfig(device="cpu", learning_rate=0.001)
    )
    
    print("\nScanning dataset...")
    inspector = DatasetInspector(RAW_DATA_DIR)
    inspector.scan()
    
    print("\nTraining conventional ResNet-18 baseline...")
    evaluator = BaselineEvaluator(inspector, config)
    result = evaluator.train_and_evaluate()
    
    print("\n" + "=" * 60)
    print("Baseline Results:")
    print(f"- Known Classes Test Accuracy: {result['known_class_accuracy'] * 100:.2f}%")
    print(f"- Known Classes Test F1: {result['known_class_f1'] * 100:.2f}%")
    print(f"- Unseen Class Behavior: {result['unseen_class_note']}")
    print("=" * 60)

if __name__ == "__main__":
    main()
