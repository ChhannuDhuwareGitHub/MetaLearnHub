"""
Evaluation Audit Script for MetaLearnHub.

Performs:
1. Reproducibility audit on checkpoint 8133a306 with group-leakage-free sampling.
2. Leave-One-Class-Out (LOCO) 5-fold cross-validation audit across all cattle breeds.
"""

import sys
import os
import json
import time
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np

from ml.config import (
    ExperimentConfig, PreprocessConfig, EpisodeConfig,
    TrainingConfig, BREED_NAMES, RAW_DATA_DIR, EXPERIMENTS_DIR
)
from ml.datasets.inspector import DatasetInspector
from ml.datasets.splitter import DatasetSplitter
from ml.training.trainer import MetaTrainer
from ml.algorithms.prototypical_networks.protonet import PrototypicalNetworkAlgorithm
from ml.evaluation.evaluator import FewShotEvaluator


def run_reproducibility_audit():
    print("=" * 60)
    print("Step 1: Reproducibility & Leakage-Free Audit (Checkpoint 8133a306)")
    print("=" * 60)
    
    inspector = DatasetInspector(RAW_DATA_DIR)
    inspector.scan()
    
    ckpt_id = "8133a306"
    ckpt_path = Path(f"checkpoints/{ckpt_id}/best_model.pt")
    
    if not ckpt_path.exists():
        print(f"Error: Checkpoint {ckpt_path} not found.")
        return None
    
    config = ExperimentConfig.load(f"experiments/{ckpt_id}/config.json")
    algo = PrototypicalNetworkAlgorithm(device="cpu")
    algo.load_checkpoint(str(ckpt_path))
    
    evaluator = FewShotEvaluator(algo, inspector, config)
    results = evaluator.evaluate_all_shots(n_way=2, query_per_class=10)
    
    audit_dir = EXPERIMENTS_DIR / "audit_reproducibility_8133a306"
    audit_dir.mkdir(parents=True, exist_ok=True)
    
    audit_record = {
        "audit_timestamp": datetime.now().isoformat(),
        "original_experiment_id": ckpt_id,
        "audit_seed": config.random_seed,
        "held_out_class": config.held_out_class,
        "n_way": 2,
        "eval_shots": config.eval_shots,
        "eval_episodes": config.eval_episodes,
        "shot_results": {},
    }
    
    for shot, res in results.items():
        res_dict = res.to_dict()
        audit_record["shot_results"][str(shot)] = res_dict
    
    with open(audit_dir / "audit_summary.json", "w") as f:
        json.dump(audit_record, f, indent=2)
    
    print(f"\nReproducibility audit saved to {audit_dir / 'audit_summary.json'}")
    return audit_record


def run_loco_audit():
    print("\n" + "=" * 60)
    print("Step 2: Leave-One-Class-Out (LOCO) 5-Fold Audit")
    print("=" * 60)
    
    inspector = DatasetInspector(RAW_DATA_DIR)
    inspector.scan()
    
    loco_dir = EXPERIMENTS_DIR / "loco_audit_seed42"
    loco_dir.mkdir(parents=True, exist_ok=True)
    
    fold_results = {}
    
    for breed_idx, held_out in enumerate(BREED_NAMES, start=1):
        print(f"\n--- Fold {breed_idx}/5: Held-Out Class = {held_out} ---")
        
        config = ExperimentConfig(
            experiment_name=f"loco_fold_{breed_idx}",
            random_seed=42,
            held_out_class=held_out,
            preprocess=PreprocessConfig(image_size=128, center_crop_size=128),
            episode=EpisodeConfig(
                n_way=4,
                k_shot=5,
                query_per_class=2,
                episodes_per_epoch=25,
                num_epochs=2,
            ),
            training=TrainingConfig(device="cpu", num_workers=0),
            eval_shots=[1, 5],
            eval_episodes=10,
        )
        
        trainer = MetaTrainer(config, inspector)
        record = trainer.train()
        exp_id = record['experiment_id']
        
        evaluator = FewShotEvaluator(trainer.algorithm, inspector, config)
        eval_res = evaluator.evaluate_all_shots(n_way=2, query_per_class=10)
        
        fold_dict = {
            "held_out_class": held_out,
            "experiment_id": exp_id,
            "meta_train_classes": config.meta_train_classes,
            "shots": {},
        }
        
        for shot, res in eval_res.items():
            fold_dict["shots"][str(shot)] = {
                "mean_accuracy": round(res.mean_accuracy, 4),
                "std_accuracy": round(res.std_accuracy, 4),
                "ci_95_accuracy": round(1.96 * res.std_accuracy / max(1, np.sqrt(res.num_episodes)), 4),
                "mean_f1": round(res.mean_f1, 4),
                "std_f1": round(res.std_f1, 4),
                "confusion_matrix": res.confusion_matrix,
                "class_names": res.class_names,
            }
        
        fold_results[held_out] = fold_dict
    
    # Compute LOCO aggregates across all 5 folds
    loco_summary = {
        "audit_timestamp": datetime.now().isoformat(),
        "loco_folds": fold_results,
        "aggregate": {},
    }
    
    for shot in [1, 5]:
        accs = [fold_results[b]["shots"][str(shot)]["mean_accuracy"] for b in BREED_NAMES]
        f1s = [fold_results[b]["shots"][str(shot)]["mean_f1"] for b in BREED_NAMES]
        
        loco_summary["aggregate"][f"{shot}_shot"] = {
            "mean_accuracy": round(float(np.mean(accs)), 4),
            "std_accuracy": round(float(np.std(accs)), 4),
            "mean_f1": round(float(np.mean(f1s)), 4),
            "std_f1": round(float(np.std(f1s)), 4),
            "per_fold_accuracies": dict(zip(BREED_NAMES, accs)),
            "per_fold_f1s": dict(zip(BREED_NAMES, f1s)),
        }
    
    with open(loco_dir / "loco_summary.json", "w") as f:
        json.dump(loco_summary, f, indent=2)
    
    print("\n" + "=" * 60)
    print("LOCO 5-Fold Evaluation Audit Completed")
    print("=" * 60)
    for shot in [1, 5]:
        agg = loco_summary["aggregate"][f"{shot}_shot"]
        print(f"\n{shot}-Shot LOCO Aggregate Accuracy: {agg['mean_accuracy']*100:.2f}% ± {agg['std_accuracy']*100:.2f}%")
        print(f"{shot}-Shot LOCO Aggregate Macro F1: {agg['mean_f1']*100:.2f}% ± {agg['std_f1']*100:.2f}%")
        print("Per-fold Accuracies:")
        for b, acc in agg["per_fold_accuracies"].items():
            print(f"  - {b}: {acc*100:.2f}%")
    
    return loco_summary


if __name__ == "__main__":
    reprod = run_reproducibility_audit()
    loco = run_loco_audit()
