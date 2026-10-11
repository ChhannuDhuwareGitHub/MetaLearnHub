"""
Evaluation Audit Script for MetaLearnHub.

Performs a comprehensive research audit covering:
1. Reproducibility check on existing ProtoNet experiment 8133a306.
2. Scientific baseline comparison audit (separating known-class accuracy from dynamic adaptation).
3. Leave-One-Class-Out (LOCO) evaluation across all 5 cattle breeds with group leakage safety.
4. Validation and aggregation of metrics across folds with 95% confidence intervals.
"""

import os
import sys
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from ml.config import ExperimentConfig, PreprocessConfig, EpisodeConfig, TrainingConfig, BREED_NAMES, RAW_DATA_DIR, EXPERIMENTS_DIR
from ml.datasets.inspector import DatasetInspector
from ml.datasets.splitter import DatasetSplitter
from ml.algorithms.prototypical_networks.protonet import PrototypicalNetworkAlgorithm
from ml.training.trainer import MetaTrainer
from ml.evaluation.evaluator import FewShotEvaluator, BaselineEvaluator, EvaluationResult


def run_reproducibility_audit(inspector: DatasetInspector) -> Dict[str, Any]:
    """
    Step 2: Re-run ProtoNet evaluation on experiment 8133a306 checkpoint using recorded seed 42.
    Saves predictions, labels, and per-episode metrics without overwriting original files.
    """
    logger.info("=" * 60)
    logger.info("STEP 2: Reproducibility Audit on Experiment 8133a306")
    logger.info("=" * 60)

    original_exp_id = "8133a306"
    ckpt_path = PROJECT_ROOT / "checkpoints" / original_exp_id / "best_model.pt"
    orig_config_path = EXPERIMENTS_DIR / original_exp_id / "config.json"
    
    if not ckpt_path.exists() or not orig_config_path.exists():
        logger.error(f"Checkpoint or config missing for {original_exp_id}")
        return {}

    # Load original config
    config = ExperimentConfig.load(orig_config_path)
    config.training.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.eval_episodes = 10  # matching original eval_episodes count
    
    # Load model algorithm
    algorithm = PrototypicalNetworkAlgorithm(device=config.training.device)
    algorithm.load_checkpoint(str(ckpt_path))
    
    # Re-evaluate
    evaluator = FewShotEvaluator(algorithm, inspector, config)
    repro_results = evaluator.evaluate_all_shots(n_way=2, query_per_class=10)
    
    # Save to distinct audit directory
    audit_exp_id = f"audit_reproducibility_{original_exp_id}"
    audit_dir = EXPERIMENTS_DIR / audit_exp_id
    audit_dir.mkdir(parents=True, exist_ok=True)
    
    repro_summary = {
        'audit_experiment_id': audit_exp_id,
        'original_experiment_id': original_exp_id,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'random_seed': config.random_seed,
        'held_out_class': config.held_out_class,
        'shot_results': {str(k): res.to_dict() for k, res in repro_results.items()}
    }
    
    with open(audit_dir / "reproducibility_summary.json", "w") as f:
        json.dump(repro_summary, f, indent=2)
        
    # Compare with original saved results
    orig_1shot_path = EXPERIMENTS_DIR / original_exp_id / "evaluation" / "eval_1shot.json"
    orig_5shot_path = EXPERIMENTS_DIR / original_exp_id / "evaluation" / "eval_5shot.json"
    diff_report = {}
    REPRO_TOLERANCE = 0.02  # 2 percentage-point tolerance for "reproduced"

    for shot_label, orig_path in [('1shot', orig_1shot_path), ('5shot', orig_5shot_path)]:
        if not orig_path.exists():
            continue
        with open(orig_path, "r") as f:
            orig_data = json.load(f)

        shot_key = shot_label.replace('shot', '')  # '1' or '5'
        new_data = repro_summary['shot_results'].get(shot_key, {})

        orig_acc = orig_data.get('mean_accuracy')
        audit_acc = new_data.get('mean_accuracy')
        orig_f1 = orig_data.get('mean_f1')
        audit_f1 = new_data.get('mean_f1')
        orig_classes = orig_data.get('class_names', [])
        audit_classes = new_data.get('class_names', [])

        acc_delta = (audit_acc - orig_acc) if (orig_acc is not None and audit_acc is not None) else None
        f1_delta = (audit_f1 - orig_f1) if (orig_f1 is not None and audit_f1 is not None) else None
        acc_reproduced = abs(acc_delta) <= REPRO_TOLERANCE if acc_delta is not None else None
        class_pairing_match = orig_classes == audit_classes

        explanation_parts = []
        if acc_delta is not None:
            if acc_reproduced:
                explanation_parts.append(
                    f"Accuracy reproduced within tolerance: original {orig_acc*100:.2f}% vs audit {audit_acc*100:.2f}% "
                    f"(delta {acc_delta*100:+.2f}pp, threshold ±{REPRO_TOLERANCE*100:.0f}pp)."
                )
            else:
                explanation_parts.append(
                    f"Accuracy NOT reproduced: original {orig_acc*100:.2f}% vs audit {audit_acc*100:.2f}% "
                    f"(delta {acc_delta*100:+.2f}pp, exceeds ±{REPRO_TOLERANCE*100:.0f}pp threshold)."
                )
        if not class_pairing_match:
            explanation_parts.append(
                f"Known-class pairing differs: original {orig_classes} vs audit {audit_classes}. "
                f"This indicates non-deterministic episode sampling which affects comparability."
            )
        if f1_delta is not None:
            explanation_parts.append(
                f"F1 delta: original {orig_f1*100:.2f}% vs audit {audit_f1*100:.2f}% ({f1_delta*100:+.2f}pp)."
            )

        diff_report[shot_label] = {
            'original_mean_accuracy': orig_acc,
            'audit_mean_accuracy': audit_acc,
            'accuracy_delta': round(acc_delta, 4) if acc_delta is not None else None,
            'accuracy_reproduced': acc_reproduced,
            'original_mean_f1': orig_f1,
            'audit_mean_f1': audit_f1,
            'f1_delta': round(f1_delta, 4) if f1_delta is not None else None,
            'original_class_pairing': orig_classes,
            'audit_class_pairing': audit_classes,
            'class_pairing_match': class_pairing_match,
            'explanation': " ".join(explanation_parts),
        }

    with open(audit_dir / "comparison_diff_report.json", "w") as f:
        json.dump(diff_report, f, indent=2)
        
    logger.info(f"Reproducibility audit saved to {audit_dir}")
    return repro_summary


def run_baseline_precision_audit(inspector: DatasetInspector) -> Dict[str, Any]:
    """
    Step 3: Baseline model audit with clear distinction between supervised known-class accuracy
    and novel-class adaptation.
    """
    logger.info("=" * 60)
    logger.info("STEP 3: Scientific Baseline Precision Audit")
    logger.info("=" * 60)

    config = ExperimentConfig(
        experiment_name="baseline_audit",
        held_out_class="Red Dane cattle",
        random_seed=42
    )
    config.training.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    baseline_eval = BaselineEvaluator(inspector, config)
    res = baseline_eval.train_and_evaluate()
    
    audit_notes = {
        'supervised_known_class_test_accuracy': res.get('known_class_accuracy'),
        'supervised_known_class_macro_f1': res.get('known_class_f1'),
        'best_val_accuracy': res.get('best_val_accuracy'),
        'architectural_boundary_explanation': (
            "The conventional ResNet-18 classifier is trained with a fixed 4-class linear head (FC layer outputs 4 logits). "
            "Because its output dimension is strictly 4, it cannot output class index 4 (the 5th held-out breed). "
            "Evaluating it directly on unseen class images without dynamic adaptation or fine-tuning yields 0% direct accuracy on the novel label. "
            "This 0% result is NOT a fair head-to-head performance loss, but an architectural boundary demonstrating why meta-learning "
            "(dynamic distance-based prototypes) is necessary for few-shot transfer."
        )
    }
    res['audit_notes'] = audit_notes
    
    audit_exp_id = res['experiment_id']
    logger.info(f"Baseline known-class test acc: {res.get('known_class_accuracy')*100:.2f}%")
    return res


def run_loco_evaluation_audit(inspector: DatasetInspector) -> Dict[str, Any]:
    """
    Step 4 & 5: Leave-One-Class-Out (LOCO) evaluation across all 5 cattle breeds.
    Evaluates 1-shot and 5-shot performance per fold, reporting per-fold & aggregate metrics with 95% CIs.
    """
    logger.info("=" * 60)
    logger.info("STEP 4 & 5: Leave-One-Class-Out (LOCO) Audit Across All 5 Breeds")
    logger.info("=" * 60)

    audit_exp_id = "loco_audit_seed42"
    audit_dir = EXPERIMENTS_DIR / audit_exp_id
    audit_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fold_results = {}
    
    # Store per-fold metric summaries
    all_1shot_acc = []
    all_1shot_f1 = []
    all_5shot_acc = []
    all_5shot_f1 = []

    for fold_idx, held_out in enumerate(BREED_NAMES, start=1):
        breed_slug = held_out.lower().replace(" ", "_")
        logger.info(f"\n--- Fold {fold_idx}/5: Held-out Class = '{held_out}' ---")
        
        config = ExperimentConfig(
            experiment_name=f"loco_fold_{breed_slug}",
            held_out_class=held_out,
            random_seed=42,
            episode=EpisodeConfig(
                n_way=4,
                k_shot=5,
                query_per_class=2,
                episodes_per_epoch=20,
                num_epochs=3
            ),
            eval_shots=[1, 5],
            eval_episodes=20
        )
        config.training.device = device
        config.training.num_workers = 0
        
        # Train Meta-Learning model for this fold
        trainer = MetaTrainer(config, inspector)
        train_stats = trainer.train()
        
        # Evaluate algorithm on held-out fold
        algorithm = trainer.algorithm
        evaluator = FewShotEvaluator(algorithm, inspector, config)
        eval_res = evaluator.evaluate_all_shots(n_way=2, query_per_class=10)
        
        fold_dir = audit_dir / f"fold_{fold_idx}_{breed_slug}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        
        fold_data = {
            'fold': fold_idx,
            'held_out_class': held_out,
            'meta_train_classes': config.meta_train_classes,
            'train_best_val_acc': train_stats.get('best_val_accuracy'),
            'eval_shots': {}
        }
        
        for k_shot, res in eval_res.items():
            rdict = res.to_dict()
            fold_data['eval_shots'][str(k_shot)] = rdict
            if k_shot == 1:
                all_1shot_acc.append(res.mean_accuracy)
                all_1shot_f1.append(res.mean_f1)
            elif k_shot == 5:
                all_5shot_acc.append(res.mean_accuracy)
                all_5shot_f1.append(res.mean_f1)
                
        with open(fold_dir / "fold_results.json", "w") as f:
            json.dump(fold_data, f, indent=2)
            
        fold_results[held_out] = fold_data
        
        logger.info(
            f"Fold {fold_idx} ({held_out}): 1-Shot Acc = {fold_data['eval_shots']['1']['mean_accuracy']*100:.2f}% | "
            f"5-Shot Acc = {fold_data['eval_shots']['5']['mean_accuracy']*100:.2f}%"
        )

    # Compute aggregate metrics across 5 folds
    n_folds = len(BREED_NAMES)
    
    mean_1shot_acc = float(np.mean(all_1shot_acc))
    std_1shot_acc = float(np.std(all_1shot_acc))
    ci95_1shot_acc = float(1.96 * std_1shot_acc / np.sqrt(n_folds))
    mean_1shot_f1 = float(np.mean(all_1shot_f1))
    std_1shot_f1 = float(np.std(all_1shot_f1))
    
    mean_5shot_acc = float(np.mean(all_5shot_acc))
    std_5shot_acc = float(np.std(all_5shot_acc))
    ci95_5shot_acc = float(1.96 * std_5shot_acc / np.sqrt(n_folds))
    mean_5shot_f1 = float(np.mean(all_5shot_f1))
    std_5shot_f1 = float(np.std(all_5shot_f1))
    
    loco_summary = {
        'experiment_id': audit_exp_id,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'total_folds': n_folds,
        'episodes_per_fold_shot': 20,
        'metrics_validation': {
            'task_definition': "2-Way Few-Shot Evaluation Episode (1 Unseen Breed + 1 Sampled Known Breed)",
            'label_mapping': "0: Unseen Held-out Class, 1: Sampled Known Class",
            'f1_type': "Macro-averaged F1 score (unweighted mean of per-class F1 for both classes)"
        },
        'aggregate_metrics': {
            '1_shot': {
                'mean_accuracy': round(mean_1shot_acc, 4),
                'std_accuracy': round(std_1shot_acc, 4),
                'ci_95_accuracy': round(ci95_1shot_acc, 4),
                'mean_macro_f1': round(mean_1shot_f1, 4),
                'std_macro_f1': round(std_1shot_f1, 4)
            },
            '5_shot': {
                'mean_accuracy': round(mean_5shot_acc, 4),
                'std_accuracy': round(std_5shot_acc, 4),
                'ci_95_accuracy': round(ci95_5shot_acc, 4),
                'mean_macro_f1': round(mean_5shot_f1, 4),
                'std_macro_f1': round(std_5shot_f1, 4)
            }
        },
        'per_fold_summary': fold_results
    }

    summary_path = audit_dir / "loco_summary.json"
    with open(summary_path, "w") as f:
        json.dump(loco_summary, f, indent=2)

    logger.info("=" * 60)
    logger.info(f"LOCO Audit Completed! Summary saved to {summary_path}")
    logger.info(f"Aggregate 1-Shot Acc: {mean_1shot_acc*100:.2f}% ± {std_1shot_acc*100:.2f}% (95% CI: ±{ci95_1shot_acc*100:.2f}%)")
    logger.info(f"Aggregate 5-Shot Acc: {mean_5shot_acc*100:.2f}% ± {std_5shot_acc*100:.2f}% (95% CI: ±{ci95_5shot_acc*100:.2f}%)")
    logger.info("=" * 60)

    return loco_summary


def main():
    logger.info("Starting MetaLearnHub Evaluation Audit...")
    inspector = DatasetInspector(RAW_DATA_DIR)
    inspector.scan()
    
    # Step 2: Reproducibility audit
    run_reproducibility_audit(inspector)
    
    # Step 3: Scientific baseline precision audit
    run_baseline_precision_audit(inspector)
    
    # Step 4 & 5: LOCO evaluation audit
    run_loco_evaluation_audit(inspector)
    
    logger.info("ALL AUDIT STEPS COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
