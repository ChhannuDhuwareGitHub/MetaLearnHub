"""
Corrected Evaluation Audit Script for MetaLearnHub.

TASK 1: Generate a corrected reproducibility comparison report from existing
        artifacts, in a NEW output location. Does NOT re-run evaluation — it
        reads the already-saved original and audit results, computes actual
        numerical differences, and flags protocol mismatches.

TASK 2: Run the missing 5-fold LOCO evaluation, training a fresh ProtoNet
        per fold (held-out class rotated) and evaluating 1-shot and 5-shot
        adaptation with group-level leakage prevention.

All outputs go to NEW unique directories under experiments/.
"""

import os
import sys
import json
import time
import uuid
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

from ml.config import (
    ExperimentConfig, PreprocessConfig, EpisodeConfig, TrainingConfig,
    BREED_NAMES, RAW_DATA_DIR, EXPERIMENTS_DIR,
)
from ml.datasets.inspector import DatasetInspector
from ml.datasets.splitter import DatasetSplitter
from ml.algorithms.prototypical_networks.protonet import PrototypicalNetworkAlgorithm
from ml.training.trainer import MetaTrainer
from ml.evaluation.evaluator import FewShotEvaluator, EvaluationResult


# ============================================================
# TASK 1 — Corrected Reproducibility Comparison
# ============================================================

def generate_corrected_comparison_report() -> Dict[str, Any]:
    """
    Build a corrected comparison report from existing artifact files.

    Reads:
      - Original eval results:  experiments/8133a306/evaluation/eval_{1,5}shot.json
      - Audit rerun results:    experiments/audit_reproducibility_8133a306/audit_summary.json
      - Original experiment record: experiments/8133a306/experiment.json

    Outputs to a NEW directory:
      experiments/corrected_comparison_<uid>/

    The old report at audit_reproducibility_8133a306/comparison_diff_report.json
    is preserved as-is (historical evidence).
    """
    logger.info("=" * 60)
    logger.info("TASK 1: Generating Corrected Reproducibility Comparison Report")
    logger.info("=" * 60)

    uid = str(uuid.uuid4())[:8]
    output_dir = EXPERIMENTS_DIR / f"corrected_comparison_{uid}"
    output_dir.mkdir(parents=True, exist_ok=True)

    original_exp_id = "8133a306"
    audit_exp_id = "audit_reproducibility_8133a306"

    # --- Load original experiment metadata ---
    orig_experiment_path = EXPERIMENTS_DIR / original_exp_id / "experiment.json"
    orig_config_path = EXPERIMENTS_DIR / original_exp_id / "config.json"
    orig_experiment = {}
    orig_config = {}
    if orig_experiment_path.exists():
        with open(orig_experiment_path) as f:
            orig_experiment = json.load(f)
    if orig_config_path.exists():
        with open(orig_config_path) as f:
            orig_config = json.load(f)

    # --- Load audit rerun summary ---
    audit_summary_path = EXPERIMENTS_DIR / audit_exp_id / "audit_summary.json"
    repro_summary_path = EXPERIMENTS_DIR / audit_exp_id / "reproducibility_summary.json"

    audit_summary = {}
    if audit_summary_path.exists():
        with open(audit_summary_path) as f:
            audit_summary = json.load(f)
    repro_summary = {}
    if repro_summary_path.exists():
        with open(repro_summary_path) as f:
            repro_summary = json.load(f)

    # Use whichever audit file has shot_results
    audit_shot_results = audit_summary.get("shot_results", {})
    if not audit_shot_results and repro_summary:
        audit_shot_results = repro_summary.get("shot_results", {})

    # --- Protocol metadata for comparability assessment ---
    orig_seed = orig_config.get("random_seed")
    orig_held_out = orig_config.get("held_out_class")
    orig_eval_episodes = orig_config.get("eval_episodes")
    orig_preprocess_size = orig_config.get("preprocess", {}).get("image_size")
    orig_checkpoint = orig_experiment.get("checkpoint_path", "")

    audit_seed = audit_summary.get("audit_seed") or repro_summary.get("random_seed")
    audit_held_out = audit_summary.get("held_out_class") or repro_summary.get("held_out_class")
    audit_eval_episodes = audit_summary.get("eval_episodes") or repro_summary.get("eval_episodes")

    protocol_metadata = {
        "original_experiment_id": original_exp_id,
        "audit_experiment_id": audit_exp_id,
        "original_seed": orig_seed,
        "audit_seed": audit_seed,
        "seeds_match": orig_seed == audit_seed,
        "original_held_out_class": orig_held_out,
        "audit_held_out_class": audit_held_out,
        "held_out_class_match": orig_held_out == audit_held_out,
        "original_eval_episodes": orig_eval_episodes,
        "audit_eval_episodes": audit_eval_episodes,
        "eval_episodes_match": orig_eval_episodes == audit_eval_episodes,
        "original_image_size": orig_preprocess_size,
        "original_checkpoint_path": orig_checkpoint,
        "original_eval_timestamp": orig_experiment.get("timestamp"),
        "audit_timestamp": audit_summary.get("audit_timestamp") or repro_summary.get("timestamp"),
    }

    # --- Per-shot comparison ---
    REPRO_TOLERANCE = 0.02  # 2 pp
    shot_comparisons = {}

    for shot_key in ["1", "5"]:
        orig_path = EXPERIMENTS_DIR / original_exp_id / "evaluation" / f"eval_{shot_key}shot.json"
        if not orig_path.exists():
            continue

        with open(orig_path) as f:
            orig_data = json.load(f)

        audit_data = audit_shot_results.get(shot_key, {})

        orig_acc = orig_data.get("mean_accuracy")
        audit_acc = audit_data.get("mean_accuracy")
        orig_f1 = orig_data.get("mean_f1")
        audit_f1 = audit_data.get("mean_f1")
        orig_classes = orig_data.get("class_names", [])
        audit_classes = audit_data.get("class_names", [])
        orig_episodes = orig_data.get("episode_accuracies", [])
        audit_episodes = audit_data.get("episode_accuracies", [])

        # Compute deltas
        acc_delta = round(audit_acc - orig_acc, 4) if (orig_acc is not None and audit_acc is not None) else None
        f1_delta = round(audit_f1 - orig_f1, 4) if (orig_f1 is not None and audit_f1 is not None) else None
        acc_reproduced = abs(acc_delta) <= REPRO_TOLERANCE if acc_delta is not None else None
        class_pairing_match = orig_classes == audit_classes
        episodes_match = orig_episodes == audit_episodes

        # Determine if protocols are directly comparable
        protocols_comparable = (
            protocol_metadata["seeds_match"]
            and protocol_metadata["held_out_class_match"]
            and protocol_metadata["eval_episodes_match"]
        )

        # Build honest explanation
        explanation_parts = []

        if not protocols_comparable:
            explanation_parts.append(
                "WARNING: Protocols are NOT directly comparable. "
                f"Seed match: {protocol_metadata['seeds_match']}, "
                f"held-out match: {protocol_metadata['held_out_class_match']}, "
                f"eval-episodes match: {protocol_metadata['eval_episodes_match']}."
            )

        if not class_pairing_match:
            explanation_parts.append(
                f"Known-class pairing differs: original used {orig_classes}, audit used {audit_classes}. "
                "This means the evaluator sampled a different known class as the second way in 2-way episodes, "
                "making per-episode accuracy sequences non-comparable even with the same seed."
            )

        if acc_delta is not None:
            if acc_reproduced:
                explanation_parts.append(
                    f"Accuracy within ±{REPRO_TOLERANCE*100:.0f}pp tolerance: "
                    f"original {orig_acc*100:.2f}% vs audit {audit_acc*100:.2f}% "
                    f"(delta {acc_delta*100:+.2f}pp)."
                )
            else:
                explanation_parts.append(
                    f"Accuracy DIFFERS beyond ±{REPRO_TOLERANCE*100:.0f}pp tolerance: "
                    f"original {orig_acc*100:.2f}% vs audit {audit_acc*100:.2f}% "
                    f"(delta {acc_delta*100:+.2f}pp)."
                )

        if f1_delta is not None:
            explanation_parts.append(
                f"Macro-F1: original {orig_f1*100:.2f}% vs audit {audit_f1*100:.2f}% "
                f"(delta {f1_delta*100:+.2f}pp)."
            )

        if not episodes_match:
            explanation_parts.append(
                "Per-episode accuracy sequences differ, confirming non-identical episode sampling."
            )

        shot_comparisons[f"{shot_key}shot"] = {
            "original_mean_accuracy": orig_acc,
            "audit_mean_accuracy": audit_acc,
            "accuracy_delta_pp": round(acc_delta * 100, 2) if acc_delta is not None else None,
            "accuracy_reproduced_within_2pp": acc_reproduced,
            "original_mean_f1": orig_f1,
            "audit_mean_f1": audit_f1,
            "f1_delta_pp": round(f1_delta * 100, 2) if f1_delta is not None else None,
            "original_class_pairing": orig_classes,
            "audit_class_pairing": audit_classes,
            "class_pairing_match": class_pairing_match,
            "original_episode_accuracies": orig_episodes,
            "audit_episode_accuracies": audit_episodes,
            "episode_sequences_match": episodes_match,
            "protocols_directly_comparable": protocols_comparable,
            "explanation": " ".join(explanation_parts),
        }

    report = {
        "report_id": f"corrected_comparison_{uid}",
        "generated_at": datetime.now().isoformat(),
        "purpose": (
            "Corrected reproducibility comparison replacing the previous hardcoded report. "
            "All values are computed from existing artifact files, not hardcoded."
        ),
        "protocol_metadata": protocol_metadata,
        "tolerance_pp": REPRO_TOLERANCE * 100,
        "shot_comparisons": shot_comparisons,
        "previous_report_location": str(EXPERIMENTS_DIR / audit_exp_id / "comparison_diff_report.json"),
        "previous_report_issue": (
            "The previous comparison_diff_report.json contained a hardcoded explanation "
            "claiming 'Exact match on mean accuracy (57.50%)' which was factually incorrect: "
            "the audit rerun produced 66.50% 1-shot accuracy, not 57.50%."
        ),
    }

    report_path = output_dir / "corrected_comparison_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Corrected comparison report saved to {report_path}")
    return report


# ============================================================
# TASK 2 — Five-Fold LOCO Evaluation
# ============================================================

def run_loco_evaluation() -> Dict[str, Any]:
    """
    Run Leave-One-Class-Out evaluation across all cattle breeds.

    For each of the 5 breeds:
      1. Hold out 1 breed entirely from meta-training.
      2. Train a ProtoNet on the remaining 4 breeds.
      3. Evaluate 1-shot and 5-shot adaptation on the held-out breed.

    All group-level leakage prevention is enforced by the existing
    EpisodicSampler and FewShotEvalDataset (runtime assertions on
    sample-group disjointness between support and query sets).

    Configuration:
      - Meta-train: 4-way, 5-shot, 2 query/class, 20 episodes/epoch, 3 epochs
      - Eval: 2-way (1 unseen + 1 known), 10 query/class, 20 episodes per shot
      - Seed: 42 (fixed across all folds)
      - Device: CPU (no GPU assumed)

    Outputs to: experiments/loco_audit_<uid>/
    """
    logger.info("=" * 60)
    logger.info("TASK 2: Five-Fold LOCO Evaluation")
    logger.info("=" * 60)

    # Verify eligible classes from the dataset
    inspector = DatasetInspector(RAW_DATA_DIR)
    inspector.scan()

    discovered_breeds = sorted(set(g.breed for g in inspector.sample_groups.values()))
    logger.info(f"Discovered breeds in dataset: {discovered_breeds}")
    logger.info(f"Configured BREED_NAMES: {BREED_NAMES}")

    # Verify all BREED_NAMES exist in dataset
    missing_breeds = [b for b in BREED_NAMES if b not in discovered_breeds]
    if missing_breeds:
        logger.error(f"Breeds in BREED_NAMES but not found in dataset: {missing_breeds}")
        raise ValueError(f"Cannot run LOCO: missing breeds {missing_breeds}")

    eligible_classes = list(BREED_NAMES)
    n_folds = len(eligible_classes)
    logger.info(f"Eligible LOCO folds: {n_folds} (one per breed)")

    uid = str(uuid.uuid4())[:8]
    audit_exp_id = f"loco_audit_{uid}"
    audit_dir = EXPERIMENTS_DIR / audit_exp_id
    audit_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42

    # LOCO config (consistent across all folds)
    loco_config_template = {
        "meta_train_n_way": 4,
        "meta_train_k_shot": 5,
        "meta_train_query_per_class": 2,
        "episodes_per_epoch": 20,
        "num_epochs": 3,
        "eval_n_way": 2,
        "eval_query_per_class": 10,
        "eval_episodes": 20,
        "eval_shots": [1, 5],
        "random_seed": seed,
        "image_size": 224,
        "device": device,
        "backbone": "resnet18",
        "pretrained": True,
        "distance_metric": "euclidean",
    }

    # Save config template
    with open(audit_dir / "loco_config.json", "w") as f:
        json.dump(loco_config_template, f, indent=2)

    fold_results = {}
    completed_folds = 0
    failed_folds = []

    all_1shot_acc = []
    all_1shot_f1 = []
    all_5shot_acc = []
    all_5shot_f1 = []

    total_start_time = time.time()

    for fold_idx, held_out in enumerate(eligible_classes, start=1):
        breed_slug = held_out.lower().replace(" ", "_")
        fold_id = f"fold_{fold_idx}_{breed_slug}"
        fold_dir = audit_dir / fold_id
        fold_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\n{'='*60}")
        logger.info(f"FOLD {fold_idx}/{n_folds}: Held-out = '{held_out}'")
        logger.info(f"Meta-train classes: {[b for b in eligible_classes if b != held_out]}")
        logger.info(f"{'='*60}")

        fold_start = time.time()

        try:
            config = ExperimentConfig(
                experiment_name=f"loco_{fold_id}",
                held_out_class=held_out,
                random_seed=seed,
                preprocess=PreprocessConfig(image_size=224),
                episode=EpisodeConfig(
                    n_way=4,
                    k_shot=5,
                    query_per_class=2,
                    episodes_per_epoch=20,
                    num_epochs=3,
                ),
                training=TrainingConfig(device=device, num_workers=0),
                eval_shots=[1, 5],
                eval_episodes=20,
            )
            config.validate()

            # Record split metadata
            splitter = DatasetSplitter(inspector)
            split = splitter.create_split(held_out_class=held_out, seed=seed)
            split_info = {
                "held_out_class": held_out,
                "meta_train_classes": config.meta_train_classes,
                "held_out_groups": split.assignments[held_out].get("total_groups"),
                "train_images": split.total_train_images,
                "val_images": split.total_val_images,
                "test_images": split.total_test_images,
                "unseen_images": split.total_unseen_images,
            }

            # Verify no leakage
            violations = split.validate_no_leakage(inspector)
            if violations:
                raise RuntimeError(f"Leakage detected in fold {fold_idx}: {violations}")

            # Train
            trainer = MetaTrainer(config, inspector)
            train_record = trainer.train()
            train_exp_id = train_record["experiment_id"]

            # Evaluate using the trained algorithm
            evaluator = FewShotEvaluator(trainer.algorithm, inspector, config)
            eval_results = evaluator.evaluate_all_shots(n_way=2, query_per_class=10)

            fold_data = {
                "fold_index": fold_idx,
                "fold_id": fold_id,
                "held_out_class": held_out,
                "meta_train_classes": config.meta_train_classes,
                "random_seed": seed,
                "training_experiment_id": train_exp_id,
                "training_duration_s": round(train_record.get("training_duration_seconds", 0), 2),
                "best_val_accuracy": train_record.get("best_val_accuracy"),
                "split_info": split_info,
                "status": "completed",
                "eval_shots": {},
            }

            for k_shot, res in eval_results.items():
                rdict = res.to_dict()
                fold_data["eval_shots"][str(k_shot)] = rdict

                if k_shot == 1:
                    all_1shot_acc.append(res.mean_accuracy)
                    all_1shot_f1.append(res.mean_f1)
                elif k_shot == 5:
                    all_5shot_acc.append(res.mean_accuracy)
                    all_5shot_f1.append(res.mean_f1)

            fold_data["fold_duration_s"] = round(time.time() - fold_start, 2)

            with open(fold_dir / "fold_results.json", "w") as f:
                json.dump(fold_data, f, indent=2)

            fold_results[held_out] = fold_data
            completed_folds += 1

            logger.info(
                f"Fold {fold_idx} COMPLETED | "
                f"1-shot: {fold_data['eval_shots']['1']['mean_accuracy']*100:.2f}% | "
                f"5-shot: {fold_data['eval_shots']['5']['mean_accuracy']*100:.2f}%"
            )

        except Exception as e:
            logger.error(f"Fold {fold_idx} ({held_out}) FAILED: {e}", exc_info=True)
            failed_folds.append({
                "fold_index": fold_idx,
                "held_out_class": held_out,
                "error": str(e),
            })
            fold_results[held_out] = {
                "fold_index": fold_idx,
                "held_out_class": held_out,
                "status": "failed",
                "error": str(e),
            }
            with open(fold_dir / "fold_results.json", "w") as f:
                json.dump(fold_results[held_out], f, indent=2)

    # --- Aggregate metrics (only from completed folds) ---
    aggregate = {}
    if completed_folds > 0:
        for shot_label, accs, f1s in [
            ("1_shot", all_1shot_acc, all_1shot_f1),
            ("5_shot", all_5shot_acc, all_5shot_f1),
        ]:
            if accs:
                n = len(accs)
                mean_acc = float(np.mean(accs))
                std_acc = float(np.std(accs))
                ci95_acc = float(1.96 * std_acc / np.sqrt(n)) if n > 1 else float("nan")
                mean_f1 = float(np.mean(f1s))
                std_f1 = float(np.std(f1s))

                aggregate[shot_label] = {
                    "completed_folds": n,
                    "mean_accuracy": round(mean_acc, 4),
                    "std_accuracy": round(std_acc, 4),
                    "ci_95_accuracy": round(ci95_acc, 4) if not np.isnan(ci95_acc) else None,
                    "mean_macro_f1": round(mean_f1, 4),
                    "std_macro_f1": round(std_f1, 4),
                    "per_fold_accuracies": {
                        breed: fold_results[breed]["eval_shots"][shot_label.replace("_shot", "")]["mean_accuracy"]
                        for breed in eligible_classes
                        if fold_results.get(breed, {}).get("status") == "completed"
                    },
                    "per_fold_f1s": {
                        breed: fold_results[breed]["eval_shots"][shot_label.replace("_shot", "")]["mean_f1"]
                        for breed in eligible_classes
                        if fold_results.get(breed, {}).get("status") == "completed"
                    },
                }

    total_duration = round(time.time() - total_start_time, 2)

    loco_summary = {
        "experiment_id": audit_exp_id,
        "generated_at": datetime.now().isoformat(),
        "total_folds_attempted": n_folds,
        "completed_folds": completed_folds,
        "failed_folds": len(failed_folds),
        "failed_fold_details": failed_folds,
        "eligible_classes": eligible_classes,
        "loco_config": loco_config_template,
        "total_duration_s": total_duration,
        "evaluation_protocol": {
            "task": "2-Way Few-Shot Classification (1 unseen held-out + 1 sampled known breed)",
            "label_mapping": "0 = Unseen Held-out Class, 1 = Sampled Known Class",
            "f1_type": "Macro-averaged F1 (unweighted mean of per-class F1)",
            "leakage_prevention": (
                "Group-level disjointness enforced at episode construction: "
                "base images and their _c counterparts share a sample-group ID "
                "and are never split across support and query sets."
            ),
        },
        "aggregate_metrics": aggregate,
        "per_fold_results": fold_results,
    }

    summary_path = audit_dir / "loco_summary.json"
    with open(summary_path, "w") as f:
        json.dump(loco_summary, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info(f"LOCO EVALUATION {'COMPLETED' if not failed_folds else 'PARTIALLY COMPLETED'}")
    logger.info(f"Completed folds: {completed_folds}/{n_folds}")
    logger.info(f"Total duration: {total_duration:.1f}s")
    logger.info(f"Summary: {summary_path}")
    if completed_folds > 0 and "1_shot" in aggregate:
        a1 = aggregate["1_shot"]
        logger.info(f"1-Shot Aggregate: {a1['mean_accuracy']*100:.2f}% ± {a1['std_accuracy']*100:.2f}%")
    if completed_folds > 0 and "5_shot" in aggregate:
        a5 = aggregate["5_shot"]
        logger.info(f"5-Shot Aggregate: {a5['mean_accuracy']*100:.2f}% ± {a5['std_accuracy']*100:.2f}%")
    logger.info("=" * 60)

    return loco_summary


# ============================================================
# Main entry point
# ============================================================

def main():
    logger.info("=" * 60)
    logger.info("MetaLearnHub — Corrected Audit Runner")
    logger.info("=" * 60)

    # TASK 1: Corrected reproducibility comparison
    comparison_report = generate_corrected_comparison_report()

    # TASK 2: Five-fold LOCO evaluation
    loco_summary = run_loco_evaluation()

    logger.info("\n" + "=" * 60)
    logger.info("ALL AUDIT TASKS COMPLETED")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
