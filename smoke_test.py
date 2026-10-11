import os
import json
import torch
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO)

from ml.config import ExperimentConfig, PreprocessConfig, EpisodeConfig, TrainingConfig
from ml.training.trainer import MetaTrainer
from ml.evaluation.evaluator import FewShotEvaluator

def run_smoke_test():
    print("=" * 60)
    print("MetaLearnHub Smoke Test")
    print("=" * 60)
    
    # 1. Detect GPU/CUDA
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"
    
    print(f"CUDA Available: {cuda_available}")
    print(f"GPU Name: {gpu_name}")
    print(f"Device Count: {device_count}")
    
    # We will use AMP (mixed precision) implicitly if CUDA is available by letting 
    # algorithms handle it, but wait, `prototypical_networks/protonet.py` might not 
    # have torch.cuda.amp.autocast explicitly. We will set device="cuda".
    target_device = "cuda" if cuda_available else "cpu"
    print(f"Using Device: {target_device}")
    
    # 2. Configure Smoke Test
    config = ExperimentConfig(
        experiment_name="smoke_test",
        held_out_class="Red Dane cattle",
        preprocess=PreprocessConfig(
            image_size=128,
            center_crop_size=128
        ),
        episode=EpisodeConfig(
            n_way=4, # User asked for 4 meta-training classes, wait, n_way = 4
            k_shot=5,
            query_per_class=2,
            episodes_per_epoch=25,
            num_epochs=2
        ),
        training=TrainingConfig(
            device=target_device,
            num_workers=0 # avoid multiprocessing issues in Windows
        ),
        eval_shots=[1, 5],
        eval_episodes=10
    )
    
    # In LOCO, we have 5 total classes, 1 held out. So 4 available for training.
    # We can set n_way=4.
    
    print("-" * 40)
    print("Configuration:")
    print(f"- Held out class: {config.held_out_class}")
    print(f"- Meta-training classes: {len(config.meta_train_classes)}")
    print(f"- N-way: {config.episode.n_way}, K-shot: {config.episode.k_shot}")
    print(f"- Image Size: {config.preprocess.image_size}x{config.preprocess.image_size}")
    print(f"- Episodes per epoch: {config.episode.episodes_per_epoch}")
    print(f"- Epochs: {config.episode.num_epochs}")
    print(f"- Total Training Episodes: {config.episode.episodes_per_epoch * config.episode.num_epochs}")
    print(f"- Evaluation Episodes: {config.eval_episodes}")
    print("-" * 40)
    
    from ml.datasets.inspector import DatasetInspector
    from ml.config import RAW_DATA_DIR
    
    print("\nScanning dataset...")
    inspector = DatasetInspector(RAW_DATA_DIR)
    inspector.scan()
    
    # 3. Run Training
    experiment_id = "8133a306"
    print(f"\nSkipping training, using experiment_id {experiment_id}")
    
    # 4. Run Evaluation
    print("\nStarting Evaluation...")
    from ml.algorithms.prototypical_networks.protonet import PrototypicalNetworkAlgorithm
    algorithm = PrototypicalNetworkAlgorithm(device=target_device)
    algorithm.load_checkpoint(f"checkpoints/{experiment_id}/best_model.pt")
    
    evaluator = FewShotEvaluator(algorithm, inspector, config)
    eval_results = evaluator.evaluate_all_shots()
    
    print("\nEvaluation Completed.")
    for shot, res in eval_results.items():
        print(f"{shot}-Shot Accuracy: {res.mean_accuracy*100:.2f}% ± {res.ci_95_accuracy*100:.2f}%")
        print(f"{shot}-Shot F1 Score: {res.mean_f1*100:.2f}%")
        
    print(f"\nExperiment Record saved at: checkpoints/{experiment_id}")

if __name__ == "__main__":
    run_smoke_test()
