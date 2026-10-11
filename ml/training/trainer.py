"""
Meta-Training Loop for MetaLearnHub.

Implements episodic meta-training with logging, checkpointing,
and progress tracking.
"""

import time
import json
import uuid
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Callable

import torch
import numpy as np

from ml.config import ExperimentConfig, CHECKPOINTS_DIR, EXPERIMENTS_DIR
from ml.datasets.inspector import DatasetInspector
from ml.datasets.splitter import DatasetSplitter, DataSplit
from ml.datasets.cattle_dataset import (
    CattleDataset, EpisodicSampler,
    get_train_transforms, get_eval_transforms,
)
from ml.algorithms.prototypical_networks.protonet import PrototypicalNetworkAlgorithm

logger = logging.getLogger(__name__)


class TrainingState:
    """Tracks the current state of meta-training."""
    
    def __init__(self):
        self.experiment_id: str = ""
        self.status: str = "idle"  # idle, running, completed, failed, cancelled
        self.current_epoch: int = 0
        self.total_epochs: int = 0
        self.current_episode: int = 0
        self.episodes_per_epoch: int = 0
        self.train_loss: float = 0.0
        self.train_accuracy: float = 0.0
        self.val_loss: float = 0.0
        self.val_accuracy: float = 0.0
        self.best_val_accuracy: float = 0.0
        self.elapsed_seconds: float = 0.0
        self.start_time: Optional[float] = None
        self.epoch_losses: List[float] = []
        self.epoch_accuracies: List[float] = []
        self.val_losses: List[float] = []
        self.val_accuracies: List[float] = []
        self.error_message: str = ""
        self.checkpoint_path: str = ""
    
    def to_dict(self) -> dict:
        return {
            'experiment_id': self.experiment_id,
            'status': self.status,
            'current_epoch': self.current_epoch,
            'total_epochs': self.total_epochs,
            'current_episode': self.current_episode,
            'episodes_per_epoch': self.episodes_per_epoch,
            'train_loss': self.train_loss,
            'train_accuracy': self.train_accuracy,
            'val_loss': self.val_loss,
            'val_accuracy': self.val_accuracy,
            'best_val_accuracy': self.best_val_accuracy,
            'elapsed_seconds': round(self.elapsed_seconds, 2),
            'epoch_losses': self.epoch_losses,
            'epoch_accuracies': self.epoch_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'error_message': self.error_message,
            'checkpoint_path': self.checkpoint_path,
        }


# Global training state for API access
_current_training_state = TrainingState()
_cancel_requested = False


def get_training_state() -> TrainingState:
    """Get current training state."""
    return _current_training_state


def request_cancel():
    """Request cancellation of current training."""
    global _cancel_requested
    _cancel_requested = True


def get_device(config_device: str) -> torch.device:
    """Determine compute device."""
    if config_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config_device)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Note: full determinism may reduce performance
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class MetaTrainer:
    """
    Manages the complete meta-training pipeline.
    
    Pipeline:
        1. Load and inspect dataset
        2. Create leakage-safe split
        3. Build episodic sampler
        4. Train ProtoNet episodically
        5. Validate per epoch
        6. Save checkpoints
        7. Log everything
    """
    
    def __init__(
        self,
        config: ExperimentConfig,
        inspector: DatasetInspector,
        progress_callback: Optional[Callable] = None,
    ):
        self.config = config
        self.inspector = inspector
        self.progress_callback = progress_callback
        
        # Validate configuration
        config.validate()
        
        # Set device
        self.device = get_device(config.training.device)
        logger.info(f"Using device: {self.device}")
        
        # Set seed
        set_seed(config.random_seed)
    
    def train(self) -> dict:
        """
        Run the full meta-training pipeline.
        
        Returns:
            Experiment result dictionary
        """
        global _current_training_state, _cancel_requested
        _cancel_requested = False
        
        state = _current_training_state
        state.experiment_id = str(uuid.uuid4())[:8]
        state.status = "running"
        state.total_epochs = self.config.episode.num_epochs
        state.episodes_per_epoch = self.config.episode.episodes_per_epoch
        state.start_time = time.time()
        state.epoch_losses = []
        state.epoch_accuracies = []
        state.val_losses = []
        state.val_accuracies = []
        state.error_message = ""
        
        try:
            # Create split
            splitter = DatasetSplitter(self.inspector)
            split = splitter.create_split(
                held_out_class=self.config.held_out_class,
                seed=self.config.random_seed,
            )
            splitter.save_split(split)
            
            logger.info(
                f"Split created. Meta-train classes: {split.meta_train_classes}, "
                f"Held-out: {split.held_out_class}"
            )
            logger.info(
                f"Train images: {split.total_train_images}, "
                f"Val images: {split.total_val_images}, "
                f"Test images: {split.total_test_images}, "
                f"Unseen images: {split.total_unseen_images}"
            )
            
            # Build datasets
            train_transform = get_train_transforms(self.config.preprocess)
            eval_transform = get_eval_transforms(self.config.preprocess)
            
            train_paths = splitter.get_image_paths_for_split(split, 'train')
            val_paths = splitter.get_image_paths_for_split(split, 'val')
            
            train_dataset = CattleDataset(train_paths, transform=train_transform)
            val_dataset = CattleDataset(val_paths, transform=eval_transform)
            
            logger.info(
                f"Train dataset: {len(train_dataset)} images, "
                f"{train_dataset.num_classes} classes"
            )
            logger.info(
                f"Val dataset: {len(val_dataset)} images, "
                f"{val_dataset.num_classes} classes"
            )
            
            # Validate episode feasibility
            total_needed = self.config.episode.k_shot + self.config.episode.query_per_class
            for cls in train_dataset.classes:
                n_imgs = len(train_dataset.get_class_images(cls))
                if n_imgs < total_needed:
                    # Reduce query_per_class to fit
                    max_query = n_imgs - self.config.episode.k_shot
                    if max_query < 1:
                        raise ValueError(
                            f"Class '{cls}' has only {n_imgs} images. "
                            f"Cannot support {self.config.episode.k_shot}-shot episodes."
                        )
                    logger.warning(
                        f"Class '{cls}' has {n_imgs} images. "
                        f"Reducing query_per_class from {self.config.episode.query_per_class} "
                        f"to {max_query}"
                    )
                    self.config.episode.query_per_class = min(
                        self.config.episode.query_per_class, max_query
                    )
            
            # Initialize algorithm
            self.algorithm = PrototypicalNetworkAlgorithm(
                backbone_name=self.config.model.backbone,
                pretrained=self.config.model.pretrained,
                distance_metric=self.config.model.distance_metric,
                learning_rate=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                device=str(self.device),
                freeze_backbone=self.config.model.freeze_backbone,
            )
            self.algorithm.setup_scheduler(
                step_size=self.config.training.lr_step_size,
                gamma=self.config.training.lr_gamma,
            )
            
            best_val_acc = 0.0
            checkpoint_dir = CHECKPOINTS_DIR / state.experiment_id
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # ============ TRAINING LOOP ============
            for epoch in range(1, self.config.episode.num_epochs + 1):
                if _cancel_requested:
                    state.status = "cancelled"
                    logger.info("Training cancelled by user.")
                    break
                
                state.current_epoch = epoch
                
                # Create episodic sampler for this epoch
                train_sampler = EpisodicSampler(
                    dataset=train_dataset,
                    n_way=self.config.episode.n_way,
                    k_shot=self.config.episode.k_shot,
                    query_per_class=self.config.episode.query_per_class,
                    num_episodes=self.config.episode.episodes_per_epoch,
                    seed=self.config.random_seed + epoch,
                )
                
                # Train episode loop
                epoch_losses = []
                epoch_accs = []
                
                for ep_idx, episode in enumerate(train_sampler, 1):
                    if _cancel_requested:
                        break
                    
                    state.current_episode = ep_idx
                    
                    metrics = self.algorithm.meta_train_step(
                        support_images=episode['support_images'],
                        support_labels=episode['support_labels'],
                        query_images=episode['query_images'],
                        query_labels=episode['query_labels'],
                    )
                    
                    epoch_losses.append(metrics['loss'])
                    epoch_accs.append(metrics['accuracy'])
                    
                    state.train_loss = metrics['loss']
                    state.train_accuracy = metrics['accuracy']
                    state.elapsed_seconds = time.time() - state.start_time
                
                # Epoch statistics
                avg_loss = np.mean(epoch_losses)
                avg_acc = np.mean(epoch_accs)
                state.epoch_losses.append(float(avg_loss))
                state.epoch_accuracies.append(float(avg_acc))
                
                # Validation
                val_sampler = EpisodicSampler(
                    dataset=val_dataset,
                    n_way=min(self.config.episode.n_way, val_dataset.num_classes),
                    k_shot=self.config.episode.k_shot,
                    query_per_class=self.config.episode.query_per_class,
                    num_episodes=min(20, self.config.episode.episodes_per_epoch),
                    seed=self.config.random_seed,
                )
                
                val_losses = []
                val_accs = []
                for episode in val_sampler:
                    metrics = self.algorithm.meta_eval_step(
                        support_images=episode['support_images'],
                        support_labels=episode['support_labels'],
                        query_images=episode['query_images'],
                        query_labels=episode['query_labels'],
                    )
                    val_losses.append(metrics['loss'])
                    val_accs.append(metrics['accuracy'])
                
                val_avg_loss = np.mean(val_losses)
                val_avg_acc = np.mean(val_accs)
                state.val_loss = float(val_avg_loss)
                state.val_accuracy = float(val_avg_acc)
                state.val_losses.append(float(val_avg_loss))
                state.val_accuracies.append(float(val_avg_acc))
                
                # Step scheduler
                if self.algorithm.scheduler:
                    self.algorithm.scheduler.step()
                
                # Save best checkpoint
                if val_avg_acc > best_val_acc:
                    best_val_acc = val_avg_acc
                    state.best_val_accuracy = float(best_val_acc)
                    
                    ckpt_path = checkpoint_dir / "best_model.pt"
                    self.algorithm.save_checkpoint(
                        str(ckpt_path),
                        extra={
                            'epoch': epoch,
                            'val_accuracy': best_val_acc,
                            'config': self.config.to_dict(),
                        }
                    )
                    state.checkpoint_path = str(ckpt_path)
                
                logger.info(
                    f"Epoch {epoch}/{self.config.episode.num_epochs} | "
                    f"Train Loss: {avg_loss:.4f} | Train Acc: {avg_acc:.4f} | "
                    f"Val Loss: {val_avg_loss:.4f} | Val Acc: {val_avg_acc:.4f} | "
                    f"Best Val: {best_val_acc:.4f}"
                )
                
                if self.progress_callback:
                    self.progress_callback(state.to_dict())
            
            # Save final checkpoint
            final_ckpt_path = checkpoint_dir / "final_model.pt"
            self.algorithm.save_checkpoint(
                str(final_ckpt_path),
                extra={
                    'epoch': self.config.episode.num_epochs,
                    'config': self.config.to_dict(),
                }
            )
            
            if state.status != "cancelled":
                state.status = "completed"
            state.elapsed_seconds = time.time() - state.start_time
            
            # Save experiment record
            experiment_record = self._create_experiment_record(state, split)
            self._save_experiment_record(experiment_record)
            
            return experiment_record
            
        except Exception as e:
            state.status = "failed"
            state.error_message = str(e)
            logger.error(f"Training failed: {e}", exc_info=True)
            raise
    
    def _create_experiment_record(self, state: TrainingState, split: DataSplit) -> dict:
        """Create a comprehensive experiment record."""
        return {
            'experiment_id': state.experiment_id,
            'timestamp': datetime.now().isoformat(),
            'status': state.status,
            'config': self.config.to_dict(),
            'held_out_class': self.config.held_out_class,
            'meta_train_classes': self.config.meta_train_classes,
            'algorithm': self.config.algorithm,
            'backbone': self.config.model.backbone,
            'pretrained': self.config.model.pretrained,
            'n_way': self.config.episode.n_way,
            'k_shot': self.config.episode.k_shot,
            'query_per_class': self.config.episode.query_per_class,
            'episodes_per_epoch': self.config.episode.episodes_per_epoch,
            'num_epochs': self.config.episode.num_epochs,
            'learning_rate': self.config.training.learning_rate,
            'random_seed': self.config.random_seed,
            'device': str(self.device),
            'training_duration_seconds': round(state.elapsed_seconds, 2),
            'best_val_accuracy': state.best_val_accuracy,
            'final_train_loss': state.epoch_losses[-1] if state.epoch_losses else None,
            'final_train_accuracy': state.epoch_accuracies[-1] if state.epoch_accuracies else None,
            'epoch_losses': state.epoch_losses,
            'epoch_accuracies': state.epoch_accuracies,
            'val_losses': state.val_losses,
            'val_accuracies': state.val_accuracies,
            'checkpoint_path': state.checkpoint_path,
            'split_info': {
                'train_images': split.total_train_images,
                'val_images': split.total_val_images,
                'test_images': split.total_test_images,
                'unseen_images': split.total_unseen_images,
            },
        }
    
    def _save_experiment_record(self, record: dict):
        """Save experiment record to JSON."""
        exp_dir = EXPERIMENTS_DIR / record['experiment_id']
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        record_path = exp_dir / "experiment.json"
        with open(record_path, 'w') as f:
            json.dump(record, f, indent=2)
        
        # Also save config separately
        config_path = exp_dir / "config.json"
        self.config.save(config_path)
        
        logger.info(f"Experiment record saved to {record_path}")
