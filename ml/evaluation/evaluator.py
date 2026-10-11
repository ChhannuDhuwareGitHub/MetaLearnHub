"""
Few-Shot Evaluation Module for MetaLearnHub.

Evaluates meta-learned models on held-out classes using
1-shot, 3-shot, 5-shot, and 10-shot protocols.

Implements proper evaluation methodology:
- Support images are never reused as query images
- Multiple evaluation episodes with mean ± std reporting
- Confusion matrices for multi-way evaluation
- Separate adaptation and inference timing
"""

import time
import json
import uuid
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from datetime import datetime

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix as compute_confusion_matrix,
)

from ml.config import ExperimentConfig, EXPERIMENTS_DIR
from ml.datasets.inspector import DatasetInspector
from ml.datasets.splitter import DatasetSplitter
from ml.datasets.cattle_dataset import (
    FewShotEvalDataset, get_eval_transforms,
)
from ml.algorithms.prototypical_networks.protonet import PrototypicalNetworkAlgorithm

logger = logging.getLogger(__name__)


class EvaluationResult:
    """Stores results from a single evaluation run."""
    
    def __init__(self):
        self.k_shot: int = 0
        self.n_way: int = 0
        self.num_episodes: int = 0
        self.held_out_class: str = ""
        self.class_names: List[str] = []
        
        # Per-episode metrics
        self.episode_accuracies: List[float] = []
        self.episode_precisions: List[float] = []
        self.episode_recalls: List[float] = []
        self.episode_f1s: List[float] = []
        
        # Aggregated metrics
        self.mean_accuracy: float = 0.0
        self.std_accuracy: float = 0.0
        self.mean_precision: float = 0.0
        self.std_precision: float = 0.0
        self.mean_recall: float = 0.0
        self.std_recall: float = 0.0
        self.mean_f1: float = 0.0
        self.std_f1: float = 0.0
        
        # Timing
        self.total_adaptation_time: float = 0.0
        self.total_inference_time: float = 0.0
        self.mean_adaptation_time: float = 0.0
        self.mean_inference_time: float = 0.0
        
        # Confusion matrix (aggregated across episodes)
        self.confusion_matrix: Optional[List[List[int]]] = None
        
        # Per-episode details
        self.episode_details: List[dict] = []
    
    def compute_aggregates(self):
        """Compute mean ± std from per-episode metrics."""
        if self.episode_accuracies:
            self.mean_accuracy = float(np.mean(self.episode_accuracies))
            self.std_accuracy = float(np.std(self.episode_accuracies))
        if self.episode_precisions:
            self.mean_precision = float(np.mean(self.episode_precisions))
            self.std_precision = float(np.std(self.episode_precisions))
        if self.episode_recalls:
            self.mean_recall = float(np.mean(self.episode_recalls))
            self.std_recall = float(np.std(self.episode_recalls))
        if self.episode_f1s:
            self.mean_f1 = float(np.mean(self.episode_f1s))
            self.std_f1 = float(np.std(self.episode_f1s))
        if self.num_episodes > 0:
            self.mean_adaptation_time = self.total_adaptation_time / self.num_episodes
            self.mean_inference_time = self.total_inference_time / self.num_episodes

    @property
    def ci_95_accuracy(self) -> float:
        if self.num_episodes > 0:
            return float(1.96 * self.std_accuracy / np.sqrt(self.num_episodes))
        return 0.0
    
    def to_dict(self) -> dict:
        return {
            'k_shot': self.k_shot,
            'n_way': self.n_way,
            'num_episodes': self.num_episodes,
            'held_out_class': self.held_out_class,
            'class_names': self.class_names,
            'mean_accuracy': round(self.mean_accuracy, 4),
            'std_accuracy': round(self.std_accuracy, 4),
            'mean_precision': round(self.mean_precision, 4),
            'std_precision': round(self.std_precision, 4),
            'mean_recall': round(self.mean_recall, 4),
            'std_recall': round(self.std_recall, 4),
            'mean_f1': round(self.mean_f1, 4),
            'std_f1': round(self.std_f1, 4),
            'mean_adaptation_time_ms': round(self.mean_adaptation_time * 1000, 2),
            'mean_inference_time_ms': round(self.mean_inference_time * 1000, 2),
            'total_adaptation_time_s': round(self.total_adaptation_time, 3),
            'total_inference_time_s': round(self.total_inference_time, 3),
            'confusion_matrix': self.confusion_matrix,
            'episode_accuracies': [round(a, 4) for a in self.episode_accuracies],
            'episode_details': self.episode_details,
            'ci_95_accuracy': round(1.96 * self.std_accuracy / max(1, np.sqrt(self.num_episodes)), 4) if self.num_episodes > 0 else 0,
        }


class FewShotEvaluator:
    """
    Evaluates a meta-learned model on held-out classes using few-shot protocols.
    
    For each shot count (1, 3, 5, 10):
        1. Sample K support images from the unseen breed
        2. Adapt (compute prototype)
        3. Classify remaining images as queries
        4. Record metrics
        5. Repeat for num_episodes
        6. Report mean ± std
    """
    
    def __init__(
        self,
        algorithm: PrototypicalNetworkAlgorithm,
        inspector: DatasetInspector,
        config: ExperimentConfig,
    ):
        self.algorithm = algorithm
        self.inspector = inspector
        self.config = config
        self.device = next(algorithm.get_model().parameters()).device
    
    def evaluate_all_shots(
        self,
        n_way: int = 2,
        query_per_class: int = 10,
    ) -> Dict[int, EvaluationResult]:
        """
        Run evaluation for all configured shot counts.
        
        Returns:
            Dict mapping shot_count -> EvaluationResult
        """
        results = {}
        
        for k_shot in self.config.eval_shots:
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating {k_shot}-shot on held-out class: {self.config.held_out_class}")
            logger.info(f"{'='*60}")
            
            try:
                result = self.evaluate_shot(
                    k_shot=k_shot,
                    n_way=n_way,
                    query_per_class=query_per_class,
                    num_episodes=self.config.eval_episodes,
                )
                results[k_shot] = result
                
                logger.info(
                    f"{k_shot}-shot: Accuracy = {result.mean_accuracy:.4f} "
                    f"± {result.std_accuracy:.4f}, "
                    f"F1 = {result.mean_f1:.4f} ± {result.std_f1:.4f}"
                )
            except Exception as e:
                logger.error(f"Failed {k_shot}-shot evaluation: {e}")
                continue
        
        return results
    
    def evaluate_shot(
        self,
        k_shot: int,
        n_way: int = 2,
        query_per_class: int = 10,
        num_episodes: int = 50,
    ) -> EvaluationResult:
        """
        Evaluate a specific shot count.
        
        Args:
            k_shot: Number of support images
            n_way: Number of classes in evaluation
            query_per_class: Query images per class
            num_episodes: Number of evaluation episodes
        
        Returns:
            EvaluationResult with mean/std metrics
        """
        result = EvaluationResult()
        result.k_shot = k_shot
        result.n_way = n_way
        result.num_episodes = num_episodes
        result.held_out_class = self.config.held_out_class
        
        # Build evaluation dataset
        eval_transform = get_eval_transforms(self.config.preprocess)
        splitter = DatasetSplitter(self.inspector)
        split = splitter.create_split(
            held_out_class=self.config.held_out_class,
            seed=self.config.random_seed,
        )
        
        unseen_paths = splitter.get_image_paths_for_split(split, 'unseen')
        known_paths = splitter.get_image_paths_for_split(split, 'test')
        
        # If not enough known test images, use val images too
        if len(known_paths) < (n_way - 1) * (k_shot + query_per_class):
            known_paths.extend(splitter.get_image_paths_for_split(split, 'val'))
        
        # Check available unseen images
        max_unseen_query = len(unseen_paths) - k_shot
        if max_unseen_query < 1:
            raise ValueError(
                f"Not enough unseen images. Have {len(unseen_paths)}, "
                f"need at least {k_shot + 1} for {k_shot}-shot."
            )
        actual_query = min(query_per_class, max_unseen_query)
        
        eval_dataset = FewShotEvalDataset(
            unseen_images=unseen_paths,
            known_images=known_paths,
            transform=eval_transform,
            seed=self.config.random_seed,
        )
        
        # Aggregated confusion matrix
        all_preds = []
        all_labels = []
        
        for ep_idx in range(num_episodes):
            # Re-seed for each episode but deterministically
            eval_dataset.rng.seed(self.config.random_seed + ep_idx * 1000)
            
            try:
                episode = eval_dataset.sample_episode(
                    k_shot=k_shot,
                    n_way=n_way,
                    query_per_class=actual_query,
                    include_known_classes=(n_way > 1),
                )
            except Exception as e:
                logger.warning(f"Episode {ep_idx} sampling failed: {e}")
                continue
            
            # Adaptation timing
            t_adapt_start = time.time()
            adapted_state = self.algorithm.adapt(
                support_images=episode['support_images'],
                support_labels=episode['support_labels'],
            )
            t_adapt = time.time() - t_adapt_start
            
            # Inference timing
            t_infer_start = time.time()
            predictions, confidences = self.algorithm.predict(
                query_images=episode['query_images'],
                adapted_state=adapted_state,
            )
            t_infer = time.time() - t_infer_start
            
            # Compute episode metrics (using macro-averaging across all N classes)
            labels_np = episode['query_labels'].numpy()
            preds_np = predictions.numpy()
            
            accuracy = accuracy_score(labels_np, preds_np)
            precision = precision_score(labels_np, preds_np, average='macro', zero_division=0)
            recall = recall_score(labels_np, preds_np, average='macro', zero_division=0)
            f1 = f1_score(labels_np, preds_np, average='macro', zero_division=0)
            
            result.episode_accuracies.append(float(accuracy))
            result.episode_precisions.append(float(precision))
            result.episode_recalls.append(float(recall))
            result.episode_f1s.append(float(f1))
            result.total_adaptation_time += t_adapt
            result.total_inference_time += t_infer
            
            all_preds.extend(preds_np.tolist())
            all_labels.extend(labels_np.tolist())
            
            result.episode_details.append({
                'episode': ep_idx,
                'accuracy': round(float(accuracy), 4),
                'precision': round(float(precision), 4),
                'recall': round(float(recall), 4),
                'f1': round(float(f1), 4),
                'adaptation_time_ms': round(t_adapt * 1000, 2),
                'inference_time_ms': round(t_infer * 1000, 2),
                'class_names': episode['class_names'],
                'query_labels': labels_np.tolist(),
                'predictions': preds_np.tolist(),
                'confidences': [round(float(c), 4) for c in confidences.numpy()],
                'support_group_ids': episode.get('support_group_ids', []),
                'query_group_ids': episode.get('query_group_ids', []),
            })
        
        # Compute aggregated confusion matrix
        if all_preds and all_labels:
            cm = compute_confusion_matrix(
                all_labels, all_preds,
                labels=list(range(n_way))
            )
            result.confusion_matrix = cm.tolist()
            if episode:
                result.class_names = episode.get('class_names', [])
        
        result.num_episodes = len(result.episode_accuracies)
        result.compute_aggregates()
        
        return result
    
    def save_results(
        self,
        results: Dict[int, EvaluationResult],
        experiment_id: str = None,
    ) -> Path:
        """Save evaluation results to the experiments directory."""
        if experiment_id is None:
            experiment_id = str(uuid.uuid4())[:8]
        
        eval_dir = EXPERIMENTS_DIR / experiment_id / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)
        
        # Save per-shot results
        summary = {
            'experiment_id': experiment_id,
            'timestamp': datetime.now().isoformat(),
            'held_out_class': self.config.held_out_class,
            'meta_train_classes': self.config.meta_train_classes,
            'algorithm': self.config.algorithm,
            'backbone': self.config.model.backbone,
            'random_seed': self.config.random_seed,
            'shot_results': {},
        }
        
        for k_shot, result in results.items():
            result_dict = result.to_dict()
            summary['shot_results'][str(k_shot)] = result_dict
            
            # Save individual result file
            shot_path = eval_dir / f"eval_{k_shot}shot.json"
            with open(shot_path, 'w') as f:
                json.dump(result_dict, f, indent=2)
        
        # Save summary
        summary_path = eval_dir / "evaluation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Evaluation results saved to {eval_dir}")
        return eval_dir


class BaselineEvaluator:
    """
    Conventional ResNet-18 baseline for comparison.
    
    Uses a standard supervised classifier trained on meta-training classes,
    then attempts to classify held-out class images.
    
    This baseline does NOT perform few-shot adaptation — it demonstrates
    the limitation that meta-learning aims to address.
    """
    
    def __init__(
        self,
        inspector: DatasetInspector,
        config: ExperimentConfig,
    ):
        self.inspector = inspector
        self.config = config
    
    def train_and_evaluate(self) -> dict:
        """
        Train a conventional ResNet-18 classifier and evaluate.
        
        The baseline:
        1. Trains on meta-training classes using standard supervised learning
        2. Evaluates on test split of meta-training classes (known accuracy)
        3. Attempts to classify held-out class (expected to fail)
        4. Reports both results
        """
        from ml.training.trainer import get_device, set_seed
        from ml.datasets.splitter import DatasetSplitter
        from ml.datasets.cattle_dataset import CattleDataset, get_train_transforms, get_eval_transforms
        from torchvision import models
        import torch.nn as nn
        import torch.optim as optim
        
        set_seed(self.config.random_seed)
        device = get_device(self.config.training.device)
        
        # Create split
        splitter = DatasetSplitter(self.inspector)
        split = splitter.create_split(
            held_out_class=self.config.held_out_class,
            seed=self.config.random_seed,
        )
        
        # Build datasets
        train_paths = splitter.get_image_paths_for_split(split, 'train')
        val_paths = splitter.get_image_paths_for_split(split, 'val')
        test_paths = splitter.get_image_paths_for_split(split, 'test')
        unseen_paths = splitter.get_image_paths_for_split(split, 'unseen')
        
        train_transform = get_train_transforms(self.config.preprocess)
        eval_transform = get_eval_transforms(self.config.preprocess)
        
        train_dataset = CattleDataset(train_paths, transform=train_transform)
        val_dataset = CattleDataset(val_paths, transform=eval_transform)
        test_dataset = CattleDataset(test_paths, transform=eval_transform)
        
        num_classes = train_dataset.num_classes
        
        # Create standard ResNet-18 classifier
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if self.config.model.pretrained else None)
        model.fc = nn.Linear(512, num_classes)
        model = model.to(device)
        
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()
        
        # Train
        from torch.utils.data import DataLoader
        
        train_loader = DataLoader(
            train_dataset, batch_size=32, shuffle=True,
            num_workers=0, pin_memory=False
        )
        val_loader = DataLoader(
            val_dataset, batch_size=32, shuffle=False,
            num_workers=0
        )
        
        best_val_acc = 0.0
        num_epochs = min(30, self.config.episode.num_epochs)
        
        logger.info(f"Training baseline ResNet-18 for {num_epochs} epochs...")
        
        for epoch in range(1, num_epochs + 1):
            model.train()
            train_losses = []
            train_correct = 0
            train_total = 0
            
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
                preds = outputs.argmax(dim=1)
                train_correct += (preds == labels).sum().item()
                train_total += labels.size(0)
            
            # Validate
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    preds = outputs.argmax(dim=1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)
            
            val_acc = val_correct / max(1, val_total)
            train_acc = train_correct / max(1, train_total)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            
            if epoch % 5 == 0:
                logger.info(
                    f"Baseline Epoch {epoch}/{num_epochs} | "
                    f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}"
                )
        
        # Evaluate on test set (known classes)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
        model.eval()
        test_preds = []
        test_labels_list = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                test_preds.extend(preds.cpu().numpy())
                test_labels_list.extend(labels.numpy())
        
        known_accuracy = accuracy_score(test_labels_list, test_preds) if test_labels_list else 0
        known_f1 = f1_score(test_labels_list, test_preds, average='macro', zero_division=0) if test_labels_list else 0
        
        # Attempt to classify unseen class (the baseline cannot adapt)
        # Just feed unseen images through the model — expected to misclassify
        unseen_dataset = CattleDataset(unseen_paths, transform=eval_transform)
        unseen_loader = DataLoader(unseen_dataset, batch_size=32, shuffle=False, num_workers=0)
        
        unseen_preds = []
        with torch.no_grad():
            for images, _ in unseen_loader:
                images = images.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                max_conf, preds = probs.max(dim=1)
                unseen_preds.extend([
                    {'predicted_class': int(p), 'confidence': float(c)}
                    for p, c in zip(preds.cpu(), max_conf.cpu())
                ])
        
        result = {
            'experiment_id': f"baseline_{str(uuid.uuid4())[:8]}",
            'timestamp': datetime.now().isoformat(),
            'method': 'conventional_resnet18',
            'held_out_class': self.config.held_out_class,
            'meta_train_classes': self.config.meta_train_classes,
            'num_epochs': num_epochs,
            'known_class_accuracy': round(float(known_accuracy), 4),
            'known_class_f1': round(float(known_f1), 4),
            'best_val_accuracy': round(float(best_val_acc), 4),
            'unseen_class_note': (
                "The baseline cannot adapt to unseen classes. "
                "All unseen images are classified as one of the known training classes. "
                "This demonstrates the limitation that meta-learning addresses."
            ),
            'unseen_predictions_sample': unseen_preds[:10],
        }
        
        # Save
        eval_dir = EXPERIMENTS_DIR / result['experiment_id']
        eval_dir.mkdir(parents=True, exist_ok=True)
        with open(eval_dir / "baseline_result.json", 'w') as f:
            json.dump(result, f, indent=2)
        
        return result
