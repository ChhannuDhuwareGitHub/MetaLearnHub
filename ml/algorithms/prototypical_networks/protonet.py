"""
Prototypical Networks Implementation for MetaLearnHub.

Reference: Snell et al., "Prototypical Networks for Few-Shot Learning", NeurIPS 2017.

Architecture:
    Image → Backbone → Embedding → Prototype (class mean) → Distance → Prediction

Each class prototype is the mean embedding of its support examples.
Classification is done by finding the nearest prototype to the query embedding.
"""

from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.algorithms.base import MetaLearningAlgorithm
from ml.backbones.resnet import get_backbone


class PrototypicalNetwork(nn.Module):
    """
    Prototypical Network model.
    
    Computes class prototypes as the mean embedding of support examples,
    then classifies queries based on distance to prototypes.
    """
    
    def __init__(
        self,
        backbone_name: str = "resnet18",
        pretrained: bool = True,
        distance_metric: str = "euclidean",
        freeze_backbone: bool = False,
    ):
        super().__init__()
        
        self.backbone = get_backbone(
            name=backbone_name,
            pretrained=pretrained,
            freeze=freeze_backbone,
        )
        self.distance_metric = distance_metric
        self.embedding_dim = self.backbone.embedding_dim
    
    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for a single episode.
        
        Args:
            support_images: [N*K, 3, H, W]
            support_labels: [N*K] - class labels (0 to N-1)
            query_images: [N*Q, 3, H, W]
        
        Returns:
            log_probabilities: [N*Q, N] - log prob for each query/class
            prototypes: [N, D] - class prototypes
        """
        # Compute embeddings
        n_support = support_images.size(0)
        n_query = query_images.size(0)
        
        # Embed all images at once for efficiency
        all_images = torch.cat([support_images, query_images], dim=0)
        all_embeddings = self.backbone(all_images)
        
        support_embeddings = all_embeddings[:n_support]  # [N*K, D]
        query_embeddings = all_embeddings[n_support:]     # [N*Q, D]
        
        # Compute prototypes: mean of support embeddings per class
        prototypes = self._compute_prototypes(support_embeddings, support_labels)
        
        # Compute distances from queries to prototypes
        distances = self._compute_distances(query_embeddings, prototypes)
        
        # Convert to log-probabilities (negative distance → softmax)
        log_probs = F.log_softmax(-distances, dim=1)
        
        return log_probs, prototypes
    
    def _compute_prototypes(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute class prototypes as mean embeddings.
        
        Args:
            embeddings: [N*K, D]
            labels: [N*K]
        
        Returns:
            prototypes: [N, D]
        """
        classes = torch.unique(labels)
        n_classes = len(classes)
        dim = embeddings.size(1)
        
        prototypes = torch.zeros(n_classes, dim, device=embeddings.device)
        
        for i, cls in enumerate(classes):
            mask = labels == cls
            prototypes[i] = embeddings[mask].mean(dim=0)
        
        return prototypes
    
    def _compute_distances(
        self,
        queries: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute distances between query embeddings and prototypes.
        
        Args:
            queries: [Q, D]
            prototypes: [N, D]
        
        Returns:
            distances: [Q, N]
        """
        if self.distance_metric == "euclidean":
            # Euclidean distance: ||q - p||^2
            # Expand for broadcasting: [Q, 1, D] - [1, N, D] → [Q, N, D]
            diff = queries.unsqueeze(1) - prototypes.unsqueeze(0)
            distances = (diff ** 2).sum(dim=2)
            return distances
        
        elif self.distance_metric == "cosine":
            # Cosine distance: 1 - cosine_similarity
            queries_norm = F.normalize(queries, p=2, dim=1)
            protos_norm = F.normalize(prototypes, p=2, dim=1)
            similarity = torch.mm(queries_norm, protos_norm.t())
            distances = 1.0 - similarity
            return distances
        
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
    
    def embed(self, images: torch.Tensor) -> torch.Tensor:
        """
        Embed images without episode structure.
        
        Args:
            images: [B, 3, H, W]
        
        Returns:
            embeddings: [B, D]
        """
        return self.backbone(images)
    
    def compute_prototypes_from_embeddings(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Public wrapper for prototype computation."""
        return self._compute_prototypes(embeddings, labels)
    
    def classify_against_prototypes(
        self,
        query_embeddings: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Classify query embeddings against precomputed prototypes.
        
        Args:
            query_embeddings: [Q, D]
            prototypes: [N, D]
        
        Returns:
            predictions: [Q] class indices
            confidences: [Q] probability of predicted class
        """
        distances = self._compute_distances(query_embeddings, prototypes)
        log_probs = F.log_softmax(-distances, dim=1)
        probs = torch.exp(log_probs)
        
        confidences, predictions = probs.max(dim=1)
        return predictions, confidences


class PrototypicalNetworkAlgorithm(MetaLearningAlgorithm):
    """
    Prototypical Networks algorithm wrapper.
    
    Implements the MetaLearningAlgorithm interface for ProtoNet.
    """
    
    def __init__(
        self,
        backbone_name: str = "resnet18",
        pretrained: bool = True,
        distance_metric: str = "euclidean",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = "cpu",
        freeze_backbone: bool = False,
    ):
        self.device = device
        self.model = PrototypicalNetwork(
            backbone_name=backbone_name,
            pretrained=pretrained,
            distance_metric=distance_metric,
            freeze_backbone=freeze_backbone,
        ).to(device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.scheduler = None
    
    def setup_scheduler(self, step_size: int = 20, gamma: float = 0.5):
        """Setup learning rate scheduler."""
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=step_size, gamma=gamma
        )
    
    def meta_train_step(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        query_labels: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Single meta-training episode step.
        
        Returns dict with 'loss' and 'accuracy'.
        """
        self.model.train()
        
        support_images = support_images.to(self.device)
        support_labels = support_labels.to(self.device)
        query_images = query_images.to(self.device)
        query_labels = query_labels.to(self.device)
        
        # Forward pass
        log_probs, prototypes = self.model(
            support_images, support_labels, query_images
        )
        
        # Loss: negative log-likelihood
        loss = F.nll_loss(log_probs, query_labels)
        
        # Accuracy
        predictions = log_probs.argmax(dim=1)
        accuracy = (predictions == query_labels).float().mean().item()
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy,
        }
    
    def meta_eval_step(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        query_labels: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Single meta-evaluation episode step (no gradient).
        """
        self.model.eval()
        
        support_images = support_images.to(self.device)
        support_labels = support_labels.to(self.device)
        query_images = query_images.to(self.device)
        query_labels = query_labels.to(self.device)
        
        with torch.no_grad():
            log_probs, prototypes = self.model(
                support_images, support_labels, query_images
            )
            loss = F.nll_loss(log_probs, query_labels)
            predictions = log_probs.argmax(dim=1)
            accuracy = (predictions == query_labels).float().mean().item()
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy,
            'predictions': predictions.cpu(),
            'labels': query_labels.cpu(),
            'log_probs': log_probs.cpu(),
        }
    
    def adapt(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Adapt to new classes by computing prototypes from support set.
        
        Returns dict with 'prototypes' and 'class_embeddings'.
        """
        self.model.eval()
        support_images = support_images.to(self.device)
        support_labels = support_labels.to(self.device)
        
        with torch.no_grad():
            embeddings = self.model.embed(support_images)
            prototypes = self.model.compute_prototypes_from_embeddings(
                embeddings, support_labels
            )
        
        return {
            'prototypes': prototypes,
            'embeddings': embeddings,
            'labels': support_labels,
        }
    
    def predict(
        self,
        query_images: torch.Tensor,
        adapted_state: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict classes for query images using adapted prototypes.
        
        Returns (predictions, confidences).
        """
        self.model.eval()
        query_images = query_images.to(self.device)
        prototypes = adapted_state['prototypes']
        
        with torch.no_grad():
            query_embeddings = self.model.embed(query_images)
            predictions, confidences = self.model.classify_against_prototypes(
                query_embeddings, prototypes
            )
        
        return predictions.cpu(), confidences.cpu()
    
    def get_model(self) -> nn.Module:
        """Get the underlying ProtoNet model."""
        return self.model
    
    def save_checkpoint(self, path: str, extra: dict = None):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        if extra:
            checkpoint.update(extra)
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint
