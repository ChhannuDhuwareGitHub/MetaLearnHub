"""
Meta-Learning Algorithm Base Class.

Defines the interface that all meta-learning algorithms must implement,
enabling pluggable algorithm swapping in the MetaLearnHub pipeline.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class MetaLearningAlgorithm(ABC):
    """
    Abstract base class for meta-learning algorithms.
    
    All algorithms must implement:
    - meta_train_step: A single meta-training episode step
    - adapt: Adapt to a new class from a support set
    - predict: Predict query images against adapted representation
    
    Future implementations:
    - Prototypical Networks ✓
    - Matching Networks (planned)
    - Relation Networks (planned)
    - First-Order MAML (planned)
    """
    
    @abstractmethod
    def meta_train_step(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        query_labels: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Perform a single meta-training episode step.
        
        Args:
            support_images: [N*K, C, H, W]
            support_labels: [N*K]
            query_images: [N*Q, C, H, W]
            query_labels: [N*Q]
        
        Returns:
            Dict with 'loss', 'accuracy', and any other metrics
        """
        pass
    
    @abstractmethod
    def adapt(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
    ) -> object:
        """
        Adapt to new class(es) from a support set.
        
        Args:
            support_images: [N*K, C, H, W]
            support_labels: [N*K]
        
        Returns:
            Adapted representation (algorithm-specific)
        """
        pass
    
    @abstractmethod
    def predict(
        self,
        query_images: torch.Tensor,
        adapted_state: object,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict class labels for query images.
        
        Args:
            query_images: [Q, C, H, W]
            adapted_state: Output from adapt()
        
        Returns:
            (predictions, confidences)
        """
        pass
    
    @abstractmethod
    def get_model(self) -> nn.Module:
        """Get the underlying model for checkpointing."""
        pass
