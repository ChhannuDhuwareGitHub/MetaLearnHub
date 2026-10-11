"""
Few-Shot Inference / User Adaptation Module for MetaLearnHub.

Handles the real application workflow:
1. User creates a new temporary class
2. User uploads K labeled support images
3. System computes prototype for the new class
4. User uploads query images
5. System classifies them against the new class

This module does NOT modify the underlying meta-training model.
"""

import time
import uuid
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from io import BytesIO

import torch
import numpy as np
from PIL import Image

from ml.config import PreprocessConfig
from ml.datasets.cattle_dataset import get_eval_transforms
from ml.algorithms.prototypical_networks.protonet import PrototypicalNetworkAlgorithm

logger = logging.getLogger(__name__)


class TemporaryClass:
    """
    Represents a user-created temporary class with a computed prototype.
    
    This class is ephemeral — it does not permanently modify the meta-training model.
    """
    
    def __init__(self, class_name: str, class_id: str = None):
        self.class_name = class_name
        self.class_id = class_id or str(uuid.uuid4())[:8]
        self.support_image_paths: List[str] = []
        self.support_embeddings: Optional[torch.Tensor] = None
        self.prototype: Optional[torch.Tensor] = None
        self.created_at: str = ""
    
    @property
    def k_shot(self) -> int:
        return len(self.support_image_paths)
    
    @property
    def is_ready(self) -> bool:
        return self.prototype is not None
    
    def to_dict(self) -> dict:
        return {
            'class_id': self.class_id,
            'class_name': self.class_name,
            'k_shot': self.k_shot,
            'is_ready': self.is_ready,
            'support_images': self.support_image_paths,
            'created_at': self.created_at,
        }


class FewShotPredictor:
    """
    Manages user-created temporary classes and few-shot predictions.
    
    Workflow:
        1. create_class(name) → TemporaryClass
        2. add_support_images(class_id, images)
        3. build_prototype(class_id)
        4. predict(query_image, class_ids)
    """
    
    def __init__(
        self,
        algorithm: PrototypicalNetworkAlgorithm,
        preprocess_config: PreprocessConfig = None,
    ):
        self.algorithm = algorithm
        self.transform = get_eval_transforms(preprocess_config or PreprocessConfig())
        self.temporary_classes: Dict[str, TemporaryClass] = {}
    
    def create_class(self, class_name: str) -> TemporaryClass:
        """Create a new temporary class."""
        temp_class = TemporaryClass(class_name=class_name)
        from datetime import datetime
        temp_class.created_at = datetime.now().isoformat()
        self.temporary_classes[temp_class.class_id] = temp_class
        logger.info(f"Created temporary class: {class_name} (id={temp_class.class_id})")
        return temp_class
    
    def add_support_image(
        self,
        class_id: str,
        image_path: str = None,
        image_bytes: bytes = None,
    ) -> TemporaryClass:
        """
        Add a support image to a temporary class.
        
        Accepts either a file path or raw bytes.
        """
        if class_id not in self.temporary_classes:
            raise ValueError(f"Unknown class_id: {class_id}")
        
        temp_class = self.temporary_classes[class_id]
        
        if image_path:
            temp_class.support_image_paths.append(image_path)
        elif image_bytes:
            # Save to temp location
            temp_dir = Path("data/processed/temp_uploads")
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_path = temp_dir / f"{class_id}_{temp_class.k_shot}.jpg"
            with open(temp_path, 'wb') as f:
                f.write(image_bytes)
            temp_class.support_image_paths.append(str(temp_path))
        
        # Invalidate prototype (needs rebuild)
        temp_class.prototype = None
        temp_class.support_embeddings = None
        
        return temp_class
    
    def build_prototype(self, class_id: str) -> TemporaryClass:
        """
        Compute the prototype for a temporary class from its support images.
        
        The prototype is the mean embedding of all support images.
        """
        if class_id not in self.temporary_classes:
            raise ValueError(f"Unknown class_id: {class_id}")
        
        temp_class = self.temporary_classes[class_id]
        
        if temp_class.k_shot == 0:
            raise ValueError(f"Class '{temp_class.class_name}' has no support images.")
        
        # Load and embed all support images
        embeddings = []
        self.algorithm.get_model().eval()
        
        with torch.no_grad():
            for img_path in temp_class.support_image_paths:
                img = Image.open(img_path).convert('RGB')
                img_tensor = self.transform(img).unsqueeze(0)
                embedding = self.algorithm.get_model().embed(
                    img_tensor.to(next(self.algorithm.get_model().parameters()).device)
                )
                embeddings.append(embedding)
        
        # Compute prototype (mean embedding)
        all_embeddings = torch.cat(embeddings, dim=0)
        prototype = all_embeddings.mean(dim=0, keepdim=True)
        
        temp_class.support_embeddings = all_embeddings
        temp_class.prototype = prototype
        
        logger.info(
            f"Built prototype for '{temp_class.class_name}' from "
            f"{temp_class.k_shot} support images. "
            f"Embedding dim: {prototype.shape[1]}"
        )
        
        return temp_class
    
    def predict(
        self,
        query_image_path: str = None,
        query_image_bytes: bytes = None,
        class_ids: List[str] = None,
    ) -> dict:
        """
        Classify a query image against temporary classes.
        
        Args:
            query_image_path: Path to query image
            query_image_bytes: Raw bytes of query image
            class_ids: Optional subset of class IDs to compare against
                       (defaults to all temporary classes)
        
        Returns:
            Prediction result dictionary
        """
        # Load query image
        if query_image_path:
            query_img = Image.open(query_image_path).convert('RGB')
        elif query_image_bytes:
            query_img = Image.open(BytesIO(query_image_bytes)).convert('RGB')
        else:
            raise ValueError("Must provide either query_image_path or query_image_bytes")
        
        # Select classes
        if class_ids is None:
            class_ids = list(self.temporary_classes.keys())
        
        # Check all classes are ready
        classes = []
        prototypes = []
        for cid in class_ids:
            if cid not in self.temporary_classes:
                raise ValueError(f"Unknown class_id: {cid}")
            tc = self.temporary_classes[cid]
            if not tc.is_ready:
                raise ValueError(
                    f"Class '{tc.class_name}' prototype not built. "
                    f"Call build_prototype() first."
                )
            classes.append(tc)
            prototypes.append(tc.prototype)
        
        # Stack prototypes
        all_prototypes = torch.cat(prototypes, dim=0)
        
        # Embed query
        self.algorithm.get_model().eval()
        device = next(self.algorithm.get_model().parameters()).device
        
        query_tensor = self.transform(query_img).unsqueeze(0).to(device)
        
        t_start = time.time()
        with torch.no_grad():
            query_embedding = self.algorithm.get_model().embed(query_tensor)
            predictions, confidences = self.algorithm.get_model().classify_against_prototypes(
                query_embedding, all_prototypes
            )
        inference_time = time.time() - t_start
        
        pred_idx = predictions[0].item()
        confidence = confidences[0].item()
        
        # Build result
        result = {
            'predicted_class': classes[pred_idx].class_name,
            'predicted_class_id': classes[pred_idx].class_id,
            'confidence': round(float(confidence), 4),
            'inference_time_ms': round(inference_time * 1000, 2),
            'all_classes': [],
        }
        
        # Compute distances to all prototypes for comparison
        with torch.no_grad():
            distances = self.algorithm.get_model()._compute_distances(
                query_embedding, all_prototypes
            )
            probs = torch.softmax(-distances, dim=1)[0]
        
        for i, tc in enumerate(classes):
            result['all_classes'].append({
                'class_name': tc.class_name,
                'class_id': tc.class_id,
                'probability': round(float(probs[i].item()), 4),
                'distance': round(float(distances[0, i].item()), 4),
                'k_shot': tc.k_shot,
            })
        
        return result
    
    def list_classes(self) -> List[dict]:
        """List all temporary classes."""
        return [tc.to_dict() for tc in self.temporary_classes.values()]
    
    def remove_class(self, class_id: str):
        """Remove a temporary class."""
        if class_id in self.temporary_classes:
            del self.temporary_classes[class_id]
