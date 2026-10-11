"""
Leakage-Safe Dataset Splitting for MetaLearnHub.

Implements group-aware splitting that ensures base/_c image pairs
are always assigned to the same partition, preventing data leakage.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field, asdict
from collections import defaultdict

import numpy as np

from ml.config import (
    METADATA_DIR, BREED_NAMES, ExperimentConfig
)
from ml.datasets.inspector import DatasetInspector, SampleGroup, ImageRecord


# ============================================================
# Split Assignment
# ============================================================

@dataclass
class SplitAssignment:
    """Assignment of sample groups to train/val/test splits."""
    breed: str
    role: str  # 'meta_train' or 'held_out'
    train_groups: List[str] = field(default_factory=list)  # group_ids
    val_groups: List[str] = field(default_factory=list)
    test_groups: List[str] = field(default_factory=list)


@dataclass
class DataSplit:
    """Complete data split specification."""
    held_out_class: str
    meta_train_classes: List[str] = field(default_factory=list)
    assignments: Dict[str, dict] = field(default_factory=dict)
    random_seed: int = 42
    total_train_images: int = 0
    total_val_images: int = 0
    total_test_images: int = 0
    total_unseen_images: int = 0
    
    def validate_no_leakage(self, inspector: DatasetInspector) -> List[str]:
        """
        Validate that no data leakage exists in this split.
        
        Returns list of violations (empty if clean).
        """
        violations = []
        
        # 1. Check held-out class images are NOT in training
        held_out_groups = set()
        for key, group in inspector.sample_groups.items():
            if group.breed == self.held_out_class:
                held_out_groups.add(key)
        
        if self.held_out_class in self.assignments:
            assignment = self.assignments[self.held_out_class]
            if assignment.get('train_groups') or assignment.get('val_groups'):
                violations.append(
                    f"CRITICAL: Held-out class '{self.held_out_class}' has images "
                    f"in train or validation split!"
                )
        
        # 2. Check that base/_c pairs stay together for meta-training classes
        for breed in self.meta_train_classes:
            if breed not in self.assignments:
                continue
            a = self.assignments[breed]
            train_set = set(a.get('train_groups', []))
            val_set = set(a.get('val_groups', []))
            test_set = set(a.get('test_groups', []))
            
            # Check no group appears in multiple splits
            for gid in train_set:
                if gid in val_set:
                    violations.append(
                        f"Group '{gid}' from '{breed}' appears in BOTH train and val"
                    )
                if gid in test_set:
                    violations.append(
                        f"Group '{gid}' from '{breed}' appears in BOTH train and test"
                    )
            for gid in val_set:
                if gid in test_set:
                    violations.append(
                        f"Group '{gid}' from '{breed}' appears in BOTH val and test"
                    )
        
        return violations


class DatasetSplitter:
    """
    Creates leakage-safe data splits for meta-learning experiments.
    
    Key guarantees:
    1. Base and _c images always stay in the same split
    2. Held-out class images never appear in meta-training
    3. Splits are reproducible via random seed
    """
    
    def __init__(self, inspector: DatasetInspector):
        self.inspector = inspector
    
    def create_split(
        self,
        held_out_class: str,
        seed: int = 42,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        # test_ratio is implicit = 1 - train - val
    ) -> DataSplit:
        """
        Create a leakage-safe data split.
        
        For meta-training classes: split into train/val/test at group level.
        For held-out class: all groups are reserved for few-shot evaluation.
        
        Args:
            held_out_class: Class to hold out for few-shot evaluation
            seed: Random seed for reproducibility
            train_ratio: Fraction of groups for training
            val_ratio: Fraction of groups for validation
        
        Returns:
            DataSplit specification
        """
        if held_out_class not in BREED_NAMES:
            raise ValueError(f"Unknown breed: {held_out_class}")
        
        rng = random.Random(seed)
        meta_train_classes = [b for b in BREED_NAMES if b != held_out_class]
        
        split = DataSplit(
            held_out_class=held_out_class,
            meta_train_classes=meta_train_classes,
            random_seed=seed,
        )
        
        # Split meta-training classes into train/val/test at GROUP level
        for breed in meta_train_classes:
            groups = self.inspector.get_sample_groups_by_breed(breed)
            group_ids = sorted([g.group_id for g in groups])
            
            # Shuffle with seed
            shuffled = list(group_ids)
            rng.shuffle(shuffled)
            
            n_total = len(shuffled)
            n_train = max(1, int(n_total * train_ratio))
            n_val = max(1, int(n_total * val_ratio))
            
            train_ids = shuffled[:n_train]
            val_ids = shuffled[n_train:n_train + n_val]
            test_ids = shuffled[n_train + n_val:]
            
            split.assignments[breed] = {
                'role': 'meta_train',
                'train_groups': train_ids,
                'val_groups': val_ids,
                'test_groups': test_ids,
                'total_groups': n_total,
            }
            
            # Count images
            group_lookup = {g.group_id: g for g in groups}
            for gid in train_ids:
                if gid in group_lookup:
                    split.total_train_images += len(group_lookup[gid].images)
            for gid in val_ids:
                if gid in group_lookup:
                    split.total_val_images += len(group_lookup[gid].images)
            for gid in test_ids:
                if gid in group_lookup:
                    split.total_test_images += len(group_lookup[gid].images)
        
        # Held-out class: all reserved for unseen evaluation
        held_out_groups = self.inspector.get_sample_groups_by_breed(held_out_class)
        held_out_ids = sorted([g.group_id for g in held_out_groups])
        
        split.assignments[held_out_class] = {
            'role': 'held_out',
            'train_groups': [],  # NONE in training
            'val_groups': [],    # NONE in validation
            'test_groups': held_out_ids,  # All for evaluation
            'total_groups': len(held_out_ids),
        }
        
        for g in held_out_groups:
            split.total_unseen_images += len(g.images)
        
        # Validate no leakage
        violations = split.validate_no_leakage(self.inspector)
        if violations:
            raise RuntimeError(
                f"DATA LEAKAGE DETECTED!\n" +
                "\n".join(violations)
            )
        
        return split
    
    def get_image_paths_for_split(
        self,
        split: DataSplit,
        partition: str,  # 'train', 'val', 'test', 'unseen'
        breed: str = None,
    ) -> List[Tuple[str, str]]:
        """
        Get (image_path, breed_name) tuples for a specific partition.
        
        Args:
            split: The data split specification
            partition: One of 'train', 'val', 'test', 'unseen'
            breed: Optional specific breed to filter
        
        Returns:
            List of (filepath, breed_name) tuples
        """
        results = []
        
        if partition == 'unseen':
            # Get all images from held-out class
            breed_name = split.held_out_class
            groups = self.inspector.get_sample_groups_by_breed(breed_name)
            for group in groups:
                for img in group.images:
                    if img.is_valid:
                        results.append((img.filepath, breed_name))
        else:
            # Get images from meta-training classes
            target_breeds = [breed] if breed else split.meta_train_classes
            
            for breed_name in target_breeds:
                if breed_name not in split.assignments:
                    continue
                assignment = split.assignments[breed_name]
                
                group_key = f"{partition}_groups"
                group_ids = set(assignment.get(group_key, []))
                
                groups = self.inspector.get_sample_groups_by_breed(breed_name)
                for group in groups:
                    if group.group_id in group_ids:
                        for img in group.images:
                            if img.is_valid:
                                results.append((img.filepath, breed_name))
        
        return results
    
    def save_split(self, split: DataSplit, output_dir: Path = None) -> Path:
        """Save split to JSON."""
        output_dir = output_dir or METADATA_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        
        split_data = asdict(split)
        path = output_dir / f"split_{split.held_out_class.replace(' ', '_')}_seed{split.random_seed}.json"
        with open(path, 'w') as f:
            json.dump(split_data, f, indent=2)
        
        return path


def create_leakage_safe_split(
    inspector: DatasetInspector,
    held_out_class: str,
    seed: int = 42,
) -> DataSplit:
    """
    Convenience function to create a leakage-safe split.
    """
    splitter = DatasetSplitter(inspector)
    split = splitter.create_split(held_out_class=held_out_class, seed=seed)
    splitter.save_split(split)
    return split
