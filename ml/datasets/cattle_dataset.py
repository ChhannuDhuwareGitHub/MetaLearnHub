"""
PyTorch Dataset and Episodic Sampler for MetaLearnHub.

Provides torch Dataset classes and episodic batch samplers
for meta-learning training and evaluation.
"""

import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from PIL import Image

from ml.config import PreprocessConfig, BREED_NAMES
from ml.datasets.inspector import parse_filename


# ============================================================
# Image Transforms
# ============================================================

def get_train_transforms(config: PreprocessConfig) -> transforms.Compose:
    """Get training transforms with augmentation."""
    transform_list = [
        transforms.Resize((config.image_size, config.image_size)),
    ]
    
    if config.use_augmentation:
        if config.random_horizontal_flip:
            transform_list.append(transforms.RandomHorizontalFlip())
        if config.random_rotation > 0:
            transform_list.append(
                transforms.RandomRotation(config.random_rotation)
            )
        transform_list.append(
            transforms.ColorJitter(
                brightness=config.color_jitter_brightness,
                contrast=config.color_jitter_contrast,
                saturation=config.color_jitter_saturation,
                hue=config.color_jitter_hue,
            )
        )
    
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config.normalize_mean,
            std=config.normalize_std,
        ),
    ])
    
    return transforms.Compose(transform_list)


def get_eval_transforms(config: PreprocessConfig) -> transforms.Compose:
    """Get evaluation transforms (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config.normalize_mean,
            std=config.normalize_std,
        ),
    ])


# ============================================================
# MetaLearn Dataset
# ============================================================

class CattleDataset(Dataset):
    """
    PyTorch Dataset for cattle breed images.
    
    Stores images organized by class and sample group for leakage-safe episodic sampling.
    """
    
    def __init__(
        self,
        image_paths: List[Tuple[str, str]],  # (filepath, breed_name)
        transform: Optional[Callable] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
    ):
        """
        Args:
            image_paths: List of (filepath, breed_name) tuples
            transform: Image transform pipeline
            class_to_idx: Optional class-to-index mapping
        """
        self.image_paths = image_paths
        self.transform = transform
        
        # Build class-to-index mapping
        classes = sorted(set(breed for _, breed in image_paths))
        if class_to_idx is None:
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
        else:
            self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        # Build index of images per class and group
        self.class_images: Dict[str, List[int]] = defaultdict(list)
        self.class_groups: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
        
        for idx, (filepath, breed) in enumerate(image_paths):
            self.class_images[breed].append(idx)
            group_id, _ = parse_filename(Path(filepath).name)
            self.class_groups[breed][group_id].append(idx)
        
        self.classes = classes
        self.num_classes = len(classes)
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        filepath, breed = self.image_paths[idx]
        label = self.class_to_idx[breed]
        
        image = Image.open(filepath).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_images(self, class_name: str) -> List[int]:
        """Get indices of all images for a given class."""
        return self.class_images.get(class_name, [])

    def get_class_groups(self, class_name: str) -> Dict[str, List[int]]:
        """Get mapping of group_id -> list of image indices for a given class."""
        return self.class_groups.get(class_name, {})


# ============================================================
# Episodic Sampler
# ============================================================

class EpisodicSampler:
    """
    Samples N-way K-shot episodes for meta-learning.
    
    Each episode consists of:
    - N classes sampled from available classes
    - K support images per class
    - Q query images per class
    
    Guarantees group-level disjointness between support and query sets.
    """
    
    def __init__(
        self,
        dataset: CattleDataset,
        n_way: int,
        k_shot: int,
        query_per_class: int,
        num_episodes: int,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.query_per_class = query_per_class
        self.num_episodes = num_episodes
        self.rng = random.Random(seed)
        
        # Validate
        if n_way > dataset.num_classes:
            raise ValueError(
                f"n_way ({n_way}) > available classes ({dataset.num_classes}). "
                f"Available: {dataset.classes}"
            )
        
        # Check each class has enough groups/images
        for cls in dataset.classes:
            groups = dataset.get_class_groups(cls)
            n_groups = len(groups)
            if n_groups < 2:
                raise ValueError(
                    f"Class '{cls}' has only {n_groups} sample group(s), "
                    f"need at least 2 separate groups for support and query."
                )
    
    def __len__(self) -> int:
        return self.num_episodes
    
    def __iter__(self):
        for _ in range(self.num_episodes):
            yield self._sample_episode()
    
    def _sample_episode(self) -> dict:
        """Sample a single episode ensuring base/_c pairs never split across support/query."""
        episode_classes = self.rng.sample(self.dataset.classes, self.n_way)
        
        support_images = []
        support_labels = []
        query_images = []
        query_labels = []
        
        all_support_group_ids = []
        all_query_group_ids = []
        
        for local_label, cls_name in enumerate(episode_classes):
            groups_dict = self.dataset.get_class_groups(cls_name)
            group_ids = sorted(list(groups_dict.keys()))
            self.rng.shuffle(group_ids)
            
            # Select groups for support
            support_indices = []
            support_gids = []
            group_idx = 0
            
            while len(support_indices) < self.k_shot and group_idx < len(group_ids):
                gid = group_ids[group_idx]
                support_gids.append(gid)
                for idx in groups_dict[gid]:
                    support_indices.append(idx)
                    if len(support_indices) == self.k_shot:
                        break
                group_idx += 1
            
            # Select groups for query from remaining unused groups
            query_indices = []
            query_gids = []
            while len(query_indices) < self.query_per_class and group_idx < len(group_ids):
                gid = group_ids[group_idx]
                query_gids.append(gid)
                for idx in groups_dict[gid]:
                    query_indices.append(idx)
                    if len(query_indices) == self.query_per_class:
                        break
                group_idx += 1
            
            # Group leakage check
            sup_set = set(support_gids)
            qry_set = set(query_gids)
            assert sup_set.isdisjoint(qry_set), (
                f"Sample group leakage detected in episode for class {cls_name}! "
                f"Overlapping groups: {sup_set & qry_set}"
            )
            
            all_support_group_ids.extend(support_gids)
            all_query_group_ids.extend(query_gids)
            
            # Load images and record exact image-level group IDs
            for idx in support_indices:
                img, _ = self.dataset[idx]
                support_images.append(img)
                support_labels.append(local_label)
                filepath = self.dataset.image_paths[idx][0]
                gid, _ = parse_filename(Path(filepath).name)
                all_support_group_ids.append(gid)
            
            for idx in query_indices:
                img, _ = self.dataset[idx]
                query_images.append(img)
                query_labels.append(local_label)
                filepath = self.dataset.image_paths[idx][0]
                gid, _ = parse_filename(Path(filepath).name)
                all_query_group_ids.append(gid)
        
        # Strict group leakage check across the entire episode
        sup_set = set(all_support_group_ids)
        qry_set = set(all_query_group_ids)
        assert sup_set.isdisjoint(qry_set), (
            f"Sample group leakage detected in episode! "
            f"Overlapping groups: {sup_set & qry_set}"
        )
        
        return {
            'support_images': torch.stack(support_images),
            'support_labels': torch.tensor(support_labels, dtype=torch.long),
            'query_images': torch.stack(query_images),
            'query_labels': torch.tensor(query_labels, dtype=torch.long),
            'class_names': episode_classes,
            'support_group_ids': all_support_group_ids,
            'query_group_ids': all_query_group_ids,
        }


# ============================================================
# Few-Shot Evaluation Dataset
# ============================================================

class FewShotEvalDataset:
    """
    Creates few-shot evaluation episodes for the held-out class.
    
    Guarantees group-level disjointness between support and query sets.
    """
    
    def __init__(
        self,
        unseen_images: List[Tuple[str, str]],  # (path, breed) for held-out class
        known_images: List[Tuple[str, str]],    # (path, breed) for meta-train classes
        transform: Optional[Callable] = None,
        seed: int = 42,
    ):
        self.unseen_images = unseen_images
        self.known_images = known_images
        self.transform = transform
        self.rng = random.Random(seed)
        
        # Organize unseen images by group ID
        self.unseen_by_group: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        for path, breed in unseen_images:
            gid, _ = parse_filename(Path(path).name)
            self.unseen_by_group[gid].append((path, breed))
        
        # Organize known images by class and group ID
        self.known_by_class_group: Dict[str, Dict[str, List[Tuple[str, str]]]] = defaultdict(lambda: defaultdict(list))
        for path, breed in known_images:
            gid, _ = parse_filename(Path(path).name)
            self.known_by_class_group[breed][gid].append((path, breed))
    
    def sample_episode(
        self,
        k_shot: int,
        n_way: int = 2,
        query_per_class: int = 5,
        include_known_classes: bool = True,
    ) -> dict:
        """
        Sample a group-leakage-safe few-shot evaluation episode.
        """
        unseen_gids = sorted(list(self.unseen_by_group.keys()))
        self.rng.shuffle(unseen_gids)
        
        unseen_breed = self.unseen_images[0][1] if self.unseen_images else "Unseen Breed"
        
        # Group-aware support & query for unseen class
        support_unseen_paths = []
        support_unseen_gids = []
        g_idx = 0
        
        while len(support_unseen_paths) < k_shot and g_idx < len(unseen_gids):
            gid = unseen_gids[g_idx]
            support_unseen_gids.append(gid)
            for item in self.unseen_by_group[gid]:
                support_unseen_paths.append(item)
                if len(support_unseen_paths) == k_shot:
                    break
            g_idx += 1
        
        query_unseen_paths = []
        query_unseen_gids = []
        while len(query_unseen_paths) < query_per_class and g_idx < len(unseen_gids):
            gid = unseen_gids[g_idx]
            query_unseen_gids.append(gid)
            for item in self.unseen_by_group[gid]:
                query_unseen_paths.append(item)
                if len(query_unseen_paths) == query_per_class:
                    break
            g_idx += 1
        
        assert set(support_unseen_gids).isdisjoint(set(query_unseen_gids)), (
            f"Unseen class group leakage detected! "
            f"Support groups: {support_unseen_gids}, Query groups: {query_unseen_gids}"
        )
        
        support_images = []
        support_labels = []
        query_images = []
        query_labels = []
        class_names = [unseen_breed]  # Unseen class = label 0
        
        all_support_gids = []
        all_query_gids = []
        
        # Load unseen support
        for path, _ in support_unseen_paths:
            img = self._load_image(path)
            support_images.append(img)
            support_labels.append(0)
            gid, _ = parse_filename(Path(path).name)
            all_support_gids.append(gid)
        
        # Load unseen query
        for path, _ in query_unseen_paths:
            img = self._load_image(path)
            query_images.append(img)
            query_labels.append(0)
            gid, _ = parse_filename(Path(path).name)
            all_query_gids.append(gid)
        
        # Add known classes
        if include_known_classes and n_way > 1:
            available_known = sorted(list(self.known_by_class_group.keys()))
            n_known = min(n_way - 1, len(available_known))
            selected_known = self.rng.sample(available_known, n_known)
            
            for local_label, cls_name in enumerate(selected_known, start=1):
                class_names.append(cls_name)
                cls_groups = self.known_by_class_group[cls_name]
                cls_gids = sorted(list(cls_groups.keys()))
                self.rng.shuffle(cls_gids)
                
                k_sup_paths = []
                kg_idx = 0
                while len(k_sup_paths) < k_shot and kg_idx < len(cls_gids):
                    gid = cls_gids[kg_idx]
                    for item in cls_groups[gid]:
                        k_sup_paths.append(item)
                        if len(k_sup_paths) == k_shot:
                            break
                    kg_idx += 1
                
                k_qry_paths = []
                while len(k_qry_paths) < query_per_class and kg_idx < len(cls_gids):
                    gid = cls_gids[kg_idx]
                    for item in cls_groups[gid]:
                        k_qry_paths.append(item)
                        if len(k_qry_paths) == query_per_class:
                            break
                    kg_idx += 1
                
                for path, _ in k_sup_paths:
                    img = self._load_image(path)
                    support_images.append(img)
                    support_labels.append(local_label)
                    gid, _ = parse_filename(Path(path).name)
                    all_support_gids.append(gid)
                
                for path, _ in k_qry_paths:
                    img = self._load_image(path)
                    query_images.append(img)
                    query_labels.append(local_label)
                    gid, _ = parse_filename(Path(path).name)
                    all_query_gids.append(gid)
        
        # Strict full episode group disjointness check
        sup_set = set(all_support_gids)
        qry_set = set(all_query_gids)
        assert sup_set.isdisjoint(qry_set), (
            f"Eval group leakage detected! Intersection: {sup_set & qry_set}"
        )
        
        return {
            'support_images': torch.stack(support_images),
            'support_labels': torch.tensor(support_labels, dtype=torch.long),
            'query_images': torch.stack(query_images),
            'query_labels': torch.tensor(query_labels, dtype=torch.long),
            'class_names': class_names,
            'unseen_class': unseen_breed,
            'k_shot': k_shot,
            'n_way': len(class_names),
            'support_group_ids': all_support_gids,
            'query_group_ids': all_query_gids,
        }
    
    def _load_image(self, path: str) -> torch.Tensor:
        """Load and transform a single image."""
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

