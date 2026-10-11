"""
MetaLearnHub Configuration Module.

Centralizes all configurable parameters for the meta-learning pipeline.
"""

import os
import json
import hashlib
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Literal
from pathlib import Path


# ============================================================
# Path Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "cattle" / "Cattle Breeds"
PROCESSED_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

# Ensure directories exist
for d in [PROCESSED_DIR, METADATA_DIR, EXPERIMENTS_DIR, CHECKPOINTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ============================================================
# Breed Configuration
# ============================================================

BREED_NAMES = [
    "Ayrshire cattle",
    "Brown Swiss cattle",
    "Holstein Friesian cattle",
    "Jersey cattle",
    "Red Dane cattle",
]

BREED_SHORT_NAMES = {
    "Ayrshire cattle": "ayrshire",
    "Brown Swiss cattle": "brown_swiss",
    "Holstein Friesian cattle": "holstein_friesian",
    "Jersey cattle": "jersey",
    "Red Dane cattle": "red_dane",
}


# ============================================================
# Image Preprocessing Configuration
# ============================================================

@dataclass
class PreprocessConfig:
    """Image preprocessing configuration for ResNet-18 backbone."""
    image_size: int = 224
    center_crop_size: int = 224
    # ImageNet normalization
    normalize_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalize_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    # Training augmentation
    use_augmentation: bool = True
    random_horizontal_flip: bool = True
    random_rotation: float = 15.0
    color_jitter_brightness: float = 0.2
    color_jitter_contrast: float = 0.2
    color_jitter_saturation: float = 0.2
    color_jitter_hue: float = 0.1


# ============================================================
# Episode Configuration
# ============================================================

@dataclass
class EpisodeConfig:
    """Configuration for episodic meta-training."""
    n_way: int = 3  # Number of classes per episode
    k_shot: int = 5  # Support images per class
    query_per_class: int = 5  # Query images per class
    episodes_per_epoch: int = 100
    num_epochs: int = 50
    
    def validate(self, num_available_classes: int):
        """Validate episode config against available classes."""
        if self.n_way > num_available_classes:
            raise ValueError(
                f"n_way ({self.n_way}) exceeds available classes "
                f"({num_available_classes}). Max n_way = {num_available_classes}"
            )
        if self.n_way < 2:
            raise ValueError(f"n_way must be >= 2, got {self.n_way}")
        if self.k_shot < 1:
            raise ValueError(f"k_shot must be >= 1, got {self.k_shot}")


# ============================================================
# Model Configuration
# ============================================================

@dataclass
class ModelConfig:
    """Configuration for the meta-learning model."""
    backbone: str = "resnet18"
    pretrained: bool = True
    embedding_dim: int = 512  # ResNet-18 output dim
    distance_metric: Literal["euclidean", "cosine"] = "euclidean"
    freeze_backbone: bool = False


# ============================================================
# Training Configuration
# ============================================================

@dataclass
class TrainingConfig:
    """Configuration for the training process."""
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    lr_scheduler: str = "step"
    lr_step_size: int = 20
    lr_gamma: float = 0.5
    optimizer: str = "adam"
    device: Literal["auto", "cpu", "cuda"] = "auto"
    num_workers: int = 4
    pin_memory: bool = True


# ============================================================
# Experiment Configuration
# ============================================================

@dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    # Experiment identity
    experiment_name: str = "protonet_experiment"
    random_seed: int = 42
    
    # Leave-One-Class-Out
    held_out_class: str = "Red Dane cattle"
    
    # Algorithm
    algorithm: str = "prototypical_networks"
    
    # Sub-configurations
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Evaluation shots
    eval_shots: List[int] = field(default_factory=lambda: [1, 3, 5, 10])
    eval_episodes: int = 50  # Number of evaluation episodes per shot
    
    @property
    def meta_train_classes(self) -> List[str]:
        """Classes used for meta-training (all except held-out)."""
        return [b for b in BREED_NAMES if b != self.held_out_class]
    
    def validate(self):
        """Validate the full configuration."""
        if self.held_out_class not in BREED_NAMES:
            raise ValueError(
                f"held_out_class '{self.held_out_class}' not in {BREED_NAMES}"
            )
        num_train = len(self.meta_train_classes)
        self.episode.validate(num_train)
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return asdict(self)
    
    def save(self, path: Path):
        """Save configuration to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'ExperimentConfig':
        """Load configuration from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        config = cls()
        # Recursively update fields
        config.preprocess = PreprocessConfig(**data.get('preprocess', {}))
        config.episode = EpisodeConfig(**data.get('episode', {}))
        config.model = ModelConfig(**data.get('model', {}))
        config.training = TrainingConfig(**data.get('training', {}))
        for key in ['experiment_name', 'random_seed', 'held_out_class', 
                     'algorithm', 'eval_shots', 'eval_episodes']:
            if key in data:
                setattr(config, key, data[key])
        return config
    
    def get_hash(self) -> str:
        """Get a hash of the configuration for identification."""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:12]
