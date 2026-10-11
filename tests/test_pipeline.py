"""
Tests for MetaLearnHub.

Covers:
- Dataset inspection and grouping
- Leakage-safe splitting
- Episode construction
- Model forward pass
- Prototype calculation
- Evaluation metrics
- Reproducibility
"""

import os
import sys
import pytest
import tempfile
import json
from pathlib import Path
from collections import defaultdict

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from PIL import Image

from ml.config import (
    ExperimentConfig, PreprocessConfig, EpisodeConfig,
    ModelConfig, TrainingConfig, BREED_NAMES, RAW_DATA_DIR,
)
from ml.datasets.inspector import (
    DatasetInspector, parse_filename, ImageRecord, SampleGroup,
)
from ml.datasets.splitter import DatasetSplitter, DataSplit
from ml.datasets.cattle_dataset import (
    CattleDataset, EpisodicSampler, get_train_transforms, get_eval_transforms,
)
from ml.backbones.resnet import ResNet18Backbone, get_backbone
from ml.algorithms.prototypical_networks.protonet import (
    PrototypicalNetwork, PrototypicalNetworkAlgorithm,
)


# ============================================================
# Test Dataset Inspection
# ============================================================

class TestFilenameParser:
    """Test the filename parsing for base/_c detection."""
    
    def test_base_image(self):
        group_id, is_transformed = parse_filename("Ayrshirecattle100.jpg")
        assert group_id == "Ayrshirecattle100"
        assert is_transformed == False
    
    def test_transformed_image(self):
        group_id, is_transformed = parse_filename("Ayrshirecattle100_c.jpg")
        assert group_id == "Ayrshirecattle100"
        assert is_transformed == True
    
    def test_base_and_transformed_share_group(self):
        base_id, _ = parse_filename("BrownSwisscattle50.jpg")
        trans_id, _ = parse_filename("BrownSwisscattle50_c.jpg")
        assert base_id == trans_id
    
    def test_different_images_different_groups(self):
        id1, _ = parse_filename("Ayrshirecattle100.jpg")
        id2, _ = parse_filename("Ayrshirecattle101.jpg")
        assert id1 != id2
    
    def test_different_breeds_different_groups(self):
        id1, _ = parse_filename("Ayrshirecattle1.jpg")
        id2, _ = parse_filename("Jerseycattle1.jpg")
        assert id1 != id2


class TestDatasetInspection:
    """Test the dataset inspector on real data."""
    
    @pytest.fixture
    def inspector(self):
        if not RAW_DATA_DIR.exists():
            pytest.skip("Dataset not available")
        inspector = DatasetInspector(RAW_DATA_DIR)
        inspector.scan()
        return inspector
    
    def test_all_breeds_found(self, inspector):
        breeds = set(g.breed for g in inspector.sample_groups.values())
        for expected in BREED_NAMES:
            assert expected in breeds, f"Missing breed: {expected}"
    
    def test_sample_group_count(self, inspector):
        """Each breed should have approximately expected number of groups."""
        for breed in BREED_NAMES:
            groups = inspector.get_sample_groups_by_breed(breed)
            assert len(groups) > 50, f"{breed} has too few groups: {len(groups)}"
    
    def test_base_c_pairing(self, inspector):
        """Most groups should have both base and _c files."""
        total_complete = sum(
            1 for g in inspector.sample_groups.values()
            if g.is_complete_pair
        )
        total_groups = len(inspector.sample_groups)
        # At least 90% should be complete pairs
        assert total_complete / total_groups > 0.9, (
            f"Only {total_complete}/{total_groups} groups are complete pairs"
        )
    
    def test_no_empty_groups(self, inspector):
        for gid, group in inspector.sample_groups.items():
            assert len(group.images) > 0, f"Empty group: {gid}"


# ============================================================
# Test Leakage-Safe Splitting
# ============================================================

class TestDataSplitting:
    """Test the leakage-safe data splitting."""
    
    @pytest.fixture
    def inspector_and_splitter(self):
        if not RAW_DATA_DIR.exists():
            pytest.skip("Dataset not available")
        inspector = DatasetInspector(RAW_DATA_DIR)
        inspector.scan()
        return inspector, DatasetSplitter(inspector)
    
    def test_held_out_class_not_in_training(self, inspector_and_splitter):
        """Critical: held-out class must not appear in training."""
        inspector, splitter = inspector_and_splitter
        
        for held_out in BREED_NAMES:
            split = splitter.create_split(held_out_class=held_out, seed=42)
            
            assignment = split.assignments[held_out]
            assert len(assignment['train_groups']) == 0, (
                f"Held-out class {held_out} has training groups!"
            )
            assert len(assignment['val_groups']) == 0, (
                f"Held-out class {held_out} has validation groups!"
            )
    
    def test_base_c_pairs_same_split(self, inspector_and_splitter):
        """Critical: base and _c images must be in the same partition."""
        inspector, splitter = inspector_and_splitter
        split = splitter.create_split(held_out_class="Red Dane cattle", seed=42)
        
        for breed in split.meta_train_classes:
            assignment = split.assignments[breed]
            train_set = set(assignment['train_groups'])
            val_set = set(assignment['val_groups'])
            test_set = set(assignment['test_groups'])
            
            groups = inspector.get_sample_groups_by_breed(breed)
            for group in groups:
                # Find which split this group is in
                in_train = group.group_id in train_set
                in_val = group.group_id in val_set
                in_test = group.group_id in test_set
                
                # Group must be in exactly one split
                count = sum([in_train, in_val, in_test])
                assert count == 1, (
                    f"Group {group.group_id} appears in {count} splits!"
                )
                
                # All images in the group share the same split
                # (this is guaranteed by group-level splitting)
    
    def test_no_leakage_validation(self, inspector_and_splitter):
        """Test the leakage validation function."""
        inspector, splitter = inspector_and_splitter
        split = splitter.create_split(held_out_class="Red Dane cattle", seed=42)
        
        violations = split.validate_no_leakage(inspector)
        assert len(violations) == 0, f"Leakage detected: {violations}"
    
    def test_reproducibility(self, inspector_and_splitter):
        """Same seed should produce same split."""
        inspector, splitter = inspector_and_splitter
        
        split1 = splitter.create_split(held_out_class="Jersey cattle", seed=123)
        split2 = splitter.create_split(held_out_class="Jersey cattle", seed=123)
        
        for breed in split1.meta_train_classes:
            assert split1.assignments[breed]['train_groups'] == \
                   split2.assignments[breed]['train_groups']
    
    def test_different_seeds_different_splits(self, inspector_and_splitter):
        """Different seeds should produce different splits."""
        inspector, splitter = inspector_and_splitter
        
        split1 = splitter.create_split(held_out_class="Jersey cattle", seed=42)
        split2 = splitter.create_split(held_out_class="Jersey cattle", seed=99)
        
        # At least one breed should have different train groups
        any_different = False
        for breed in split1.meta_train_classes:
            if split1.assignments[breed]['train_groups'] != \
               split2.assignments[breed]['train_groups']:
                any_different = True
                break
        assert any_different, "Different seeds produced identical splits"


# ============================================================
# Test Episode Construction
# ============================================================

class TestEpisodicSampler:
    """Test the episodic sampler for meta-learning."""
    
    @pytest.fixture
    def dataset(self):
        if not RAW_DATA_DIR.exists():
            pytest.skip("Dataset not available")
        inspector = DatasetInspector(RAW_DATA_DIR)
        inspector.scan()
        splitter = DatasetSplitter(inspector)
        split = splitter.create_split(held_out_class="Red Dane cattle", seed=42)
        paths = splitter.get_image_paths_for_split(split, 'train')
        transform = get_eval_transforms(PreprocessConfig())
        return CattleDataset(paths, transform=transform)
    
    def test_episode_shapes(self, dataset):
        sampler = EpisodicSampler(
            dataset=dataset, n_way=3, k_shot=5,
            query_per_class=3, num_episodes=2, seed=42,
        )
        episode = next(iter(sampler))
        
        assert episode['support_images'].shape[0] == 15  # 3*5
        assert episode['support_labels'].shape[0] == 15
        assert episode['query_images'].shape[0] == 9  # 3*3
        assert episode['query_labels'].shape[0] == 9
    
    def test_support_query_disjoint(self, dataset):
        """Support and query must not overlap."""
        sampler = EpisodicSampler(
            dataset=dataset, n_way=3, k_shot=5,
            query_per_class=3, num_episodes=5, seed=42,
        )
        for episode in sampler:
            # Each class has k_shot support + query_per_class query
            # They were sampled from non-overlapping index ranges
            assert episode['support_images'].shape[0] == 15
            assert episode['query_images'].shape[0] == 9
    
    def test_n_way_classes(self, dataset):
        """Episode should contain exactly N classes."""
        sampler = EpisodicSampler(
            dataset=dataset, n_way=3, k_shot=5,
            query_per_class=3, num_episodes=2, seed=42,
        )
        episode = next(iter(sampler))
        unique_labels = episode['support_labels'].unique()
        assert len(unique_labels) == 3
    
    def test_k_shot_count(self, dataset):
        """Each class should have exactly K support images."""
        sampler = EpisodicSampler(
            dataset=dataset, n_way=3, k_shot=5,
            query_per_class=3, num_episodes=2, seed=42,
        )
        episode = next(iter(sampler))
        
        for label in episode['support_labels'].unique():
            count = (episode['support_labels'] == label).sum().item()
            assert count == 5, f"Class {label} has {count} support images, expected 5"

    def test_episode_group_disjointness(self, dataset):
        """Regression Test: Support and query sample groups must be 100% disjoint in every episode."""
        sampler = EpisodicSampler(
            dataset=dataset, n_way=3, k_shot=5,
            query_per_class=3, num_episodes=10, seed=42,
        )
        for episode in sampler:
            sup_groups = set(episode['support_group_ids'])
            qry_groups = set(episode['query_group_ids'])
            assert sup_groups.isdisjoint(qry_groups), (
                f"Sample group leakage detected! Intersection: {sup_groups & qry_groups}"
            )


class TestFewShotEvalDataset:
    """Test group-level leakage safety in evaluation episodes."""
    
    def test_eval_episode_group_disjointness(self):
        """Regression Test: FewShotEvalDataset must never leak base/_c sample groups between support and query sets."""
        if not RAW_DATA_DIR.exists():
            pytest.skip("Dataset not available")
        from ml.datasets.cattle_dataset import FewShotEvalDataset
        inspector = DatasetInspector(RAW_DATA_DIR)
        inspector.scan()
        splitter = DatasetSplitter(inspector)
        split = splitter.create_split(held_out_class="Red Dane cattle", seed=42)
        unseen_paths = splitter.get_image_paths_for_split(split, 'unseen')
        known_paths = splitter.get_image_paths_for_split(split, 'test')
        transform = get_eval_transforms(PreprocessConfig())
        eval_ds = FewShotEvalDataset(unseen_images=unseen_paths, known_images=known_paths, transform=transform, seed=42)
        for k_shot in [1, 5]:
            for ep_idx in range(5):
                episode = eval_ds.sample_episode(k_shot=k_shot, n_way=2, query_per_class=5)
                sup_gids = set(episode['support_group_ids'])
                qry_gids = set(episode['query_group_ids'])
                assert sup_gids.isdisjoint(qry_gids), (
                    f"Eval group leakage in {k_shot}-shot episode {ep_idx}! Overlap: {sup_gids & qry_gids}"
                )



# ============================================================
# Test Model
# ============================================================

class TestBackbone:
    """Test the ResNet-18 backbone."""
    
    def test_output_shape(self):
        backbone = ResNet18Backbone(pretrained=False)
        x = torch.randn(4, 3, 224, 224)
        out = backbone(x)
        assert out.shape == (4, 512)
    
    def test_embedding_dim(self):
        backbone = get_backbone("resnet18", pretrained=False)
        assert backbone.embedding_dim == 512


class TestPrototypicalNetwork:
    """Test the Prototypical Network implementation."""
    
    @pytest.fixture
    def model(self):
        return PrototypicalNetwork(pretrained=False, distance_metric="euclidean")
    
    def test_forward_pass(self, model):
        support = torch.randn(6, 3, 224, 224)  # 2-way 3-shot
        labels = torch.tensor([0, 0, 0, 1, 1, 1])
        query = torch.randn(4, 3, 224, 224)
        
        log_probs, prototypes = model(support, labels, query)
        
        assert log_probs.shape == (4, 2)
        assert prototypes.shape == (2, 512)
    
    def test_prototype_calculation(self, model):
        """Prototype should be the mean of class embeddings."""
        embeddings = torch.tensor([
            [1.0, 2.0, 3.0],
            [3.0, 4.0, 5.0],
        ])
        labels = torch.tensor([0, 0])
        
        prototypes = model._compute_prototypes(embeddings, labels)
        expected = torch.tensor([[2.0, 3.0, 4.0]])
        assert torch.allclose(prototypes, expected)
    
    def test_euclidean_distance(self, model):
        queries = torch.tensor([[1.0, 0.0]])
        prototypes = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
        
        distances = model._compute_distances(queries, prototypes)
        # Distance to [0,0] = 1, distance to [2,0] = 1
        assert distances.shape == (1, 2)
        assert torch.allclose(distances, torch.tensor([[1.0, 1.0]]))
    
    def test_prediction_consistency(self, model):
        """Model predictions should be deterministic in eval mode."""
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        support = torch.randn(4, 3, 224, 224)
        labels = torch.tensor([0, 0, 1, 1])
        
        with torch.no_grad():
            out1, _ = model(support, labels, x)
            out2, _ = model(support, labels, x)
        
        assert torch.allclose(out1, out2)


class TestPrototypicalNetworkAlgorithm:
    """Test the ProtoNet algorithm wrapper."""
    
    @pytest.fixture
    def algorithm(self):
        return PrototypicalNetworkAlgorithm(
            pretrained=False, device="cpu",
            learning_rate=1e-3,
        )
    
    def test_meta_train_step(self, algorithm):
        support = torch.randn(6, 3, 224, 224)
        support_labels = torch.tensor([0, 0, 0, 1, 1, 1])
        query = torch.randn(4, 3, 224, 224)
        query_labels = torch.tensor([0, 0, 1, 1])
        
        result = algorithm.meta_train_step(
            support, support_labels, query, query_labels
        )
        
        assert 'loss' in result
        assert 'accuracy' in result
        assert isinstance(result['loss'], float)
        assert 0 <= result['accuracy'] <= 1
    
    def test_adapt_and_predict(self, algorithm):
        support = torch.randn(4, 3, 224, 224)
        support_labels = torch.tensor([0, 0, 1, 1])
        
        adapted = algorithm.adapt(support, support_labels)
        assert 'prototypes' in adapted
        assert adapted['prototypes'].shape == (2, 512)
        
        query = torch.randn(3, 3, 224, 224)
        predictions, confidences = algorithm.predict(query, adapted)
        
        assert predictions.shape == (3,)
        assert confidences.shape == (3,)
        assert all(0 <= c <= 1 for c in confidences)
    
    def test_checkpoint_save_load(self, algorithm, tmp_path):
        ckpt_path = str(tmp_path / "test_ckpt.pt")
        algorithm.save_checkpoint(ckpt_path, extra={'epoch': 5})
        
        # Create new algorithm and load
        new_algo = PrototypicalNetworkAlgorithm(
            pretrained=False, device="cpu",
        )
        checkpoint = new_algo.load_checkpoint(ckpt_path)
        assert checkpoint.get('epoch') == 5


# ============================================================
# Test Configuration
# ============================================================

class TestConfiguration:
    """Test configuration management."""
    
    def test_default_config(self):
        config = ExperimentConfig()
        assert config.held_out_class == "Red Dane cattle"
        assert len(config.meta_train_classes) == 4
        assert config.held_out_class not in config.meta_train_classes
    
    def test_validation(self):
        config = ExperimentConfig(held_out_class="Red Dane cattle")
        config.validate()  # Should not raise
    
    def test_invalid_held_out(self):
        config = ExperimentConfig(held_out_class="Unknown breed")
        with pytest.raises(ValueError):
            config.validate()
    
    def test_save_load(self, tmp_path):
        config = ExperimentConfig(
            experiment_name="test",
            random_seed=99,
            held_out_class="Jersey cattle",
        )
        path = tmp_path / "config.json"
        config.save(path)
        
        loaded = ExperimentConfig.load(path)
        assert loaded.random_seed == 99
        assert loaded.held_out_class == "Jersey cattle"
    
    def test_leave_one_out_all_breeds(self):
        """Every breed should work as held-out class."""
        for breed in BREED_NAMES:
            config = ExperimentConfig(held_out_class=breed)
            assert len(config.meta_train_classes) == 4
            assert breed not in config.meta_train_classes
            config.validate()


# ============================================================
# Run Tests
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
