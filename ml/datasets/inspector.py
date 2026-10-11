"""
Dataset Inspection Module for MetaLearnHub.

Scans the raw cattle breed dataset, identifies base/_c pairs,
detects data quality issues, and generates a comprehensive report.
"""

import os
import re
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from datetime import datetime

from PIL import Image
import numpy as np

from ml.config import RAW_DATA_DIR, METADATA_DIR, BREED_NAMES


# ============================================================
# Data Structures
# ============================================================

@dataclass
class ImageRecord:
    """Record for a single image file."""
    filepath: str
    filename: str
    breed: str
    sample_group_id: str
    is_transformed: bool  # True if filename ends with _c
    width: int = 0
    height: int = 0
    file_size: int = 0
    is_valid: bool = True
    error_message: str = ""
    file_hash: str = ""
    
    @property
    def aspect_ratio(self) -> float:
        if self.height > 0:
            return round(self.width / self.height, 3)
        return 0.0


@dataclass
class SampleGroup:
    """A group of related images (base + _c counterpart)."""
    group_id: str
    breed: str
    images: List[ImageRecord] = field(default_factory=list)
    
    @property
    def has_base(self) -> bool:
        return any(not img.is_transformed for img in self.images)
    
    @property
    def has_transformed(self) -> bool:
        return any(img.is_transformed for img in self.images)
    
    @property
    def is_complete_pair(self) -> bool:
        return self.has_base and self.has_transformed


@dataclass
class BreedStats:
    """Statistics for a single breed."""
    breed_name: str
    total_files: int = 0
    base_files: int = 0
    transformed_files: int = 0
    sample_groups: int = 0
    complete_pairs: int = 0
    incomplete_groups: int = 0
    corrupted_files: int = 0
    zero_byte_files: int = 0
    min_width: int = 0
    max_width: int = 0
    min_height: int = 0
    max_height: int = 0
    avg_width: float = 0.0
    avg_height: float = 0.0


@dataclass
class DatasetReport:
    """Full dataset inspection report."""
    timestamp: str = ""
    raw_data_path: str = ""
    total_files: int = 0
    total_jpg_files: int = 0
    total_non_jpg_files: int = 0
    total_sample_groups: int = 0
    total_corrupted: int = 0
    total_zero_byte: int = 0
    total_complete_pairs: int = 0
    total_incomplete_groups: int = 0
    breed_stats: Dict[str, dict] = field(default_factory=dict)
    corrupted_files: List[str] = field(default_factory=list)
    zero_byte_files: List[str] = field(default_factory=list)
    missing_counterparts: List[dict] = field(default_factory=list)
    unsupported_files: List[str] = field(default_factory=list)
    dimension_distribution: Dict[str, int] = field(default_factory=dict)
    breeds_found: List[str] = field(default_factory=list)
    data_quality_passed: bool = True
    quality_warnings: List[str] = field(default_factory=list)


# ============================================================
# File Parsing
# ============================================================

def parse_filename(filename: str) -> Tuple[str, bool]:
    """
    Parse a filename to extract sample_group_id and transformed status.
    
    Examples:
        Ayrshirecattle100.jpg -> ("Ayrshirecattle100", False)
        Ayrshirecattle100_c.jpg -> ("Ayrshirecattle100", True)
    
    Returns:
        (sample_group_id, is_transformed)
    """
    name = Path(filename).stem  # Remove extension
    
    if name.endswith('_c'):
        return name[:-2], True
    else:
        return name, False


def compute_file_hash(filepath: str, chunk_size: int = 8192) -> str:
    """Compute MD5 hash of a file for duplicate detection."""
    h = hashlib.md5()
    try:
        with open(filepath, 'rb') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


# ============================================================
# Dataset Inspector
# ============================================================

class DatasetInspector:
    """Inspects the raw cattle breed dataset and generates reports."""
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = Path(data_dir) if data_dir else RAW_DATA_DIR
        self.records: List[ImageRecord] = []
        self.sample_groups: Dict[str, SampleGroup] = {}
        self.breed_groups: Dict[str, List[SampleGroup]] = defaultdict(list)
    
    def scan(self) -> DatasetReport:
        """
        Full dataset scan. Returns a comprehensive report.
        """
        report = DatasetReport(
            timestamp=datetime.now().isoformat(),
            raw_data_path=str(self.data_dir)
        )
        
        if not self.data_dir.exists():
            report.data_quality_passed = False
            report.quality_warnings.append(
                f"Data directory not found: {self.data_dir}"
            )
            return report
        
        # Scan each breed directory
        breeds_found = []
        for breed_dir in sorted(self.data_dir.iterdir()):
            if not breed_dir.is_dir():
                continue
            breed_name = breed_dir.name
            breeds_found.append(breed_name)
            self._scan_breed(breed_dir, breed_name, report)
        
        report.breeds_found = breeds_found
        
        # Check for expected breeds
        for expected in BREED_NAMES:
            if expected not in breeds_found:
                report.quality_warnings.append(
                    f"Expected breed directory not found: {expected}"
                )
        
        # Aggregate statistics
        self._aggregate_stats(report)
        
        # Check for duplicates across breeds
        self._check_duplicates(report)
        
        return report
    
    def _scan_breed(self, breed_dir: Path, breed_name: str, report: DatasetReport):
        """Scan a single breed directory."""
        breed_stats = BreedStats(breed_name=breed_name)
        widths = []
        heights = []
        
        for filepath in sorted(breed_dir.iterdir()):
            if filepath.is_dir():
                continue
            
            filename = filepath.name
            
            # Check if JPG
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                report.unsupported_files.append(str(filepath))
                report.total_non_jpg_files += 1
                continue
            
            report.total_files += 1
            if filename.lower().endswith(('.jpg', '.jpeg')):
                report.total_jpg_files += 1
            
            # Check for zero-byte files
            file_size = filepath.stat().st_size
            if file_size == 0:
                report.zero_byte_files.append(str(filepath))
                report.total_zero_byte += 1
                breed_stats.zero_byte_files += 1
                continue
            
            # Parse filename
            group_id, is_transformed = parse_filename(filename)
            
            # Create image record
            record = ImageRecord(
                filepath=str(filepath),
                filename=filename,
                breed=breed_name,
                sample_group_id=group_id,
                is_transformed=is_transformed,
                file_size=file_size,
            )
            
            # Try to open and validate image
            try:
                with Image.open(filepath) as img:
                    img.verify()
                # Re-open to get dimensions (verify closes the file)
                with Image.open(filepath) as img:
                    record.width, record.height = img.size
                    widths.append(record.width)
                    heights.append(record.height)
            except Exception as e:
                record.is_valid = False
                record.error_message = str(e)
                report.corrupted_files.append(str(filepath))
                report.total_corrupted += 1
                breed_stats.corrupted_files += 1
            
            self.records.append(record)
            
            # Organize into sample groups
            full_group_id = f"{breed_name}/{group_id}"
            if full_group_id not in self.sample_groups:
                group = SampleGroup(group_id=group_id, breed=breed_name)
                self.sample_groups[full_group_id] = group
            self.sample_groups[full_group_id].images.append(record)
            
            # Count
            breed_stats.total_files += 1
            if is_transformed:
                breed_stats.transformed_files += 1
            else:
                breed_stats.base_files += 1
        
        # Calculate dimension stats
        if widths:
            breed_stats.min_width = min(widths)
            breed_stats.max_width = max(widths)
            breed_stats.min_height = min(heights)
            breed_stats.max_height = max(heights)
            breed_stats.avg_width = round(np.mean(widths), 1)
            breed_stats.avg_height = round(np.mean(heights), 1)
        
        # Count groups
        breed_sample_groups = [
            g for g in self.sample_groups.values() if g.breed == breed_name
        ]
        breed_stats.sample_groups = len(breed_sample_groups)
        breed_stats.complete_pairs = sum(
            1 for g in breed_sample_groups if g.is_complete_pair
        )
        breed_stats.incomplete_groups = sum(
            1 for g in breed_sample_groups if not g.is_complete_pair
        )
        
        # Record missing counterparts
        for g in breed_sample_groups:
            if not g.is_complete_pair:
                report.missing_counterparts.append({
                    'group_id': g.group_id,
                    'breed': breed_name,
                    'has_base': g.has_base,
                    'has_transformed': g.has_transformed,
                })
        
        self.breed_groups[breed_name] = breed_sample_groups
        report.breed_stats[breed_name] = asdict(breed_stats)
    
    def _aggregate_stats(self, report: DatasetReport):
        """Aggregate statistics across all breeds."""
        report.total_sample_groups = len(self.sample_groups)
        report.total_complete_pairs = sum(
            1 for g in self.sample_groups.values() if g.is_complete_pair
        )
        report.total_incomplete_groups = sum(
            1 for g in self.sample_groups.values() if not g.is_complete_pair
        )
        
        # Dimension distribution
        dim_counts = defaultdict(int)
        for record in self.records:
            if record.is_valid and record.width > 0:
                dim_key = f"{record.width}x{record.height}"
                dim_counts[dim_key] += 1
        report.dimension_distribution = dict(
            sorted(dim_counts.items(), key=lambda x: -x[1])
        )
        
        # Quality check
        if report.total_corrupted > 0:
            report.quality_warnings.append(
                f"Found {report.total_corrupted} corrupted file(s)"
            )
        if report.total_zero_byte > 0:
            report.quality_warnings.append(
                f"Found {report.total_zero_byte} zero-byte file(s)"
            )
        if report.total_incomplete_groups > 0:
            report.quality_warnings.append(
                f"Found {report.total_incomplete_groups} incomplete base/_c pair(s)"
            )
        if report.quality_warnings:
            report.data_quality_passed = False
    
    def _check_duplicates(self, report: DatasetReport):
        """Check for suspicious duplicate files across breeds."""
        # Only compute hashes for valid files
        hash_to_files = defaultdict(list)
        for record in self.records:
            if record.is_valid and record.file_size > 0:
                file_hash = compute_file_hash(record.filepath)
                record.file_hash = file_hash
                hash_to_files[file_hash].append(record.filepath)
        
        # Find duplicates (same hash, different file)
        duplicates = {
            h: files for h, files in hash_to_files.items()
            if len(files) > 1
        }
        if duplicates:
            report.quality_warnings.append(
                f"Found {len(duplicates)} potential duplicate group(s)"
            )
    
    def get_metadata_records(self) -> List[dict]:
        """Get all image records as metadata dictionaries."""
        return [asdict(r) for r in self.records]
    
    def get_sample_groups_by_breed(self, breed: str) -> List[SampleGroup]:
        """Get sample groups for a specific breed."""
        return self.breed_groups.get(breed, [])
    
    def save_metadata(self, output_dir: Path = None):
        """Save all metadata to the metadata directory."""
        output_dir = output_dir or METADATA_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save image records
        records_path = output_dir / "image_records.json"
        with open(records_path, 'w') as f:
            json.dump(self.get_metadata_records(), f, indent=2)
        
        # Save sample groups
        groups_data = {}
        for gid, group in self.sample_groups.items():
            groups_data[gid] = {
                'group_id': group.group_id,
                'breed': group.breed,
                'is_complete_pair': group.is_complete_pair,
                'images': [asdict(img) for img in group.images],
            }
        groups_path = output_dir / "sample_groups.json"
        with open(groups_path, 'w') as f:
            json.dump(groups_data, f, indent=2)
        
        return records_path, groups_path
    
    def save_report(self, report: DatasetReport, output_dir: Path = None) -> Path:
        """Save the inspection report."""
        output_dir = output_dir or METADATA_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "dataset_report.json"
        with open(report_path, 'w') as f:
            json.dump(asdict(report), f, indent=2)
        return report_path


def inspect_dataset(data_dir: Path = None, save: bool = True) -> DatasetReport:
    """
    Convenience function to run full dataset inspection.
    
    Returns:
        DatasetReport with full inspection results
    """
    inspector = DatasetInspector(data_dir)
    report = inspector.scan()
    
    if save:
        inspector.save_metadata()
        inspector.save_report(report)
    
    return report, inspector
