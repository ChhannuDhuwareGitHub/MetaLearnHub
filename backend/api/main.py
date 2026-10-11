"""
MetaLearnHub FastAPI Application.

Main entry point for the backend server. Exposes REST API endpoints
for dataset inspection, experiment management, training, evaluation,
and few-shot prediction.
"""

import os
import sys
import json
import logging
import asyncio
import threading
import base64
import uuid
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.config import (
    ExperimentConfig, PreprocessConfig, EpisodeConfig,
    ModelConfig, TrainingConfig, BREED_NAMES,
    RAW_DATA_DIR, METADATA_DIR, EXPERIMENTS_DIR, CHECKPOINTS_DIR,
)
from ml.datasets.inspector import DatasetInspector, inspect_dataset
from ml.datasets.splitter import DatasetSplitter, create_leakage_safe_split
from ml.training.trainer import MetaTrainer, get_training_state, request_cancel, get_device, set_seed
from ml.evaluation.evaluator import FewShotEvaluator, BaselineEvaluator
from ml.algorithms.prototypical_networks.protonet import PrototypicalNetworkAlgorithm
from ml.inference.predictor import FewShotPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================
# Application State
# ============================================================

class AppState:
    """Global application state."""
    
    def __init__(self):
        self.inspector: Optional[DatasetInspector] = None
        self.dataset_report: Optional[dict] = None
        self.current_algorithm: Optional[PrototypicalNetworkAlgorithm] = None
        self.predictor: Optional[FewShotPredictor] = None
        self.experiment_records: Dict[str, dict] = {}
        self.evaluation_results: Dict[str, dict] = {}
        self.baseline_results: Dict[str, dict] = {}
        self.training_thread: Optional[threading.Thread] = None
        self._load_existing_experiments()
    
    def _load_existing_experiments(self):
        """Load existing experiment records from disk."""
        if EXPERIMENTS_DIR.exists():
            for exp_dir in EXPERIMENTS_DIR.iterdir():
                if exp_dir.is_dir():
                    exp_file = exp_dir / "experiment.json"
                    if exp_file.exists():
                        try:
                            with open(exp_file) as f:
                                record = json.load(f)
                            self.experiment_records[record.get('experiment_id', exp_dir.name)] = record
                        except Exception:
                            pass
                    
                    eval_dir = exp_dir / "evaluation"
                    if eval_dir.exists():
                        summary_file = eval_dir / "evaluation_summary.json"
                        if summary_file.exists():
                            try:
                                with open(summary_file) as f:
                                    self.evaluation_results[exp_dir.name] = json.load(f)
                            except Exception:
                                pass
                    
                    baseline_file = exp_dir / "baseline_result.json"
                    if baseline_file.exists():
                        try:
                            with open(baseline_file) as f:
                                self.baseline_results[exp_dir.name] = json.load(f)
                        except Exception:
                            pass


app_state = AppState()


# ============================================================
# Pydantic Models
# ============================================================

class TrainRequest(BaseModel):
    held_out_class: str = "Red Dane cattle"
    algorithm: str = "prototypical_networks"
    backbone: str = "resnet18"
    pretrained: bool = True
    n_way: int = 3
    k_shot: int = 5
    query_per_class: int = 5
    episodes_per_epoch: int = 100
    num_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    random_seed: int = 42
    device: str = "auto"
    distance_metric: str = "euclidean"

class EvalRequest(BaseModel):
    experiment_id: str
    held_out_class: str = "Red Dane cattle"
    eval_shots: List[int] = [1, 3, 5, 10]
    eval_episodes: int = 50
    n_way: int = 2
    query_per_class: int = 10

class CreateClassRequest(BaseModel):
    class_name: str

class PredictRequest(BaseModel):
    class_ids: Optional[List[str]] = None


# ============================================================
# Lifespan
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize application state on startup."""
    logger.info("MetaLearnHub starting up...")
    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info(f"Raw data dir: {RAW_DATA_DIR}")
    
    # Auto-inspect dataset if available
    if RAW_DATA_DIR.exists():
        try:
            report, inspector = inspect_dataset(RAW_DATA_DIR)
            app_state.inspector = inspector
            from dataclasses import asdict
            app_state.dataset_report = asdict(report)
            logger.info(f"Dataset loaded: {report.total_files} files, "
                       f"{report.total_sample_groups} sample groups")
        except Exception as e:
            logger.error(f"Dataset inspection failed: {e}")
    else:
        logger.warning(f"Raw data directory not found: {RAW_DATA_DIR}")
    
    yield
    logger.info("MetaLearnHub shutting down...")


# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(
    title="MetaLearnHub",
    description="Few-Shot Image Classification via Meta-Learning",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Dataset Endpoints
# ============================================================

@app.get("/api/dataset/inspect")
async def inspect_dataset_endpoint():
    """Inspect the dataset and return the report."""
    if app_state.dataset_report:
        return app_state.dataset_report
    
    if not RAW_DATA_DIR.exists():
        raise HTTPException(404, f"Dataset not found at {RAW_DATA_DIR}")
    
    report, inspector = inspect_dataset(RAW_DATA_DIR)
    app_state.inspector = inspector
    from dataclasses import asdict
    app_state.dataset_report = asdict(report)
    return app_state.dataset_report


@app.get("/api/dataset/classes")
async def get_classes():
    """Get available breed classes."""
    return {
        "classes": BREED_NAMES,
        "count": len(BREED_NAMES),
    }


@app.get("/api/dataset/class/{breed_name}/images")
async def get_class_images(breed_name: str, limit: int = 10):
    """Get sample images from a breed class as base64."""
    if app_state.inspector is None:
        raise HTTPException(400, "Dataset not inspected yet")
    
    breed_name_decoded = breed_name.replace("_", " ")
    # Find the matching breed
    matched = None
    for b in BREED_NAMES:
        if b.lower().replace(" ", "_") == breed_name.lower() or b == breed_name_decoded or b == breed_name:
            matched = b
            break
    
    if not matched:
        raise HTTPException(404, f"Breed not found: {breed_name}")
    
    groups = app_state.inspector.get_sample_groups_by_breed(matched)
    images = []
    count = 0
    for group in groups:
        for img in group.images:
            if img.is_valid and not img.is_transformed and count < limit:
                try:
                    with open(img.filepath, 'rb') as f:
                        img_data = base64.b64encode(f.read()).decode()
                    images.append({
                        'filename': img.filename,
                        'data': f"data:image/jpeg;base64,{img_data}",
                        'width': img.width,
                        'height': img.height,
                        'sample_group_id': img.sample_group_id,
                    })
                    count += 1
                except Exception:
                    pass
        if count >= limit:
            break
    
    return {"breed": matched, "images": images, "total": len(images)}


@app.get("/api/dataset/stats")
async def get_dataset_stats():
    """Get dataset statistics summary."""
    if app_state.dataset_report is None:
        return {"status": "not_inspected"}
    
    report = app_state.dataset_report
    return {
        "status": "inspected",
        "total_files": report.get("total_files", 0),
        "total_sample_groups": report.get("total_sample_groups", 0),
        "breeds_found": report.get("breeds_found", []),
        "total_corrupted": report.get("total_corrupted", 0),
        "total_complete_pairs": report.get("total_complete_pairs", 0),
        "data_quality_passed": report.get("data_quality_passed", True),
        "quality_warnings": report.get("quality_warnings", []),
        "breed_stats": report.get("breed_stats", {}),
    }


# ============================================================
# Training Endpoints
# ============================================================

def _run_training(config: ExperimentConfig):
    """Training function to run in background thread."""
    try:
        if app_state.inspector is None:
            report, inspector = inspect_dataset(RAW_DATA_DIR)
            app_state.inspector = inspector
        
        trainer = MetaTrainer(config=config, inspector=app_state.inspector)
        result = trainer.train()
        
        app_state.experiment_records[result['experiment_id']] = result
        
        # Load the trained model for evaluation/prediction
        checkpoint_path = result.get('checkpoint_path')
        if checkpoint_path and Path(checkpoint_path).exists():
            device = get_device(config.training.device)
            algorithm = PrototypicalNetworkAlgorithm(
                backbone_name=config.model.backbone,
                pretrained=config.model.pretrained,
                distance_metric=config.model.distance_metric,
                device=str(device),
            )
            algorithm.load_checkpoint(checkpoint_path)
            app_state.current_algorithm = algorithm
            app_state.predictor = FewShotPredictor(algorithm, config.preprocess)
        
        logger.info(f"Training completed: {result['experiment_id']}")
    except Exception as e:
        logger.error(f"Background training failed: {e}", exc_info=True)


@app.post("/api/training/start")
async def start_training(request: TrainRequest):
    """Start meta-training in background."""
    training_state = get_training_state()
    if training_state.status == "running":
        raise HTTPException(409, "Training already in progress")
    
    config = ExperimentConfig(
        held_out_class=request.held_out_class,
        algorithm=request.algorithm,
        random_seed=request.random_seed,
        preprocess=PreprocessConfig(),
        episode=EpisodeConfig(
            n_way=request.n_way,
            k_shot=request.k_shot,
            query_per_class=request.query_per_class,
            episodes_per_epoch=request.episodes_per_epoch,
            num_epochs=request.num_epochs,
        ),
        model=ModelConfig(
            backbone=request.backbone,
            pretrained=request.pretrained,
            distance_metric=request.distance_metric,
        ),
        training=TrainingConfig(
            learning_rate=request.learning_rate,
            weight_decay=request.weight_decay,
            device=request.device,
        ),
    )
    
    # Validate
    try:
        config.validate()
    except ValueError as e:
        raise HTTPException(400, str(e))
    
    # Start background training
    thread = threading.Thread(target=_run_training, args=(config,), daemon=True)
    thread.start()
    app_state.training_thread = thread
    
    return {"status": "started", "message": "Training started in background"}


@app.get("/api/training/status")
async def get_training_status():
    """Get current training progress."""
    state = get_training_state()
    return state.to_dict()


@app.post("/api/training/cancel")
async def cancel_training():
    """Cancel current training."""
    request_cancel()
    return {"status": "cancel_requested"}


# ============================================================
# Experiment Endpoints
# ============================================================

@app.get("/api/experiments")
async def list_experiments():
    """List all experiment records."""
    experiments = list(app_state.experiment_records.values())
    # Sort by timestamp descending
    experiments.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    return {"experiments": experiments, "count": len(experiments)}


@app.get("/api/experiments/{experiment_id}")
async def get_experiment(experiment_id: str):
    """Get a specific experiment record."""
    if experiment_id in app_state.experiment_records:
        return app_state.experiment_records[experiment_id]
    
    # Try loading from disk
    exp_file = EXPERIMENTS_DIR / experiment_id / "experiment.json"
    if exp_file.exists():
        with open(exp_file) as f:
            record = json.load(f)
        app_state.experiment_records[experiment_id] = record
        return record
    
    raise HTTPException(404, f"Experiment not found: {experiment_id}")


# ============================================================
# Evaluation Endpoints
# ============================================================

def _run_evaluation(request_data: dict):
    """Evaluation function to run in background."""
    try:
        experiment_id = request_data['experiment_id']
        
        # Find checkpoint
        checkpoint_path = None
        if experiment_id in app_state.experiment_records:
            checkpoint_path = app_state.experiment_records[experiment_id].get('checkpoint_path')
        
        if not checkpoint_path:
            # Try to find it
            ckpt_dir = CHECKPOINTS_DIR / experiment_id
            best_ckpt = ckpt_dir / "best_model.pt"
            if best_ckpt.exists():
                checkpoint_path = str(best_ckpt)
        
        if not checkpoint_path or not Path(checkpoint_path).exists():
            logger.error(f"No checkpoint found for experiment {experiment_id}")
            return
        
        config = ExperimentConfig(
            held_out_class=request_data.get('held_out_class', 'Red Dane cattle'),
            eval_shots=request_data.get('eval_shots', [1, 3, 5, 10]),
            eval_episodes=request_data.get('eval_episodes', 50),
            random_seed=42,
        )
        
        if app_state.inspector is None:
            _, inspector = inspect_dataset(RAW_DATA_DIR)
            app_state.inspector = inspector
        
        device = get_device('auto')
        algorithm = PrototypicalNetworkAlgorithm(
            backbone_name='resnet18',
            pretrained=True,
            device=str(device),
        )
        algorithm.load_checkpoint(checkpoint_path)
        
        evaluator = FewShotEvaluator(
            algorithm=algorithm,
            inspector=app_state.inspector,
            config=config,
        )
        
        results = evaluator.evaluate_all_shots(
            n_way=request_data.get('n_way', 2),
            query_per_class=request_data.get('query_per_class', 10),
        )
        
        eval_dir = evaluator.save_results(results, experiment_id)
        
        # Store results
        summary = {
            'experiment_id': experiment_id,
            'timestamp': datetime.now().isoformat(),
            'held_out_class': config.held_out_class,
            'shot_results': {str(k): r.to_dict() for k, r in results.items()},
        }
        app_state.evaluation_results[experiment_id] = summary
        
        # Update algorithm for predictions
        app_state.current_algorithm = algorithm
        app_state.predictor = FewShotPredictor(algorithm, config.preprocess)
        
        logger.info(f"Evaluation completed for experiment {experiment_id}")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)


@app.post("/api/evaluation/run")
async def run_evaluation(request: EvalRequest):
    """Run few-shot evaluation on a trained model."""
    thread = threading.Thread(
        target=_run_evaluation,
        args=(request.model_dump(),),
        daemon=True,
    )
    thread.start()
    return {"status": "started", "message": "Evaluation started in background"}


@app.get("/api/evaluation/{experiment_id}")
async def get_evaluation_results(experiment_id: str):
    """Get evaluation results for an experiment."""
    if experiment_id in app_state.evaluation_results:
        return app_state.evaluation_results[experiment_id]
    
    # Try loading from disk
    eval_file = EXPERIMENTS_DIR / experiment_id / "evaluation" / "evaluation_summary.json"
    if eval_file.exists():
        with open(eval_file) as f:
            results = json.load(f)
        app_state.evaluation_results[experiment_id] = results
        return results
    
    raise HTTPException(404, f"No evaluation results for: {experiment_id}")


# ============================================================
# Baseline Endpoints
# ============================================================

def _run_baseline(config_data: dict):
    """Run baseline evaluation in background."""
    try:
        config = ExperimentConfig(
            held_out_class=config_data.get('held_out_class', 'Red Dane cattle'),
            random_seed=config_data.get('random_seed', 42),
        )
        
        if app_state.inspector is None:
            _, inspector = inspect_dataset(RAW_DATA_DIR)
            app_state.inspector = inspector
        
        baseline = BaselineEvaluator(
            inspector=app_state.inspector,
            config=config,
        )
        result = baseline.train_and_evaluate()
        app_state.baseline_results[result['experiment_id']] = result
        logger.info(f"Baseline completed: {result['experiment_id']}")
    except Exception as e:
        logger.error(f"Baseline failed: {e}", exc_info=True)


@app.post("/api/baseline/run")
async def run_baseline(held_out_class: str = "Red Dane cattle"):
    """Run conventional ResNet-18 baseline."""
    thread = threading.Thread(
        target=_run_baseline,
        args=({"held_out_class": held_out_class},),
        daemon=True,
    )
    thread.start()
    return {"status": "started", "message": "Baseline training started"}


@app.get("/api/baseline/results")
async def get_baseline_results():
    """Get all baseline results."""
    return {"results": list(app_state.baseline_results.values())}


# ============================================================
# Adaptation / Prediction Endpoints
# ============================================================

@app.post("/api/adaptation/create-class")
async def create_class(request: CreateClassRequest):
    """Create a new temporary class for few-shot prediction."""
    if app_state.predictor is None:
        raise HTTPException(
            400,
            "No model loaded. Train a model first or load a checkpoint."
        )
    
    temp_class = app_state.predictor.create_class(request.class_name)
    return temp_class.to_dict()


@app.post("/api/adaptation/{class_id}/add-support")
async def add_support_image(class_id: str, file: UploadFile = File(...)):
    """Add a support image to a temporary class."""
    if app_state.predictor is None:
        raise HTTPException(400, "No model loaded")
    
    try:
        image_bytes = await file.read()
        temp_class = app_state.predictor.add_support_image(
            class_id=class_id,
            image_bytes=image_bytes,
        )
        return temp_class.to_dict()
    except ValueError as e:
        raise HTTPException(404, str(e))


@app.post("/api/adaptation/{class_id}/build-prototype")
async def build_prototype(class_id: str):
    """Build the prototype for a temporary class."""
    if app_state.predictor is None:
        raise HTTPException(400, "No model loaded")
    
    try:
        temp_class = app_state.predictor.build_prototype(class_id)
        return {
            **temp_class.to_dict(),
            "message": f"Prototype built from {temp_class.k_shot} support images",
        }
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/prediction")
async def predict(
    file: UploadFile = File(...),
    class_ids: Optional[str] = Form(None),
):
    """Classify a query image against temporary classes."""
    if app_state.predictor is None:
        raise HTTPException(400, "No model loaded")
    
    try:
        image_bytes = await file.read()
        cids = json.loads(class_ids) if class_ids else None
        
        result = app_state.predictor.predict(
            query_image_bytes=image_bytes,
            class_ids=cids,
        )
        return result
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/adaptation/classes")
async def list_temporary_classes():
    """List all temporary classes."""
    if app_state.predictor is None:
        return {"classes": [], "model_loaded": False}
    return {
        "classes": app_state.predictor.list_classes(),
        "model_loaded": True,
    }


# ============================================================
# Model Endpoints
# ============================================================

@app.get("/api/models")
async def list_models():
    """List available model checkpoints."""
    models = []
    if CHECKPOINTS_DIR.exists():
        for exp_dir in CHECKPOINTS_DIR.iterdir():
            if exp_dir.is_dir():
                for ckpt in exp_dir.glob("*.pt"):
                    models.append({
                        'experiment_id': exp_dir.name,
                        'checkpoint': ckpt.name,
                        'path': str(ckpt),
                        'size_mb': round(ckpt.stat().st_size / (1024*1024), 2),
                    })
    return {"models": models}


@app.post("/api/models/load/{experiment_id}")
async def load_model(experiment_id: str):
    """Load a model checkpoint for prediction."""
    ckpt_path = CHECKPOINTS_DIR / experiment_id / "best_model.pt"
    if not ckpt_path.exists():
        raise HTTPException(404, f"Checkpoint not found: {ckpt_path}")
    
    try:
        device = get_device('auto')
        algorithm = PrototypicalNetworkAlgorithm(
            backbone_name='resnet18',
            pretrained=True,
            device=str(device),
        )
        algorithm.load_checkpoint(str(ckpt_path))
        app_state.current_algorithm = algorithm
        app_state.predictor = FewShotPredictor(algorithm)
        
        return {"status": "loaded", "experiment_id": experiment_id, "device": str(device)}
    except Exception as e:
        raise HTTPException(500, f"Failed to load model: {e}")


# ============================================================
# System Endpoints
# ============================================================

@app.get("/api/status")
async def get_system_status():
    """Get overall system status."""
    import torch
    
    return {
        "status": "online",
        "dataset_loaded": app_state.inspector is not None,
        "model_loaded": app_state.current_algorithm is not None,
        "total_experiments": len(app_state.experiment_records),
        "total_evaluations": len(app_state.evaluation_results),
        "breeds": BREED_NAMES,
        "cuda_available": torch.cuda.is_available(),
        "device": str(get_device('auto')),
        "training_status": get_training_state().status,
    }


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
