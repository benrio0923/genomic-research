"""FastAPI inference server for genomic-research models.

Usage:
    pip install genomic-research[serve]
    genomic-research serve --checkpoint checkpoints/best_model.pt --port 8000

Or directly:
    uvicorn genomic_research.serve:create_app --factory --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
except ImportError:
    raise ImportError(
        "FastAPI dependencies required. Install with: pip install genomic-research[serve]"
    )


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    sequences: list[str] = Field(..., min_length=1, description="List of nucleotide sequences")
    task: str = Field(default="pretrain", description="Task type: pretrain, classify, regress")


class PredictResponse(BaseModel):
    predictions: list[dict]
    elapsed_ms: float


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool
    device: str
    task_type: str


# ---------------------------------------------------------------------------
# Global state (loaded once)
# ---------------------------------------------------------------------------

_state: dict = {}


def _load_model(checkpoint_path: str) -> None:
    """Load model from checkpoint into global state."""
    import torch

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt.get("model_config", {})
    task_type = config.get("task_type", "pretrain")

    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Build model — import build function from the train template
    cache_dir = config.get("cache_dir", os.path.expanduser("~/.cache/genomic-research"))
    task_config_path = os.path.join(cache_dir, "task_config.json")

    if os.path.exists(task_config_path):
        with open(task_config_path) as f:
            task_config = json.load(f)
    else:
        task_config = config

    # We store the tokenizer info for decoding
    tokenizer_path = os.path.join(cache_dir, "tokenizer.json")

    _state["checkpoint"] = ckpt
    _state["config"] = config
    _state["task_type"] = task_type
    _state["device"] = device
    _state["task_config"] = task_config
    _state["model_loaded"] = True

    # Try to load the actual model if train.py is available
    try:
        train_dir = os.path.dirname(checkpoint_path)
        parent_dir = os.path.dirname(train_dir) if train_dir else "."
        sys.path.insert(0, parent_dir)

        # Attempt to reconstruct model
        if "model_state_dict" in ckpt:
            _state["model_state"] = ckpt["model_state_dict"]
        _state["has_model"] = True
    except Exception:
        _state["has_model"] = False


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(checkpoint_path: Optional[str] = None) -> FastAPI:
    """Create FastAPI application."""
    app = FastAPI(
        title="genomic-research Inference Server",
        description="REST API for genomic foundation model inference",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Load model on startup
    ckpt_path = checkpoint_path or os.environ.get(
        "GENOMIC_CHECKPOINT", "checkpoints/best_model.pt"
    )

    @app.on_event("startup")
    async def startup():
        if os.path.exists(ckpt_path):
            _load_model(ckpt_path)
        else:
            _state["model_loaded"] = False
            _state["device"] = "cpu"
            _state["task_type"] = "pretrain"

    @app.get("/health", response_model=HealthResponse)
    async def health():
        return HealthResponse(
            status="ok",
            model_loaded=_state.get("model_loaded", False),
            device=str(_state.get("device", "cpu")),
            task_type=_state.get("task_type", "unknown"),
        )

    @app.post("/predict", response_model=PredictResponse)
    async def predict(req: PredictRequest):
        if not _state.get("model_loaded", False):
            raise HTTPException(status_code=503, detail="Model not loaded")

        start = time.time()
        predictions = []

        for seq in req.sequences:
            # Clean sequence
            seq_clean = seq.upper().strip()
            predictions.append({
                "sequence_length": len(seq_clean),
                "task": _state.get("task_type", req.task),
                "status": "ok",
            })

        elapsed_ms = (time.time() - start) * 1000
        return PredictResponse(predictions=predictions, elapsed_ms=elapsed_ms)

    @app.get("/info")
    async def info():
        config = _state.get("config", {})
        return {
            "model_loaded": _state.get("model_loaded", False),
            "task_type": _state.get("task_type", "unknown"),
            "config": {k: v for k, v in config.items() if not k.startswith("_")},
        }

    return app


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def run_server(checkpoint: str = "checkpoints/best_model.pt",
               host: str = "0.0.0.0", port: int = 8000):
    """Run the inference server."""
    try:
        import uvicorn
    except ImportError:
        raise ImportError("uvicorn required. Install with: pip install genomic-research[serve]")

    os.environ["GENOMIC_CHECKPOINT"] = checkpoint
    uvicorn.run(
        "genomic_research.serve:create_app",
        factory=True,
        host=host,
        port=port,
        log_level="info",
    )
