from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI

from .routes import router
from ..model.utils import load_config, setup_logging


LOGGER = logging.getLogger(__name__)


def create_app() -> FastAPI:
    repo_root = Path(__file__).resolve().parents[2]
    setup_logging(repo_root)
    cfg = load_config(repo_root)

    app = FastAPI(
        title="City Risk Prediction API",
        version="0.1.0",
        description="Backend-only API for City Risk Prediction Model (synthetic baseline)",
    )

    app.state.repo_root = repo_root
    app.state.cfg = cfg

    app.include_router(router)
    return app


app = create_app()
