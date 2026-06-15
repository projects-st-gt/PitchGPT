"""Lightweight server that only mounts the mcsim router (no model loading).

Used for testing the Score Prediction UI without waiting for model artifacts.

    uvicorn inference.serve_mcsim_only:app --port 8000
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from inference.mcsim_api import router as mcsim_router

app = FastAPI(title="PitchGPT MCSim (lightweight)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(mcsim_router)
