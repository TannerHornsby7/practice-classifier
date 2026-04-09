"""
Classifier service — loads model.pkl at startup, exposes POST /predict.
Stateless: can be scaled horizontally behind a load balancer.
"""

import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="No-Show Classifier")

# --- load model once at startup ---
with open("model.pkl", "rb") as f:
    artifact = pickle.load(f)

MODEL    = artifact["model"]
FEATURES = artifact["features"]
TEST_AUC = artifact["test_auc"]

print(f"Loaded model — AUC={TEST_AUC:.4f}, features={len(FEATURES)}")


class AppointmentRecord(BaseModel):
    """One appointment row. All fields must match the trained feature set."""
    model_config = {"extra": "allow"}  # pass through unknown fields without error


@app.get("/health")
def health():
    return {"status": "ok", "test_auc": TEST_AUC, "n_features": len(FEATURES)}


@app.get("/features")
def features():
    """Return the feature list the model expects — useful for debugging."""
    return {"features": FEATURES}


@app.post("/predict")
def predict(record: AppointmentRecord):
    data = record.model_dump()

    # build a single-row DataFrame in the exact feature order the model expects
    try:
        row = pd.DataFrame([{f: data.get(f, 0) for f in FEATURES}])
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Feature construction failed: {e}")

    proba = MODEL.predict_proba(row)[0]  # [P(show), P(no-show)]
    return {
        "p_show":    round(float(proba[0]), 4),
        "p_no_show": round(float(proba[1]), 4),
        "prediction": "no_show" if proba[1] >= 0.5 else "show",
    }
