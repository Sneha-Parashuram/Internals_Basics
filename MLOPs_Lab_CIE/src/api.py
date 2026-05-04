from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd

app = FastAPI()

# Load model
MODEL_PATH = "models/model.pkl"
model = joblib.load(MODEL_PATH)

# Input schema
class InputData(BaseModel):
    stride_length_cm: float = Field(..., ge=80, le=160)
    ground_contact_ms: float = Field(..., ge=150, le=350)
    hip_drop_degrees: float = Field(..., ge=2, le=15)
    fatigue_index: float = Field(..., ge=1, le=10)

# Health endpoint
@app.get("/ping")
def health():
    return {
        "status": "healthy",
        "model_loaded": True
    }

# Prediction endpoint
@app.post("/infer")
def predict(data: InputData):
    try:
        df = pd.DataFrame([data.dict()])
        pred = model.predict(df)[0]

        return {
            "prediction": float(round(pred, 3))
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))