from fastapi import APIRouter
from app.schemas.input_data import InputData
from app.models.loader import load_model
import numpy as np

router = APIRouter(prefix="/predict", tags=["Prediction"])

# Cargar el modelo al inicio
model, model_version = load_model()

@router.post("/")
def make_prediction(data: InputData):
    X = np.array(data.features).reshape(1, -1)

    # Predicción
    prediction = model.predict(X)[0]

    # Probabilidad si el modelo la soporta
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(X)[0][int(prediction)])
    else:
        probability = None

    return {
        "prediction": int(prediction),
        "probability": round(probability, 4) if probability is not None else None,
        "model_version": model_version
    }
