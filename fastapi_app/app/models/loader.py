import joblib
import os
from app.core.config import settings

def load_model():
    model_path = settings.model_path
    model_version = settings.model_version

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo en {model_path}")

    model = joblib.load(model_path)
    return model, model_version
