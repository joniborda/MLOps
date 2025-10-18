from fastapi import APIRouter
from app.models.loader import load_model

router = APIRouter(tags=["Model Management"])

model, model_version = load_model()

@router.get("/model/info")
def model_info():
    return {
        "model_version": model_version,
        "details": "Modelo actualmente cargado en memoria."
    }

@router.post("/model/reload")
def reload_model():
    global model, model_version
    model, model_version = load_model()
    return {
        "message": "Modelo recargado correctamente.",
        "model_version": model_version
    }
