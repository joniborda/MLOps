from fastapi import FastAPI
from app.api import predict, model_info

app = FastAPI(title="MLOps Model API", version="1.0")

# Rutas principales
app.include_router(predict.router)
app.include_router(model_info.router)

@app.get("/")
def root():
    return {
        "message": "API de Predicción - Proyecto MLOps",
        "endpoints": ["/predict", "/health", "/model/info", "/model/reload"]
    }

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "FastAPI funcionando correctamente"}

