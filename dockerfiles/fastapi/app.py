from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import mlflow.sklearn
import boto3
import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="MLOps Model Service",
    description="API para servir modelos de machine learning",
    version="1.0.0"
)

# Configure MLflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))

# Configure S3 (MinIO)
s3_client = boto3.client(
    's3',
    endpoint_url=os.getenv("AWS_ENDPOINT_URL_S3", "http://s3:9000"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "minio"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "minio123")
)

# Global variable to store the loaded model
model = None
model_version = None

class PredictionRequest(BaseModel):
    """Request model for predictions"""
    features: List[float]
    
class PredictionResponse(BaseModel):
    """Response model for predictions"""
    prediction: int
    probability: float
    model_version: str

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    mlflow_connected: bool

@app.on_event("startup")
async def load_model():
    """Load the latest model from MLflow on startup"""
    global model, model_version
    try:
        # Get the latest model version
        client = mlflow.tracking.MlflowClient()
        latest_versions = client.get_latest_versions("heart_disease_classifier")
        
        if latest_versions:
            latest_version = latest_versions[0]
            model_version = latest_version.version
            model_uri = f"models:/heart_disease_classifier/{model_version}"
            
            # Load the model
            model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"Model loaded successfully. Version: {model_version}")
        else:
            logger.warning("No model found in MLflow registry")
            
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")

@app.get("/", response_model=Dict[str, str])
def read_root():
    """Root endpoint"""
    return {"message": "Welcome to the MLOps Model Service", "version": "1.0.0"}

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint"""
    mlflow_connected = False
    try:
        # Test MLflow connection
        client = mlflow.tracking.MlflowClient()
        client.search_experiments()
        mlflow_connected = True
    except Exception as e:
        logger.error(f"MLflow connection error: {str(e)}")
    
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        mlflow_connected=mlflow_connected
    )

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Make predictions using the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert features to numpy array
        features_array = np.array(request.features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features_array)[0]
        
        # Get prediction probability if available
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(features_array)[0].max()
        else:
            probability = 0.5  # Default probability if not available
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            model_version=model_version or "unknown"
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/model/info")
def get_model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=404, detail="No model loaded")
    
    return {
        "model_type": type(model).__name__,
        "version": model_version,
        "features_expected": getattr(model, 'n_features_in_', 'unknown')
    }

@app.post("/model/reload")
def reload_model():
    """Reload the latest model from MLflow"""
    global model, model_version
    try:
        load_model()
        return {"message": "Model reloaded successfully", "version": model_version}
    except Exception as e:
        logger.error(f"Model reload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model reload failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8800)
