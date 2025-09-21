"""
Ejemplo de cómo usar el modelo de enfermedades cardíacas
Este script demuestra cómo entrenar y evaluar el modelo usando los datos reales
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.model_utils import HeartDiseaseModel
import pandas as pd
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Función principal para entrenar el modelo"""
    
    # Crear instancia del modelo
    model = HeartDiseaseModel(random_state=42)
    
    # Cargar datos reales
    data_path = "data/heart.csv"
    if os.path.exists(data_path):
        logger.info("Usando datos reales del dataset de heart disease")
        data = model.load_data(data_path)
    else:
        logger.info("Datos reales no encontrados, creando datos de muestra")
        data = model.create_sample_data()
        # Guardar datos de muestra
        data.to_csv(data_path, index=False)
        logger.info(f"Datos de muestra guardados en: {data_path}")
    
    # Preprocesar datos
    X_train, X_test, y_train, y_test = model.preprocess_data(data)
    
    # Entrenar modelo
    model_params = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2
    }
    
    model.train_model(X_train, y_train, **model_params)
    
    # Evaluar modelo
    metrics = model.evaluate_model(X_test, y_test)
    
    # Mostrar importancia de características
    importance = model.get_feature_importance()
    logger.info("Importancia de características:")
    logger.info(f"\n{importance}")
    
    # Guardar modelo
    model_path = "models/heart_disease_model.pkl"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save_model(model_path)
    
    # Ejemplo de predicción
    logger.info("\n=== Ejemplo de Predicción ===")
    sample_data = X_test.head(1)
    predictions, probabilities = model.predict(sample_data)
    
    logger.info(f"Datos de entrada: {sample_data.iloc[0].to_dict()}")
    logger.info(f"Predicción: {predictions[0]}")
    logger.info(f"Probabilidad: {probabilities[0]:.4f}")
    
    return model, metrics

if __name__ == "__main__":
    model, metrics = main()
    logger.info(f"\nMétricas finales: {metrics}")
