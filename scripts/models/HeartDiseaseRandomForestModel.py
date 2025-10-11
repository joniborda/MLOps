"""
Implementacion del modelo Random Forest para clasificación de enfermedades cardíacas
Este módulo hereda de BaseModel e implementa el método train_model.
Contiene toda la lógica de entrenamiento del modelo
"""

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import logging
import os

from BaseModel import BaseModel

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HeartDiseaseRandomForestModel(BaseModel):
    """
    Modelo Random Forest para clasificación de enfermedades cardíacas
    """
   
    def train_model(self, X_train, y_train, **model_params):
        """
        Entrenar el modelo con los datos proporcionados
        
        Args:
            X_train (pd.DataFrame): Características de entrenamiento
            y_train (pd.Series): Target de entrenamiento
            **model_params: Parámetros del modelo
            
        Returns:
            dict: Métricas del modelo entrenado
        """
        logger.info("Iniciando entrenamiento del modelo Random Forest...")
        
        # Parámetros por defecto
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': self.random_state
        }
        
        # Actualizar con parámetros proporcionados
        default_params.update(model_params)
        
        # Crear y entrenar modelo
        self.model = RandomForestClassifier(**default_params)
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        logger.info(f"Modelo Random Forest entrenado con parámetros: {default_params}")
        
        return default_params

    def get_feature_importance(self):
        """
        Obtener importancia de características
        
        Returns:
            pd.DataFrame: Importancia de características
        """
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado aún")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df