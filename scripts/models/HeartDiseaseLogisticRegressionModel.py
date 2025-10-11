"""
Implementacion del modelo Regresión Logística para clasificación de enfermedades cardíacas
Este módulo hereda de BaseModel e implementa el método train_model.
Contiene toda la lógica de entrenamiento del modelo
"""

from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import logging
import os

from BaseModel import BaseModel
from sklearn.linear_model import LogisticRegression

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HeartDiseaseLogisticRegressionModel(BaseModel):
    """
    Modelo de Regresión Logística para clasificación de enfermedades cardíacas
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
        logger.info("Iniciando entrenamiento del modelo de Regresión Logística...")
        
        # Parámetros por defecto
        default_params = {
            'penalty': 'l2',
            'C': 10,
            'random_state': self.random_state
        }
        
        # Actualizar con parámetros proporcionados
        default_params.update(model_params)
        
        # Crear y entrenar modelo
        self.model = LogisticRegression(**default_params)
        self.model = LogisticRegression()
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        logger.info(f"Modelo de Regresión Logística entrenado con parámetros: {default_params}")
        
        return default_params