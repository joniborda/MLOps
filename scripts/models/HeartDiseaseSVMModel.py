"""
Implementacion del modelo SVM para clasificación de enfermedades cardíacas
Este módulo hereda de BaseModel e implementa el método train_model.
Contiene toda la lógica de entrenamiento del modelo
"""

from sklearn.svm import SVC
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

class HeartDiseaseSVMModel(BaseModel):
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
        logger.info("Iniciando entrenamiento del modelo SVM...")
        
        # Parámetros por defecto
        default_params = {
            'kernel': 'rbf',
            'C': 10,
            'gamma': 0.1,
            'random_state': self.random_state
        }
        
        # Actualizar con parámetros proporcionados
        default_params.update(model_params)
        
        # Crear y entrenar modelo
        self.model = SVC(**default_params)
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        logger.info(f"Modelo SVM entrenado con parámetros: {default_params}")
        
        return default_params