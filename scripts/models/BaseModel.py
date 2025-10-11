"""
Estructura del modelo base para clasificación de enfermedades cardíacas
Este módulo contiene toda la lógica de:
- Carga de datos
- Preprocesamiento
- Evaluación
- Guardado y cargado de modelos
- El entrenamiento del modelo debe ser implementado en el módulo que herede de BaseModel
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import logging
import os

from abc import ABC, abstractmethod

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """
    Clase base para el modelo de clasificación de enfermedades cardíacas
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.is_trained = False

    def load_data(self, file_path):
        """
        Cargar datos desde un archivo CSV
        
        Args:
            file_path (str): Ruta al archivo CSV
            
        Returns:
            pd.DataFrame: Datos cargados
        """
        try:
            logger.info(f"Cargando datos desde: {file_path}")
            data = pd.read_csv(file_path)
            logger.info(f"Datos cargados: {data.shape[0]} filas, {data.shape[1]} columnas")
            return data
        except Exception as e:
            logger.error(f"Error cargando datos: {str(e)}")
            raise

    def preprocess_data(self, data):
        """
        Preprocesar los datos para el entrenamiento
        
        Args:
            data (pd.DataFrame): Datos sin procesar
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test, feature_columns)
        """
        logger.info("Iniciando preprocesamiento de datos...")
        
        # Verificar que tenemos la columna target
        if 'target' not in data.columns:
            logger.error("Columna 'target' no encontrada en los datos")
            raise ValueError("Columna 'target' no encontrada")
        
        # Separar características y target
        X = data.drop('target', axis=1)
        y = data['target']
        
        # Guardar nombres de columnas
        self.feature_columns = X.columns.tolist()
        
        # Dividir en train y test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Escalar características
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convertir de vuelta a DataFrame
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=self.feature_columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=self.feature_columns)
        
        logger.info(f"Datos preprocesados - Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    @abstractmethod
    def train_model(self, X_train, y_train, **model_params):
        """
        Metodo abstracto para entrenar el modelo con los datos proporcionados
        """
        pass

    def evaluate_model(self, X_test, y_test):
        """
        Evaluar el modelo entrenado
        
        Args:
            X_test (pd.DataFrame): Características de prueba
            y_test (pd.Series): Target de prueba
            
        Returns:
            dict: Métricas de evaluación
        """
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado aún")
        
        logger.info("Evaluando modelo...")
        
        # Hacer predicciones
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calcular métricas
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred)
        }
        
        # Log de métricas
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        # Reporte detallado
        logger.info("Reporte de clasificación:")
        logger.info(f"\n{classification_report(y_test, y_pred)}")
        
        return metrics
    
    def predict(self, X):
        """
        Hacer predicciones con el modelo entrenado
        
        Args:
            X (pd.DataFrame): Características para predicción
            
        Returns:
            tuple: (predicciones, probabilidades)
        """
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado aún")
        
        # Escalar características
        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_columns)
        
        # Hacer predicciones
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        
        return predictions, probabilities
    
    def save_model(self, file_path):
        """
        Guardar el modelo entrenado
        
        Args:
            file_path (str): Ruta donde guardar el modelo
        """
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado aún")
        
        import joblib
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, file_path)
        logger.info(f"Modelo guardado en: {file_path}")
    
    def load_model(self, file_path):
        """
        Cargar un modelo previamente entrenado
        
        Args:
            file_path (str): Ruta del modelo guardado
        """
        import joblib
        model_data = joblib.load(file_path)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.is_trained = model_data['is_trained']
        
        logger.info(f"Modelo cargado desde: {file_path}")