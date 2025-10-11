import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import logging
import os

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import logging

# Configuración básica del logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataGenerator:
    """
    Clase para la generación de datos de muestra.
    """
    @staticmethod
    def create_sample_data(n_samples=1000, seed=42):
        """
        Crea datos sintéticos de muestra para testing.

        Args:
            n_samples (int): El número de filas a generar.
            seed (int): La semilla para la reproducibilidad de los datos.

        Returns:
            pd.DataFrame: Un DataFrame de pandas con los datos de muestra.
        """
        logger.info("Creando datos de muestra...")
        
        # Crear datos sintéticos basados en el dataset real
        np.random.seed(seed)
        
        # Generar características similares al dataset real
        data = {
            'age': np.random.normal(54, 9, n_samples),
            'sex': np.random.choice([0, 1], n_samples),
            'cp': np.random.choice([0, 1, 2, 3], n_samples),
            'trestbps': np.random.normal(131, 18, n_samples),
            'chol': np.random.normal(247, 52, n_samples),
            'fbs': np.random.choice([0, 1], n_samples),
            'restecg': np.random.choice([0, 1, 2], n_samples),
            'thalach': np.random.normal(150, 23, n_samples),
            'exang': np.random.choice([0, 1], n_samples),
            'oldpeak': np.random.exponential(1, n_samples),
            'slope': np.random.choice([0, 1, 2], n_samples),
            'ca': np.random.choice([0, 1, 2, 3, 4], n_samples),
            'thal': np.random.choice([0, 1, 2, 3], n_samples)
        }
        
        # Crear target basado en características
        target = (
            (data['age'] > 60).astype(int) +
            (data['sex'] == 1).astype(int) +
            (data['cp'] > 1).astype(int) +
            (data['trestbps'] > 140).astype(int) +
            (data['chol'] > 240).astype(int) +
            (data['oldpeak'] > 1).astype(int)
        ) > 2
        
        data['target'] = target.astype(int)
        
        df = pd.DataFrame(data)
        logger.info(f"Datos de muestra creados: {df.shape}")
        
        return df

