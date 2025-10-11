import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import logging
import os

from abc import ABC, abstractmethod
from BaseModel import BaseModel

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DummyModel(BaseModel):
    def train_model(self, X_train, y_train, **model_params):
        pass