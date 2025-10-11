"""
Este modulo se encarga de crear instancias de modelos
"""

from HeartDiseaseRandomForestModel import HeartDiseaseRandomForestModel
from HeartDiseaseLogisticRegressionModel import HeartDiseaseLogisticRegressionModel
from HeartDiseaseSVMModel import HeartDiseaseSVMModel

def CreateModel(model_type: str, random_state=42):
    model_classes = {
        'random_forest': HeartDiseaseRandomForestModel,
        'logistic_regression': HeartDiseaseLogisticRegressionModel,
        'svm': HeartDiseaseSVMModel
    }

    if model_type not in model_classes:
        raise ValueError(f"Tipo de modelo no soportado: {model_type}")

    return model_classes[model_type](random_state=random_state)
