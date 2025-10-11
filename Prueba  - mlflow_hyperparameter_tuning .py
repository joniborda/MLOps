"""
Ejemplo de experimento MLflow con búsqueda de hiperparámetros
Este script demuestra cómo usar MLflow para experimentar con diferentes hiperparámetros
y registrar los mejores modelos.
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import make_classification
import os
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_mlflow():
    """Configurar MLflow"""
    # Configurar tracking URI
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    # Crear o usar experimento existente
    experiment_name = "hyperparameter_tuning_experiment"
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
        logger.info(f"Created experiment: {experiment_name} with ID: {experiment_id}")
    except mlflow.exceptions.MlflowException:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        logger.info(f"Using existing experiment: {experiment_name} with ID: {experiment_id}")
    
    mlflow.set_experiment(experiment_name)
    return experiment_id

def create_synthetic_data():
    """Crear datos sintéticos para el experimento"""
    logger.info("Creando datos sintéticos...")
    
    # Crear dataset sintético
    X, y = make_classification(
        n_samples=1000,
        n_features=13,
        n_informative=10,
        n_redundant=3,
        n_classes=2,
        random_state=42
    )
    
    # Crear nombres de características
    feature_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
    ]
    
    # Crear DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Dividir en train y test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, df

def train_and_log_model(X_train, X_test, y_train, y_test, params, run_name):
    """Entrenar modelo y registrar en MLflow"""
    
    with mlflow.start_run(run_name=run_name):
        # Crear modelo con parámetros específicos
        model = RandomForestClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'],
            random_state=42
        )
        
        # Entrenar modelo
        model.fit(X_train, y_train)
        
        # Hacer predicciones
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Log parámetros
        mlflow.log_params(params)
        
        # Log métricas
        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })
        
        # Log modelo
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name="heart_disease_classifier"
        )
        
        # Log datos de ejemplo
        sample_data = pd.DataFrame(X_test[:5], columns=[
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ])
        mlflow.log_table(sample_data, "sample_predictions.json")
        
        logger.info(f"Run {run_name}: Accuracy={accuracy:.4f}, F1={f1:.4f}")
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "model": model
        }

def hyperparameter_tuning():
    """Realizar búsqueda de hiperparámetros"""
    logger.info("Iniciando búsqueda de hiperparámetros...")
    
    # Configurar MLflow
    setup_mlflow()
    
    # Crear datos
    X_train, X_test, y_train, y_test, df = create_synthetic_data()
    
    # Definir grid de hiperparámetros
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Realizar búsqueda en grid
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Obtener mejores parámetros
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    logger.info(f"Mejores parámetros: {best_params}")
    logger.info(f"Mejor score CV: {best_score:.4f}")
    
    # Entrenar y registrar el mejor modelo
    best_model_results = train_and_log_model(
        X_train, X_test, y_train, y_test, best_params, "best_model"
    )
    
    # Entrenar algunos modelos adicionales para comparación
    comparison_params = [
        {"n_estimators": 50, "max_depth": 5, "min_samples_split": 2, "min_samples_leaf": 1},
        {"n_estimators": 100, "max_depth": 10, "min_samples_split": 5, "min_samples_leaf": 2},
        {"n_estimators": 200, "max_depth": 15, "min_samples_split": 10, "min_samples_leaf": 4}
    ]
    
    results = [best_model_results]
    
    for i, params in enumerate(comparison_params):
        run_name = f"comparison_model_{i+1}"
        model_results = train_and_log_model(
            X_train, X_test, y_train, y_test, params, run_name
        )
        results.append(model_results)
    
    # Encontrar el mejor modelo basado en F1 score
    best_model_idx = max(range(len(results)), key=lambda i: results[i]['f1_score'])
    best_model = results[best_model_idx]
    
    logger.info(f"Mejor modelo: {best_model_idx} con F1 score: {best_model['f1_score']:.4f}")
    
    return best_model, results

def main():
    """Función principal"""
    try:
        # Realizar búsqueda de hiperparámetros
        best_model, all_results = hyperparameter_tuning()
        
        # Mostrar resumen de resultados
        logger.info("\n=== RESUMEN DE RESULTADOS ===")
        for i, result in enumerate(all_results):
            logger.info(f"Modelo {i+1}: Accuracy={result['accuracy']:.4f}, "
                       f"Precision={result['precision']:.4f}, "
                       f"Recall={result['recall']:.4f}, "
                       f"F1={result['f1_score']:.4f}")
        
        logger.info(f"\nMejor modelo encontrado con F1 score: {best_model['f1_score']:.4f}")
        
    except Exception as e:
        logger.error(f"Error en el experimento: {str(e)}")
        raise

if __name__ == "__main__":
    main()
