import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import make_classification
import os
import logging
from typing import Tuple, List, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "hyperparameter_tuning_experiment_optimized"
REGISTERED_MODEL_NAME = "heart_disease_classifier_optimized"
RANDOM_STATE = 42
FEATURE_NAMES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

def setup_mlflow() -> str:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
        logger.info(f"Created experiment: {EXPERIMENT_NAME} with ID: {experiment_id}")
    except mlflow.exceptions.MlflowException:
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        experiment_id = experiment.experiment_id
        logger.info(f"Using existing experiment: {EXPERIMENT_NAME} with ID: {experiment_id}")
    
    mlflow.set_experiment(EXPERIMENT_NAME)
    return experiment_id

def create_synthetic_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    logger.info("Creating synthetic data...")
    X, y = make_classification(
        n_samples=1000,
        n_features=13,
        n_informative=10,
        n_redundant=3,
        n_classes=2,
        random_state=RANDOM_STATE
    )
    
    X_df = pd.DataFrame(X, columns=FEATURE_NAMES)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def log_model_to_mlflow(
    model: RandomForestClassifier, 
    X_test: pd.DataFrame, 
    y_test: pd.Series,
    run_name: str
) -> Dict[str, Any]:
    with mlflow.start_run(run_name=run_name) as run:
        y_pred = model.predict(X_test)
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred)
        }
        
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME
        )
        
        sample_data = X_test.head(5)
        mlflow.log_table(sample_data, "sample_predictions.json")
        
        logger.info(f"Run '{run_name}' logged with F1 score: {metrics['f1_score']:.4f}")
        
        return {**metrics, "model": model, "run_id": run.info.run_id}

def perform_hyperparameter_tuning() -> Dict[str, Any]:
    logger.info("Starting hyperparameter tuning...")
    setup_mlflow()
    X_train, X_test, y_train, y_test = create_synthetic_data()
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rfc = RandomForestClassifier(random_state=RANDOM_STATE)
    grid_search = GridSearchCV(
        estimator=rfc,
        param_grid=param_grid,
        cv=3,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    logger.info(f"Best parameters found: {grid_search.best_params_}")
    logger.info(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")
    
    best_model = grid_search.best_estimator_
    
    best_model_results = log_model_to_mlflow(
        model=best_model,
        X_test=X_test,
        y_test=y_test,
        run_name="best_model_from_gridsearch"
    )
    
    return best_model_results

def main():
    try:
        best_model_info = perform_hyperparameter_tuning()
        
        logger.info("\n=== EXPERIMENT SUMMARY ===")
        logger.info(f"Best model logged in run ID: {best_model_info['run_id']}")
        logger.info(f"  Accuracy: {best_model_info['accuracy']:.4f}")
        logger.info(f"  Precision: {best_model_info['precision']:.4f}")
        logger.info(f"  Recall: {best_model_info['recall']:.4f}")
        logger.info(f"  F1 Score: {best_model_info['f1_score']:.4f}")
        
    except Exception as e:
        logger.error(f"An error occurred during the experiment: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()