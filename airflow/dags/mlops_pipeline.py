"""
DAG para el pipeline de MLOps
Este DAG implementa un pipeline completo de machine learning que incluye:
1. Extracción de datos
2. Preprocesamiento
3. Entrenamiento del modelo
4. Evaluación
5. Registro en MLflow
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.models import Variable
from airflow.hooks.base import BaseHook
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn
import boto3
import os
import sys
import logging

# Agregar el directorio scripts al path
sys.path.append('/opt/airflow/scripts')
from model_utils import HeartDiseaseModel

# Configuración del DAG
default_args = {
    'owner': 'mlops_team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'mlops_pipeline',
    default_args=default_args,
    description='Pipeline completo de MLOps para clasificación de enfermedades cardíacas',
    schedule=timedelta(days=1),
    catchup=False,
    tags=['mlops', 'machine_learning', 'classification']
)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_mlflow():
    """Configurar MLflow"""
    mlflow_tracking_uri = Variable.get("mlflow_tracking_uri", default_var="http://mlflow:5000")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    experiment_name = Variable.get("mlflow_experiment_name", default_var="mlops_experiment")
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
        logger.info(f"Created experiment: {experiment_name} with ID: {experiment_id}")
    except mlflow.exceptions.MlflowException:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        logger.info(f"Using existing experiment: {experiment_name} with ID: {experiment_id}")
    
    mlflow.set_experiment(experiment_name)
    return experiment_id

def extract_data(**context):
    """Extraer datos del dataset de enfermedades cardíacas"""
    logger.info("Extrayendo datos...")
    
    # Crear instancia del modelo
    model = HeartDiseaseModel(random_state=42)
    
    # Intentar cargar datos reales primero
    local_data_path = "/opt/airflow/data/heart.csv"
    if os.path.exists(local_data_path):
        logger.info("Usando datos reales del dataset de heart disease")
        data = model.load_data(local_data_path)
    else:
        logger.info("Datos reales no encontrados, creando datos de muestra")
        data = model.create_sample_data()
    
    # Guardar datos en S3 (MinIO)
    try:
        s3_client = boto3.client(
            's3',
            endpoint_url=os.getenv("AWS_ENDPOINT_URL_S3", "http://s3:9000"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "minio"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "minio123")
        )
        
        bucket_name = Variable.get("s3_bucket_name", default_var="data")
        
        # Convertir a CSV y subir
        csv_data = data.to_csv(index=False)
        s3_client.put_object(
            Bucket=bucket_name,
            Key="heart_disease.csv",
            Body=csv_data
        )
        
        logger.info(f"Datos guardados en S3 bucket: {bucket_name}")
        
    except Exception as e:
        logger.error(f"Error guardando datos en S3: {str(e)}")
        # Guardar localmente como fallback
        data.to_csv("/tmp/heart_disease.csv", index=False)
        logger.info("Datos guardados localmente como fallback")
    
    return data.shape[0]

def preprocess_data(**context):
    """Preprocesar los datos"""
    logger.info("Preprocesando datos...")
    
    # Crear instancia del modelo
    model = HeartDiseaseModel(random_state=42)
    
    try:
        # Cargar datos desde S3
        s3_client = boto3.client(
            's3',
            endpoint_url=os.getenv("AWS_ENDPOINT_URL_S3", "http://s3:9000"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "minio"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "minio123")
        )
        
        bucket_name = Variable.get("s3_bucket_name", default_var="data")
        response = s3_client.get_object(Bucket=bucket_name, Key="heart_disease.csv")
        data = pd.read_csv(response['Body'])
        
    except Exception as e:
        logger.error(f"Error cargando datos desde S3: {str(e)}")
        # Cargar desde archivo local como fallback
        data = pd.read_csv("/tmp/heart_disease.csv")
    
    # Usar el método de preprocesamiento del modelo
    X_train, X_test, y_train, y_test = model.preprocess_data(data)
    
    # Guardar datos procesados en S3
    try:
        bucket_name = Variable.get("s3_bucket_name", default_var="data")
        
        # Guardar datos de entrenamiento
        X_train.to_csv("/tmp/X_train.csv", index=False)
        y_train.to_csv("/tmp/y_train.csv", index=False)
        X_test.to_csv("/tmp/X_test.csv", index=False)
        y_test.to_csv("/tmp/y_test.csv", index=False)
        
        # Subir a S3
        for filename in ['X_train.csv', 'y_train.csv', 'X_test.csv', 'y_test.csv']:
            with open(f"/tmp/{filename}", 'rb') as f:
                s3_client.put_object(
                    Bucket=bucket_name,
                    Key=f"processed/{filename}",
                    Body=f.read()
                )
        
        logger.info("Datos procesados guardados en S3")
        
    except Exception as e:
        logger.error(f"Error guardando datos procesados: {str(e)}")
    
    return {
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'features': X_train.shape[1]
    }

def train_model(**context):
    """Entrenar el modelo de machine learning"""
    logger.info("Entrenando modelo...")
    
    # Configurar MLflow
    setup_mlflow()
    
    # Crear instancia del modelo
    model = HeartDiseaseModel(random_state=42)
    
    try:
        # Cargar datos procesados
        s3_client = boto3.client(
            's3',
            endpoint_url=os.getenv("AWS_ENDPOINT_URL_S3", "http://s3:9000"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "minio"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "minio123")
        )
        
        bucket_name = Variable.get("s3_bucket_name", default_var="data")
        
        # Descargar datos
        for filename in ['X_train.csv', 'y_train.csv', 'X_test.csv', 'y_test.csv']:
            response = s3_client.get_object(Bucket=bucket_name, Key=f"processed/{filename}")
            with open(f"/tmp/{filename}", 'wb') as f:
                f.write(response['Body'].read())
        
        X_train = pd.read_csv("/tmp/X_train.csv")
        y_train = pd.read_csv("/tmp/y_train.csv").values.ravel()
        X_test = pd.read_csv("/tmp/X_test.csv")
        y_test = pd.read_csv("/tmp/y_test.csv").values.ravel()
        
    except Exception as e:
        logger.error(f"Error cargando datos procesados: {str(e)}")
        raise
    
    # Iniciar run de MLflow
    with mlflow.start_run(run_name="heart_disease_classification"):
        # Parámetros del modelo
        model_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        }
        
        # Entrenar modelo usando la clase
        model.train_model(X_train, y_train, **model_params)
        
        # Evaluar modelo
        metrics = model.evaluate_model(X_test, y_test)
        
        # Log parámetros y métricas
        mlflow.log_params({
            **model_params,
            "train_samples": len(X_train),
            "test_samples": len(X_test)
        })
        
        mlflow.log_metrics(metrics)
        
        # Log modelo
        model_name = Variable.get("model_name", default_var="heart_disease_classifier")
        mlflow.sklearn.log_model(
            model.model,
            "model",
            registered_model_name=model_name
        )
        
        logger.info(f"Modelo entrenado con accuracy: {metrics['accuracy']:.4f}")
        
        return {
            "accuracy": metrics['accuracy'],
            "model_name": model_name,
            "metrics": metrics
        }

def evaluate_model(**context):
    """Evaluar el modelo entrenado"""
    logger.info("Evaluando modelo...")
    
    # Obtener resultados del entrenamiento
    train_results = context['task_instance'].xcom_pull(task_ids='train_model')
    
    logger.info(f"Accuracy del modelo: {train_results['accuracy']:.4f}")
    logger.info(f"Modelo registrado como: {train_results['model_name']}")
    
    return train_results

# Definir tareas del DAG
setup_mlflow_task = PythonOperator(
    task_id='setup_mlflow',
    python_callable=setup_mlflow,
    dag=dag
)

extract_data_task = PythonOperator(
    task_id='extract_data',
    python_callable=extract_data,
    dag=dag
)

preprocess_data_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag
)

train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag
)

evaluate_model_task = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    dag=dag
)

# Definir dependencias
setup_mlflow_task >> extract_data_task >> preprocess_data_task >> train_model_task >> evaluate_model_task
