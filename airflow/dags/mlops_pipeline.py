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
from airflow.models import Variable
import os
import sys
import logging
from typing import Any, Dict

# Agregar el directorio scripts al path
sys.path.append('/opt/airflow/scripts')

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


def create_s3_client():
    """Crear un cliente S3 usando las credenciales configuradas."""
    import boto3

    return boto3.client(
        's3',
        endpoint_url=os.getenv("AWS_ENDPOINT_URL_S3", "http://s3:9000"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "minio"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "minio123")
    )


def create_model(random_state=42):
    """Instanciar el modelo sin cargarlo durante el parseo del DAG."""
    from model_utils import HeartDiseaseModel

    return HeartDiseaseModel(random_state=random_state)

def setup_mlflow():
    """Configurar MLflow"""
    import mlflow
    from mlflow import exceptions as mlflow_exceptions

    mlflow_tracking_uri = os.getenv("AIRFLOW_VAR_MLFLOW_TRACKING_URI")
    if mlflow_tracking_uri is None:
        mlflow_tracking_uri = Variable.get("mlflow_tracking_uri", default_var="http://mlflow:5000")
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    experiment_name = os.getenv("AIRFLOW_VAR_MLFLOW_EXPERIMENT_NAME")
    if experiment_name is None:
        experiment_name = Variable.get("mlflow_experiment_name", default_var="mlops_experiment")
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
        logger.info(f"Created experiment: {experiment_name} with ID: {experiment_id}")
    except mlflow_exceptions.MlflowException:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment:
            experiment_id = experiment.experiment_id
            logger.info(f"Using existing experiment: {experiment_name} with ID: {experiment_id}")
        else:
            logger.exception("Failed to retrieve or create MLflow experiment")
            raise

    mlflow.set_experiment(experiment_name)
    return experiment_name

def extract_data(**context):
    """Extraer datos del dataset de enfermedades cardíacas"""
    logger.info("Extrayendo datos...")
    
    # Crear instancia del modelo
    model = create_model(random_state=42)
    
    # Intentar cargar datos reales primero
    local_data_path = "/opt/airflow/data/heart.csv"
    if os.path.exists(local_data_path):
        logger.info("Usando datos reales del dataset de heart disease")
        data = model.load_data(local_data_path)
    else:
        logger.info("Datos reales no encontrados, creando datos de muestra")
        data = model.create_sample_data()
    
    # Guardar datos en S3 (MinIO)
    bucket_name = os.getenv("AIRFLOW_VAR_S3_BUCKET_NAME")
    if bucket_name is None:
        bucket_name = Variable.get("s3_bucket_name", default_var="data")

    try:
        import pandas as pd

        s3_client = create_s3_client()

        # Convertir a CSV y subir
        csv_data = data.to_csv(index=False)
        s3_client.put_object(Bucket=bucket_name, Key="heart_disease.csv", Body=csv_data)
        logger.info("Datos guardados en S3 bucket: %s", bucket_name)
    except Exception:
        logger.exception("Error guardando datos en S3, usando fallback local")
        data.to_csv("/tmp/heart_disease.csv", index=False)
        logger.info("Datos guardados localmente como fallback")
    
    return data.shape[0]

def preprocess_data(**context):
    """Preprocesar los datos"""
    logger.info("Preprocesando datos...")

    import pandas as pd

    # Crear instancia del modelo
    model = create_model(random_state=42)

    bucket_name = os.getenv("AIRFLOW_VAR_S3_BUCKET_NAME")
    if bucket_name is None:
        bucket_name = Variable.get("s3_bucket_name", default_var="data")
    s3_client = None

    try:
        s3_client = create_s3_client()
        response = s3_client.get_object(Bucket=bucket_name, Key="heart_disease.csv")
        data = pd.read_csv(response['Body'])
        logger.info("Datos cargados desde S3")
    except Exception:
        logger.exception("Error cargando datos desde S3, usando fallback local")
        data = pd.read_csv("/tmp/heart_disease.csv")

    # Usar el método de preprocesamiento del modelo
    X_train, X_test, y_train, y_test = model.preprocess_data(data)

    # Guardar datos procesados localmente
    X_train_path = "/tmp/X_train.csv"
    y_train_path = "/tmp/y_train.csv"
    X_test_path = "/tmp/X_test.csv"
    y_test_path = "/tmp/y_test.csv"

    X_train.to_csv(X_train_path, index=False)
    y_train.to_csv(y_train_path, index=False)
    X_test.to_csv(X_test_path, index=False)
    y_test.to_csv(y_test_path, index=False)

    # Intentar subir datos procesados a S3
    if s3_client is None:
        try:
            s3_client = create_s3_client()
        except Exception:
            logger.warning("No se pudo crear el cliente S3 para subir datos procesados")
            s3_client = None

    if s3_client is not None:
        try:
            for filename in ['X_train.csv', 'y_train.csv', 'X_test.csv', 'y_test.csv']:
                with open(f"/tmp/{filename}", 'rb') as file_handle:
                    s3_client.put_object(
                        Bucket=bucket_name,
                        Key=f"processed/{filename}",
                        Body=file_handle.read()
                    )

            logger.info("Datos procesados guardados en S3")
        except Exception as error:
            logger.error(f"Error guardando datos procesados en S3: {error}")

    return {
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'features': X_train.shape[1]
    }


def _load_processed_data():
    """Cargar datos procesados desde S3 o fallback local."""
    import pandas as pd

    bucket_name = os.getenv("AIRFLOW_VAR_S3_BUCKET_NAME")
    if bucket_name is None:
        bucket_name = Variable.get("s3_bucket_name", default_var="data")

    files = ['X_train.csv', 'y_train.csv', 'X_test.csv', 'y_test.csv']
    try:
        s3_client = create_s3_client()
        for filename in files:
            response = s3_client.get_object(Bucket=bucket_name, Key=f"processed/{filename}")
            with open(f"/tmp/{filename}", 'wb') as file_handle:
                file_handle.write(response['Body'].read())
        logger.info("Datos procesados descargados desde S3")
    except Exception:
        logger.exception("No se pudieron descargar los datos procesados desde S3, usando archivos locales")

    try:
        X_train = pd.read_csv("/tmp/X_train.csv")
        y_train = pd.read_csv("/tmp/y_train.csv").values.ravel()
        X_test = pd.read_csv("/tmp/X_test.csv")
        y_test = pd.read_csv("/tmp/y_test.csv").values.ravel()
    except FileNotFoundError as error:
        raise RuntimeError("Los datos procesados no están disponibles ni en S3 ni localmente") from error

    return X_train, y_train, X_test, y_test

def _train_and_log_model(params, run_name, context):
    """Entrenar el modelo con parámetros dados y registrar resultados en MLflow."""
    setup_mlflow()
    import mlflow
    import mlflow.sklearn

    X_train, y_train, X_test, y_test = _load_processed_data()
    model = create_model(random_state=params.get("random_state", 42))

    run_name = run_name or "model_training"
    logger.info("Iniciando entrenamiento para variante '%s'", run_name)

    with mlflow.start_run(run_name=run_name):
        model.train_model(X_train, y_train, **params)
        metrics = model.evaluate_model(X_test, y_test)

        mlflow.log_params({**params, "train_samples": len(X_train), "test_samples": len(X_test)})
        mlflow.log_metrics(metrics)

        model_name = os.getenv("AIRFLOW_VAR_MODEL_NAME")
        if model_name is None:
            model_name = Variable.get("model_name", default_var="heart_disease_classifier")
        mlflow.sklearn.log_model(model.model, "model", registered_model_name=model_name)

        ts_suffix = context.get('ts_nodash')
        if ts_suffix is None and 'logical_date' in context:
            ts_suffix = context['logical_date'].strftime("%Y%m%dT%H%M%S")
        ts_suffix = ts_suffix or datetime.utcnow().strftime("%Y%m%dT%H%M%S")

        safe_run_name = run_name.replace(' ', '_')
        local_model_path = f"/opt/airflow/models/{model_name}_{safe_run_name}_{ts_suffix}.pkl"

        try:
            model.save_model(local_model_path)
            logger.info("Modelo guardado localmente en: %s", local_model_path)
        except Exception:
            logger.exception("Error guardando modelo localmente")

        logger.info(
            "Variante '%s' entrenada con accuracy: %.4f",
            run_name,
            metrics.get("accuracy", 0.0)
        )

        return {
            "accuracy": metrics.get("accuracy"),
            "model_name": model_name,
            "metrics": metrics,
            "local_model_path": local_model_path,
            "params": params,
            "run_name": run_name,
        }


def run_single_training(params, run_name, **context):
    return _train_and_log_model(params, run_name, context)


def train_model(**context):
    """Entrenar el modelo de machine learning base"""
    logger.info("Entrenando modelo base...")

    default_params = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42
    }

    return _train_and_log_model(default_params, "heart_disease_classification", context)

def evaluate_model(**context):
    """Evaluar el modelo entrenado"""
    logger.info("Evaluando modelo...")
    
    # Obtener resultados del entrenamiento
    train_results = context['task_instance'].xcom_pull(task_ids='train_model')

    if not train_results:
        logger.warning("No se encontraron resultados de entrenamiento para evaluar")
        return {}
    
    accuracy = train_results.get('accuracy')
    if accuracy is not None:
        logger.info(f"Accuracy del modelo: {accuracy:.4f}")
    else:
        logger.info("Accuracy del modelo no disponible")

    logger.info(f"Modelo registrado como: {train_results['model_name']}")
    
    # Mostrar información del modelo guardado localmente
    if 'local_model_path' in train_results:
        logger.info(f"Modelo guardado localmente en: {train_results['local_model_path']}")
        
        # Verificar que el archivo existe
        if os.path.exists(train_results['local_model_path']):
            file_size = os.path.getsize(train_results['local_model_path'])
            logger.info(f"Tamaño del archivo del modelo: {file_size} bytes")
        else:
            logger.warning("El archivo del modelo local no se encontró")
    
    return train_results

def cleanup_old_models(**context):
    """Limpiar modelos antiguos (opcional)"""
    logger.info("Limpiando modelos antiguos...")
    
    try:
        models_dir = "/opt/airflow/models"
        if os.path.exists(models_dir):
            # Listar archivos de modelos
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
            logger.info(f"Modelos encontrados: {len(model_files)}")
            
            # Mantener solo los últimos 5 modelos
            if len(model_files) > 5:
                model_files.sort(reverse=True)  
                files_to_delete = model_files[5:]  
                
                for file_to_delete in files_to_delete:
                    file_path = os.path.join(models_dir, file_to_delete)
                    os.remove(file_path)
                    logger.info(f"Modelo antiguo eliminado: {file_to_delete}")
            else:
                logger.info("No hay modelos antiguos para eliminar")
        else:
            logger.warning("Directorio de modelos no encontrado")
            
    except Exception as e:
        logger.error(f"Error en limpieza de modelos: {str(e)}")
    
    return "Limpieza completada"


def compare_models(**context):
    """Comparar resultados de entrenamiento y registrar el mejor modelo."""
    logger.info("Comparando variantes de modelos...")

    task_instance = context['task_instance']
    task_ids = ['train_model', 'train_rf_depth10', 'train_rf_depth6']
    results = []

    for task_id in task_ids:
        result = task_instance.xcom_pull(task_ids=task_id)
        if result:
            result_copy = result.copy()
            result_copy['source_task'] = task_id
            results.append(result_copy)
            logger.info(
                "Resultados %s -> accuracy: %s",
                result_copy.get('run_name', task_id),
                result_copy.get('accuracy')
            )
        else:
            logger.warning("No se encontraron resultados para la tarea %s", task_id)

    if not results:
        logger.error("No hay resultados para comparar")
        raise ValueError("No se encontraron resultados de entrenamiento para comparar")

    def accuracy_key(item):
        accuracy = item.get('accuracy')
        return accuracy if accuracy is not None else float('-inf')

    best_model = max(results, key=accuracy_key)

    logger.info(
        "Mejor variante: %s (tarea %s) con accuracy %.4f",
        best_model.get('run_name', best_model['source_task']),
        best_model['source_task'],
        best_model.get('accuracy', 0.0)
    )

    return best_model

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
    pool="ml_training_pool",
    pool_slots=1,
    dag=dag
)

evaluate_model_task = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    dag=dag
)

cleanup_models_task = PythonOperator(
    task_id='cleanup_old_models',
    python_callable=cleanup_old_models,
    dag=dag
)

compare_models_task = PythonOperator(
    task_id='compare_models',
    python_callable=compare_models,
    dag=dag
)

rf_depth10_task = PythonOperator(
    task_id="train_rf_depth10",
    python_callable=run_single_training,
    op_kwargs={
        "params": {"n_estimators": 100, "max_depth": 10, "min_samples_split": 5,
                   "min_samples_leaf": 2, "random_state": 42},
        "run_name": "rf_depth10",
    },
    pool="ml_training_pool",
    pool_slots=1,
    dag=dag,
)

rf_depth6_task = PythonOperator(
    task_id="train_rf_depth6",
    python_callable=run_single_training,
    op_kwargs={
        "params": {"n_estimators": 200, "max_depth": 6, "min_samples_split": 4,
                   "min_samples_leaf": 1, "random_state": 123},
        "run_name": "rf_depth6",
    },
    pool="ml_training_pool",
    pool_slots=1,
    dag=dag,
)


# Definir dependencias
setup_mlflow_task >> extract_data_task >> preprocess_data_task

preprocess_data_task >> train_model_task >> evaluate_model_task

# Ejecutar variantes en paralelo una vez que el modelo base fue evaluado
evaluate_model_task >> [rf_depth10_task, rf_depth6_task]

[rf_depth10_task, rf_depth6_task] >> compare_models_task >> cleanup_models_task
