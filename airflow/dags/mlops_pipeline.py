"""
DAG para el pipeline de MLOps
Este DAG implementa un pipeline completo de machine learning que incluye:
1. Extracción de datos
2. Preprocesamiento
3. Entrenamiento del modelo
4. Evaluación
5. Registro en MLflow
"""

import yaml
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
import os
import sys
import logging
from typing import Any, Dict
import copy

# Agregar el directorio scripts al path
sys.path.append('/opt/airflow/scripts')
sys.path.append('/opt/airflow/scripts/models')

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

PARAM_KEYS_EXCLUDED_FROM_SCALING = {'random_state'}


def create_s3_client():
    """Crear un cliente S3 usando las credenciales configuradas."""
    import boto3

    return boto3.client(
        's3',
        endpoint_url=os.getenv("AWS_ENDPOINT_URL_S3", "http://s3:9000"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "minio"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "minio123")
    )


# def create_model(random_state=42):
#     """Instanciar el modelo sin cargarlo durante el parseo del DAG."""
#     from DummyModel import DummyModel

#     return DummyModel(random_state=random_state)

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
    from create_sample_data import DataGenerator
    from CreateModel import CreateModel
    logger.info("Extrayendo datos...")
    
    # Crear instancia del modelo
    model = CreateModel(model_type='dummy')
    #model = create_model(random_state=42)
    
    # Intentar cargar datos reales primero
    local_data_path = "/opt/airflow/data/heart.csv"
    if os.path.exists(local_data_path):
        logger.info("Usando datos reales del dataset de heart disease")
        data = model.load_data(local_data_path)
    else:
        logger.info("Datos reales no encontrados, creando datos de muestra")
        data = DataGenerator.create_sample_data(n_samples=1000, seed=10)
    
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
    from CreateModel import CreateModel
    logger.info("Preprocesando datos...")

    import pandas as pd

    # Crear instancia del modelo
    model = CreateModel(model_type='dummy')
    #model = create_model(random_state=42)

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

def _train_and_log_model(model_type = 'random_forest', model_params=None, run_name=None, context=None):
    """Entrenar el modelo con parámetros dados y registrar resultados en MLflow."""
    from CreateModel import CreateModel
    
    setup_mlflow()
    import mlflow
    import mlflow.sklearn
    
    logger.info(f"Entrenando modelo {model_type}...")
    
    X_train, y_train, X_test, y_test = _load_processed_data()
    model = CreateModel(model_type=model_type, random_state=42)
    #model = create_model(random_state=params.get("random_state", 42))

    run_name = run_name or "model_training"
    logger.info(f"Iniciando entrenamiento de modelo {model_type}, variante {run_name}")

    with mlflow.start_run(run_name=run_name):
        model.train_model(X_train, y_train, **model_params)
        metrics = model.evaluate_model(X_test, y_test)

        mlflow.log_params({**model_params, "train_samples": len(X_train), "test_samples": len(X_test)})
        mlflow.log_metrics(metrics)

        logger.info("Obteniendo nombre del modelo para registro en MLflow...")
        model_name = os.getenv("AIRFLOW_VAR_MODEL_NAME")

        if model_name is None:
            logger.info(f"Nombre del modelo: {model_type}_classifier")

            model_name = Variable.get("model_name", default_var=f"{model_type}_classifier")
        else:
            logger.info(f'Nombre del modelo: {model_name}')

        logger.info("Obteniendo ejemplo de entrada para registro en MLflow...")
        input_example = X_train.iloc[:5] if hasattr(X_train, "iloc") else X_train[:5]
        input_example = input_example.astype({col: "float64" for col in input_example.select_dtypes("int").columns}) # Se pasan int a float64 para evitar problemas de tipo en MLflow    
        mlflow.sklearn.log_model(model.model,
                                 "model", 
                                 registered_model_name=model_name,
                                 input_example=input_example
                                 )

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
            "Modelo '%s', variante '%s', entrenada con accuracy: %.4f",
            model_type,
            run_name,
            metrics.get("accuracy", 0.0)
        )

        return {
            "model_type": model_type,
            "accuracy": metrics.get("accuracy"),
            "model_name": model_name,
            "metrics": metrics,
            "local_model_path": local_model_path,
            #"params": params,
            "model_params": model_params,
            "run_name": run_name,
        }


def run_single_training(params, run_name, **context):
    return _train_and_log_model(params, run_name, context)


def train_model(model_type = 'random_forest', **context):
    """Entrenar el modelo de machine learning"""
    logger.info(f"Entrenando modelo {model_type}...")

    try:
        with open("/opt/airflow/config/model_params.yaml", "r") as f:
            all_params = yaml.safe_load(f)
            model_params = all_params.get(model_type, {})
            if not model_params:
                raise ValueError(f"No se encontraron parámetros para el modelo {model_type}")
    except Exception as e:
        logger.error(f"Error cargando parámetros del modelo: {str(e)}")
        raise

    return _train_and_log_model(model_type, model_params, "heart_disease_classification", context)

def evaluate_model(**context):
    """Evaluar el modelo entrenado"""
    logger.info("Evaluando modelo...")

    train_task_id = context['params'].get('train_task_id') or context['ti'].task.op_kwargs.get('train_task_id')
    
    if not train_task_id:
        logger.error("No se especificó train_task_id")
        raise ValueError("train_task_id no especificado")
    
    logger.info(f"Buscando resultados de la tarea: {train_task_id}")
    
    # Obtener resultados del entrenamiento específico
    train_results = context['ti'].xcom_pull(task_ids=train_task_id, key='return_value')

    if train_results is None:
        logger.error(f"No se encontraron resultados XCom para la tarea: {train_task_id}")
    
    logger.info(f"Accuracy del modelo: {train_results['accuracy']:.4f}")
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

def select_best_model(**context):

    setup_mlflow()
    import mlflow
    import mlflow.sklearn
    from joblib import load

    logger.info("Seleccionando el mejor modelo...")

    # Recuperar resultados de evaluación de cada modelo
    task_ids = ['evaluate_rf', 'evaluate_lr', 'evaluate_svm']
    results = []

    for task_id in task_ids:
        result = context['ti'].xcom_pull(task_ids=task_id)
        if result:
            results.append(result)

    if not results:
        logger.error("No se encontraron resultados de evaluación.")
        raise ValueError("No hay resultados de evaluación para comparar.")

    # Elegir el mejor modelo según accuracy
    best_model = max(results, key=lambda x: x['accuracy'])
    logger.info(f"Mejor modelo: {best_model['model_type']} con accuracy {best_model['accuracy']:.4f}")

    # Guardar como Variable global de Airflow (opcional)
    logger.info(f"Guardando Variable global de Airflow:")
    registered_name = "heart_disease_classifier"
    Variable.set("best_model_type", registered_name)
    Variable.set("best_model_accuracy", str(best_model.get('accuracy', 0)))

    logger.info(f"Mejor modelo registrado como '{registered_name}'")
    logger.info(f"Variables guardadas:")
    logger.info(f"  - best_model_type: {registered_name}")
    logger.info(f"  - best_model_accuracy: {best_model.get('accuracy'):.4f}")

    # Registrar el modelo ganador en MLflow
    try:
        logger.info(f"Registrando modelo ganador en MLflow: {registered_name}")
        model = load(best_model['local_model_path'])

        mlflow.sklearn.log_model(model, 
                                 "model", 
                                 registered_model_name=registered_name)
        
        logger.info(f"Modelo ganador registrado correctamente en MLflow con nombre '{registered_name}'")
    except Exception as e:
        logger.error(f"Error registrando modelo en MLflow: {str(e)}")

    return best_model


def _scale_params_by_factor(params: Dict[str, Any], factor: float) -> Dict[str, Any]:
    scaled_params: Dict[str, Any] = {}
    for key, value in params.items():
        if key in PARAM_KEYS_EXCLUDED_FROM_SCALING or isinstance(value, bool):
            scaled_params[key] = value
            continue

        if isinstance(value, int):
            scaled_value = max(1, int(round(value * factor)))
            scaled_params[key] = scaled_value
        elif isinstance(value, float):
            scaled_value = max(1e-6, float(value * factor))
            scaled_params[key] = scaled_value
        else:
            scaled_params[key] = value

    return scaled_params


def _extract_accuracy(result: Dict[str, Any]) -> float:
    if not result:
        return float("-inf")

    metrics = result.get("metrics") or {}
    accuracy = metrics.get("accuracy")
    if accuracy is None:
        accuracy = result.get("accuracy")
    return accuracy if accuracy is not None else float("-inf")


def tune_best_model(**context):
    task_instance = context['ti']
    best_model_info = task_instance.xcom_pull(task_ids='select_best_model')

    if not best_model_info:
        raise ValueError("No se encontró información del mejor modelo para ajustar")

    model_type = best_model_info.get('model_type')
    if not model_type:
        raise ValueError("El mejor modelo no especifica 'model_type'")

    base_params = copy.deepcopy(best_model_info.get('model_params', {}))
    if not base_params:
        logger.warning("El mejor modelo no tiene parámetros configurados. Se utilizarán valores originales.")

    logger.info("Iniciando ajuste para el modelo %s con parámetros base: %s", model_type, base_params)

    tuning_candidates = [
        {
            "label": "original",
            "params": base_params,
            "result": best_model_info
        }
    ]

    adjustments = [
        ("minus_10pct", 0.9),
        ("plus_10pct", 1.1),
    ]

    base_run_name = best_model_info.get("run_name", f"{model_type}_best")

    for label, factor in adjustments:
        tuned_params = _scale_params_by_factor(base_params, factor)
        if tuned_params == base_params:
            logger.info("La variación %s no modificó los parámetros, se omite.", label)
            continue

        run_name = f"{base_run_name}_{label}"
        logger.info("Entrenando variante '%s' con parámetros: %s", label, tuned_params)
        tuned_result = _train_and_log_model(
            model_type=model_type,
            model_params=tuned_params,
            run_name=run_name,
            context=context
        )
        tuned_result["tuning_label"] = label
        tuning_candidates.append({
            "label": label,
            "params": tuned_params,
            "result": tuned_result
        })

    best_candidate = max(tuning_candidates, key=lambda candidate: _extract_accuracy(candidate["result"]))
    best_accuracy = _extract_accuracy(best_candidate["result"])

    logger.info("Mejor variante tras ajuste: %s con accuracy %.4f", best_candidate["label"], best_accuracy)
    logger.info("Parámetros finales seleccionados: %s", best_candidate["params"])

    Variable.set("best_model_tuning_label", best_candidate["label"])
    Variable.set("best_model_tuned_params", str(best_candidate["params"]))
    Variable.set("best_model_tuned_accuracy", str(best_accuracy))

    return {
        "model_type": model_type,
        "selected_variant": best_candidate["label"],
        "best_params": best_candidate["params"],
        "best_accuracy": best_accuracy,
        "evaluated_variants": [
            {
                "label": candidate["label"],
                "accuracy": _extract_accuracy(candidate["result"]),
                "params": candidate["params"]
            }
            for candidate in tuning_candidates
        ]
    }

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


# Distintos modelos de entrenamiento (en paralelo)
train_rf = PythonOperator(
    task_id="train_random_forest",
    python_callable=train_model,
    op_kwargs={"model_type": "random_forest"},
    dag=dag
)

train_lr = PythonOperator(
    task_id="train_logistic_regression",
    python_callable=train_model,
    op_kwargs={"model_type": "logistic_regression"},
    dag=dag
)

train_svm = PythonOperator(
    task_id="train_svm",
    python_callable=train_model,
    op_kwargs={"model_type": "svm"},
    dag=dag
)

# Evaluar los distintos modelos
evaluate_rf_task = PythonOperator(
    task_id='evaluate_rf',
    python_callable=evaluate_model,
    op_kwargs={'train_task_id': 'train_random_forest'},
    dag=dag
)

evaluate_lr_task = PythonOperator(
    task_id='evaluate_lr',
    python_callable=evaluate_model,
    op_kwargs={'train_task_id': 'train_logistic_regression'},
    dag=dag
)

evaluate_svm_task = PythonOperator(
    task_id='evaluate_svm',
    python_callable=evaluate_model,
    op_kwargs={'train_task_id': 'train_svm'},
    dag=dag
)

# Seleccion del mejor modelo
select_best_model_task = PythonOperator(
    task_id='select_best_model',
    python_callable=select_best_model,
    dag=dag
)

tune_best_model_task = PythonOperator(
    task_id='tune_best_model',
    python_callable=tune_best_model,
    dag=dag
)

cleanup_models_task = PythonOperator(
    task_id='cleanup_old_models',
    python_callable=cleanup_old_models,
    dag=dag
)


# Definir dependencias
setup_mlflow_task >>  extract_data_task >> preprocess_data_task

# Conecta la tarea de preprocesamiento con las tareas de entrenamiento en paralelo
preprocess_data_task >> [train_rf, train_lr, train_svm]

# Conecta cada tarea de entrenamiento con su respectiva tarea de evaluación
train_rf >> evaluate_rf_task
train_lr >> evaluate_lr_task
train_svm >> evaluate_svm_task

[evaluate_rf_task, evaluate_lr_task, evaluate_svm_task] >> select_best_model_task

select_best_model_task >> tune_best_model_task >> cleanup_models_task
