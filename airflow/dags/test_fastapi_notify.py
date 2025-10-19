from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import requests
import logging

def test_notify_fastapi():
    """Prueba: notifica a FastAPI que recargue el modelo."""
    logger = logging.getLogger(__name__)
    fastapi_url = "http://localhost:8800/model/reload"
    logger.info(f"Enviando POST a {fastapi_url}")

    try:
        response = requests.post(fastapi_url)
        if response.status_code == 200:
            logger.info(f"FastAPI respondió correctamente: {response.json()}")
        else:
            logger.warning(f"Error al llamar a FastAPI: {response.status_code} - {response.text}")
    except Exception as e:
        logger.error(f"No se pudo conectar con FastAPI: {e}")

# Configuración del DAG
default_args = {
    "owner": "test_user",
    "depends_on_past": False,
    "start_date": datetime(2025, 1, 1),
    "retries": 0,
}

dag = DAG(
    "test_fastapi_notify",
    default_args=default_args,
    schedule_interval=None,  # Manual
    catchup=False,
    description="DAG de prueba para notificar recarga de modelo a FastAPI",
)

test_task = PythonOperator(
    task_id="notify_fastapi_reload",
    python_callable=test_notify_fastapi,
    dag=dag,
)
