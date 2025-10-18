import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Configuración global de la aplicación FastAPI.
    Lee las variables del archivo .env y define rutas absolutas.
    """
    # Permite leer automáticamente desde el archivo .env en la raíz del proyecto
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"  
    )
    
        # Configuración FASTAPI 
    fastapi_port: int = int(os.getenv("FASTAPI_PORT", 8800))

    # Ruta del proyecto 
    BASE_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    model_path: str = os.getenv("MODEL_PATH", os.path.join(BASE_DIR, "models", "model.pkl"))
    model_version: str = os.getenv("MODEL_VERSION", "3")

    # Configuración MLflow
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow_s3_endpoint_url: str = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
    mlflow_bucket_name: str = os.getenv("MLFLOW_BUCKET_NAME", "mlflow")

    # Configuración MinIO 
    minio_endpoint: str = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
    minio_access_key: str = os.getenv("MINIO_ACCESS_KEY", "minio")
    minio_secret_key: str = os.getenv("MINIO_SECRET_ACCESS_KEY", "minio123")
    data_repo_bucket_name: str = os.getenv("DATA_REPO_BUCKET_NAME", "data")

    # Configuración PostgreSQL 
    pg_user: str = os.getenv("PG_USER", "airflow")
    pg_password: str = os.getenv("PG_PASSWORD", "airflow")
    pg_database: str = os.getenv("PG_DATABASE", "airflow")
    pg_port: int = int(os.getenv("PG_PORT", 5432))

    # Configuración Airflow
    airflow_port: int = int(os.getenv("AIRFLOW_PORT", 8080))
    airflow_image_name: str = os.getenv("AIRFLOW_IMAGE_NAME", "extending_airflow:latest")

    # Configuración AWS / S3
    aws_access_key_id: str = os.getenv("AWS_ACCESS_KEY_ID", "minio")
    aws_secret_access_key: str = os.getenv("AWS_SECRET_ACCESS_KEY", "minio123")
    aws_endpoint_url_s3: str = os.getenv("AWS_ENDPOINT_URL_S3", "http://localhost:9000")

settings = Settings()


# visualización de las configuraciones cargadas
if __name__ == "__main__":
    print("BASE_DIR:", settings.BASE_DIR)
    print("MODEL_PATH:", settings.model_path)
    print("MLFLOW_TRACKING_URI:", settings.mlflow_tracking_uri)
    print("MINIO_ENDPOINT:", settings.minio_endpoint)
