# MLOps - Proyecto Final
### Aprendizaje de Máquina I - CEIA - FIUBA

Este proyecto implementa un ambiente productivo completo de MLOps utilizando Docker y Docker Compose. Incluye todos los servicios necesarios para el ciclo de vida completo de machine learning, desde el entrenamiento hasta el despliegue de modelos.

## Integrantes

1. Sebastian Biagiola
2. Erlin Rey
3. Santiago Casado
5. Daniel Bazán
6. Jonathan Matias Borda

## Servicios Incluidos

- **Apache Airflow**: Orquestación de workflows y pipelines de ML
- **MLflow**: Gestión de experimentos, modelos y artefactos
- **FastAPI**: API REST para servir modelos entrenados
- **MinIO**: Almacenamiento de objetos (simulando S3)
- **PostgreSQL**: Base de datos para Airflow y MLflow
- **Valkey**: Base de datos key-value para Airflow

## Estructura del Proyecto

```
MLOps/
├── airflow/
│   ├── dags/                    # DAGs de Airflow
│   ├── logs/                    # Logs de Airflow
│   ├── plugins/                 # Plugins de Airflow
│   ├── config/                  # Configuración de Airflow
│   └── secrets/                 # Variables y conexiones
├── data/                        # Datasets y datos
│   └── heart.csv               # Dataset de enfermedades cardíacas
├── models/                      # Modelos entrenados guardados localmente
├── scripts/                     # Lógica del modelo y utilidades
│   ├── model_utils.py          # Clase principal del modelo
│   ├── train_model_example.py  # Ejemplo de entrenamiento
│   └── requirements.txt        # Dependencias de los scripts
├── dockerfiles/                 # Dockerfiles para cada servicio
│   ├── airflow/
│   ├── fastapi/
│   ├── mlflow/
│   └── postgres/
├── docker-compose.yaml          # Configuración de servicios
├── env.example                  # Variables de entorno de ejemplo
└── mlflow_hyperparameter_tuning.py  # Script de ejemplo MLflow
```

## Instalación y Configuración

### Prerrequisitos

1. **Docker**: Instala Docker Desktop o Docker Engine
2. **Docker Compose**: Viene incluido con Docker Desktop

### Pasos de Instalación

1. **Clona o descarga este repositorio**

2. **Crea el archivo de variables de entorno**:
   ```bash
   cp env.example .env
   ```

3. **Configura las variables de entorno** (opcional):
   - Edita el archivo `.env` si necesitas cambiar puertos o credenciales
   - En Linux/MacOS, asegúrate de configurar `AIRFLOW_UID` con tu UID de usuario

4. **Levanta todos los servicios**:
   ```bash
   docker compose --profile all up
   ```

5. **Verifica que todos los servicios estén funcionando**:
   ```bash
   docker ps -a
   ```

## Acceso a los Servicios

Una vez que todos los servicios estén funcionando, podrás acceder a:

- **Apache Airflow**: http://localhost:8080
  - Usuario: `airflow`
  - Contraseña: `airflow`
- **MLflow**: http://localhost:5000
- **MinIO Console**: http://localhost:9001
  - Usuario: `minio`
  - Contraseña: `minio123`
- **FastAPI**: http://localhost:8800
- **Documentación de la API**: http://localhost:8800/docs

## Características Implementadas

### 1. Pipeline de Airflow (`mlops_pipeline.py`)

Un DAG completo que incluye:
- **Extracción de datos**: Genera datos sintéticos de enfermedades cardíacas
- **Preprocesamiento**: Limpieza y preparación de datos
- **Entrenamiento**: Modelo Random Forest con MLflow
- **Evaluación**: Métricas de rendimiento
- **Registro**: Modelo guardado en MLflow

### 2. API FastAPI

Servicio REST que incluye:
- **Endpoint de predicción**: `/predict` para hacer predicciones
- **Health check**: `/health` para monitoreo
- **Información del modelo**: `/model/info` para detalles del modelo
- **Recarga de modelo**: `/model/reload` para actualizar el modelo

### 3. Experimentos MLflow

Script de ejemplo (`mlflow_hyperparameter_tuning.py`) que demuestra:
- **Búsqueda de hiperparámetros**: Grid search con Random Forest
- **Registro de experimentos**: Múltiples runs con diferentes parámetros
- **Comparación de modelos**: Métricas detalladas para cada configuración
- **Registro de artefactos**: Modelos y datos de ejemplo

### 4. Almacenamiento S3 (MinIO)

Buckets automáticamente creados:
- `data`: Para datasets y datos procesados
- `mlflow`: Para artefactos de MLflow

## Uso del Sistema

### 1. Ejecutar el Pipeline de Airflow

1. Accede a http://localhost:8080
2. Inicia sesión con `airflow`/`airflow`
3. Encuentra el DAG `mlops_pipeline`
4. Actívalo y ejecútalo manualmente o espera a la programación

### 2. Hacer Predicciones con la API

```python
import requests
import json

# Datos de ejemplo (13 características)
sample_data = {
    "features": [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]
}

# Hacer predicción
response = requests.post(
    "http://localhost:8800/predict",
    json=sample_data
)

result = response.json()
print(f"Predicción: {result['prediction']}")
print(f"Probabilidad: {result['probability']:.2f}")
```

### 3. Ejecutar Experimentos MLflow

```bash
# Configurar variables de entorno
export MLFLOW_TRACKING_URI=http://localhost:5000
export AWS_ACCESS_KEY_ID=minio
export AWS_SECRET_ACCESS_KEY=minio123
export AWS_ENDPOINT_URL_S3=http://localhost:9000

# Ejecutar experimento
python mlflow_hyperparameter_tuning.py
```

## Comandos Útiles

### Gestión de Servicios

```bash
# Levantar todos los servicios
docker compose --profile all up

# Levantar en segundo plano
docker compose --profile all up -d

# Ver logs de un servicio específico
docker compose logs fastapi

# Detener todos los servicios
docker compose --profile all down

# Detener y eliminar volúmenes (¡CUIDADO! Borra datos)
docker compose down --rmi all --volumes
```

### Debugging con Airflow CLI

```bash
# Levantar con perfil de debug
docker compose --profile all --profile debug up

# Usar CLI de Airflow
docker-compose run airflow-cli config list
docker-compose run airflow-cli dags list

# Eliminar un DAG específico
docker-compose run airflow-cli dags delete [dag_id]

# Ver información de un DAG
docker-compose run airflow-cli dags show [dag_id]

# Pausar/despausar un DAG
docker-compose run airflow-cli dags pause [dag_id]
docker-compose run airflow-cli dags unpause [dag_id]
```

## Personalización

### Agregar Nuevos DAGs

1. Crea tu archivo Python en `airflow/dags/`
2. Sigue la estructura del DAG de ejemplo
3. El DAG aparecerá automáticamente en la UI de Airflow

### Modificar la API

1. Edita `dockerfiles/fastapi/app.py`
2. Reconstruye el contenedor: `docker compose build fastapi`
3. Reinicia el servicio: `docker compose restart fastapi`

### Configurar Conexiones de Airflow

Edita `airflow/secrets/connections.yaml` para agregar nuevas conexiones.

## Solución de Problemas

### Servicios no inician

1. Verifica que Docker esté funcionando: `docker ps`
2. Revisa los logs: `docker compose logs [servicio]`
3. Verifica que los puertos no estén ocupados

### Problemas de permisos (Linux/MacOS)

1. Configura `AIRFLOW_UID` en `.env` con tu UID: `id -u`
2. Ejecuta: `sudo chown -R $AIRFLOW_UID:$AIRFLOW_GID airflow/`

### MLflow no se conecta

1. Verifica que MinIO esté funcionando
2. Revisa las variables de entorno de S3
3. Verifica la conectividad: `curl http://localhost:9000`

### DAGs no aparecen o hay DAGs de ejemplo

1. **Si hay DAGs de ejemplo:** Elimina los DAGs específicos:
   ```bash
   docker-compose run airflow-cli dags delete example_dag_id
   ```

2. **Si tu DAG no aparece:** Verifica que esté en `airflow/dags/` y sin errores de sintaxis

3. **Para limpiar completamente:** Detén y elimina volúmenes:
   ```bash
   docker compose --profile all down --volumes
   docker compose --profile all up
   ```

4. **Verificar DAGs disponibles:**
   ```bash
   docker-compose run airflow-cli dags list
   ```

## Contribuciones

Este proyecto está abierto a contribuciones. Algunas ideas para mejorar:

- Agregar más tipos de modelos
- Implementar monitoreo con Grafana
- Agregar tests automatizados
- Implementar CI/CD con GitHub Actions
- Agregar más visualizaciones en MLflow

## Licencia

Este proyecto está bajo la Licencia Apache 2.0. Ver el archivo LICENSE para más detalles.
