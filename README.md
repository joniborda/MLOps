# Proyecto Final de MLOps - Pipeline de Predicción de Enfermedades Cardíacas
### Aprendizaje de Máquina I - CEIA - FIUBA

Este repositorio contiene un pipeline completo de MLOps para la clasificación de enfermedades cardíacas, utilizando Docker, Apache Airflow, MLflow y FastAPI.


## Equipo

  * Sebastian Biagiola
  * Erlin Rey
  * Santiago Casado
  * Daniel Bazán
  * Jonathan Matias Borda

-----

### Tabla de Contenidos
1.  [Características](#características)
2.  [Arquitectura del Sistema](#arquitectura-del-sistema)
3.  [Estructura del Proyecto](#estructura-del-proyecto)
4.  [Modelo de Predicción](#modelo-de-predicción)
5.  [Requisitos Previos](#requisitos-previos)
6.  [Instalación y Configuración](#instalación-y-configuración)
7.  [Uso del Sistema](#uso-del-sistema)
8.  [Referencia de la API](#referencia-de-la-api)
9.  [Personalización](#personalización)
10.  [Comandos Útiles](#comandos-útiles)
11. [Solución de Problemas](#solución-de-problemas)
12. [Contribuciones](#contribuciones)
13. [Licencia](#licencia)
14. [Equipo](#equipo)

---

## Características

* **Pipeline Automatizado**: Orquestación de la extracción, preprocesamiento, entrenamiento y evaluación de modelos.
* **Comparación de Modelos**: Entrenamiento y evaluación en paralelo de Random Forest, Logistic Regression y SVM.
* **Seguimiento de Experimentos**: Gestión completa del ciclo de vida de los modelos con MLflow.
* **API REST para Inferencia**: Servicio de predicciones en tiempo real con FastAPI.
* **Orquestación de Workflows**: Flujos de trabajo gestionados con Apache Airflow.
* **Almacenamiento de Objetos**: MinIO como backend compatible con S3 para artefactos y datasets.
* **Entorno Containerizado**: Todo el stack de servicios se ejecuta en contenedores Docker.

---

## Arquitectura del Sistema

```

┌─────────────────┐
│   Airflow UI    │──────┐
│   (Port 8080)   │      │
└─────────────────┘      │
│
┌─────────────────┐      │      ┌─────────────────┐
│   MLflow UI     │──────┼──────│   PostgreSQL    │
│   (Port 5432)   │      │      │ (DB Backend)    │
└─────────────────┘      │      └─────────────────┘
│
┌─────────────────┐      │      ┌─────────────────┐
│   FastAPI       │──────┼──────│   MinIO (S3)    │
│   (Port 8800)   │      │      │ (Port 9000)     │
└─────────────────┘      │      └─────────────────┘
│
┌────▼────┐
│ Docker  │
│ Network │
└─────────┘

Almacenamiento S3 (MinIO)

Buckets automáticamente creados:
- `data`: Para datasets y datos procesados
- `mlflow`: Para artefactos de MLflow


```

#### Componentes Principales
| Componente | Descripción | Puerto |
| :--- | :--- | :--- |
| **Apache Airflow** | Orquestación de workflows y pipelines de ML | `8080` |
| **MLflow** | Gestión de experimentos, modelos y artefactos | `5000` |
| **FastAPI** | API REST para servir los modelos entrenados | `8800` |
| **MinIO** | Almacenamiento de objetos (S3-compatible) | `9000` (API), `9001` (UI) |
| **PostgreSQL**| Base de datos para los metadatos de Airflow y MLflow | `5432` |
| **Valkey** | Base de datos key-value para el broker de Celery en Airflow | `6379` |

---

## Estructura del Proyecto

```

MLOps/
├── airflow/
│   ├── dags/                 \# Contiene los DAGs de Airflow
│   ├── logs/                 \# Almacena los logs generados por Airflow
│   ├── plugins/              \# Para plugins personalizados de Airflow
│   └── config/               \# Archivos de configuración de Airflow
├── data/                     \# Datasets para el proyecto
│   └── heart.csv             \# Dataset de enfermedades cardíacas
├── models/                   \# Modelos entrenados (si se guardan localmente)
├── scripts/                  \# Lógica del modelo y scripts de utilidades
│   ├── model\_utils.py        \# Clases y funciones para el modelo
│   └── requirements.txt      \# Dependencias de Python para los scripts
├── dockerfiles/              \# Dockerfiles para construir las imágenes
│   ├── airflow/
│   └── fastapi/
├── docker-compose.yaml       \# Define y configura todos los servicios
├── env.example               \# Plantilla para las variables de entorno
└── mlflow\_hyperparameter\_tuning.py \# Script de ejemplo para MLflow

````

---

## Modelo de Predicción

Este proyecto aborda un problema de clasificación binaria para predecir la presencia de enfermedad cardíaca.

- **Qué predice**: probabilidad/presencia de enfermedad cardíaca (0 = no, 1 = sí).
- **Entrada del modelo**: 13 características clínicas numéricas del dataset de corazón (`data/heart.csv`). En el endpoint `/predict` se envían en una lista llamada `features`, en el mismo orden usado durante el entrenamiento. Referencia habitual del conjunto UCI Heart Disease:
  - `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`.
- **Estrategia y algoritmos**: se entrenan y comparan múltiples modelos definidos en `scripts/models/`:
  - Regresión Logística (`HeartDiseaseLogisticRegressionModel`)
  - Random Forest (`HeartDiseaseRandomForestModel`)
  - SVM (`HeartDiseaseSVMModel`)
- **Selección y tracking**: el desempeño se registra con MLflow; el mejor modelo se registra en el Model Registry con el nombre `heart_disease_classifier` y es el que sirve FastAPI por defecto.
- **Tuning opcional**: puedes ejecutar `mlflow_hyperparameter_tuning.py` para buscar hiperparámetros y registrar corridas en MLflow.

> Nota: el servicio de inferencia espera los valores ya numéricos y en el mismo orden que el entrenamiento. Ver ejemplo en la sección [Referencia de la API](#referencia-de-la-api).

---

## Requisitos Previos

1. **Docker**: Instala Docker Desktop o Docker Engine
2. **Docker Compose**: Viene incluido con Docker Desktop

Para verificar su instalación, ejecute:
```bash
docker --version
docker-compose --version
````

-----

## Instalación y Configuración

1.  **Clonar el repositorio**

    ```bash
    git clone [https://github.com/joniborda/MLOps.git](https://github.com/joniborda/MLOps.git)
    cd MLOps
    ```

2.  **Crear el archivo de variables de entorno**

    ```bash
    cp env.example .env
    ```

    En sistemas Linux o macOS, es necesario configurar el UID del usuario actual para evitar problemas de permisos con Airflow:

    ```bash
    echo "AIRFLOW_UID=$(id -u)" >> .env
    ```

3.  **Levantar todos los servicios**

    ```bash
    docker compose --profile all up
    ```

    
4.  **Verificar el estado de los contenedores**

    ```bash
    docker ps -a
    ```

    

-----

## Uso del Sistema

#### Acceso a los Servicios

Una vez iniciados los contenedores, las interfaces están disponibles en las siguientes URLs:

| Servicio | URL | Usuario | Contraseña |
| :--- | :--- | :--- | :--- |
| **Airflow** | `http://localhost:8080` | `airflow` | `airflow` |
| **MLflow** | `http://localhost:5000` | - | - |
| **MinIO Console** | `http://localhost:9001` | `minio` | `minio123` |
| **FastAPI** | `http://localhost:8800` | - | - |
| **API Docs** | `http://localhost:8800/docs` | - | - |

#### Ejecutar el Pipeline de Machine Learning

1.  Acceder a la interfaz de **Airflow**: `http://localhost:8080`.
2.  Iniciar sesión con las credenciales por defecto.
3.  Localizar el DAG llamado `mlops_pipeline`.
4.  Activar el DAG utilizando el interruptor (toggle).
5.  Ejecutarlo manualmente haciendo clic en el botón de "Play".

#### Ejecutar Experimentos de Hyperparameter Tuning

Puedes ejecutar un script de ejemplo para buscar hiperparámetros y registrar los resultados en MLflow.

1.  **Configurar variables de entorno en la terminal**:

    ```bash
    export MLFLOW_TRACKING_URI=http://localhost:5000
    export AWS_ACCESS_KEY_ID=minio
    export AWS_SECRET_ACCESS_KEY=minio123
    export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
    ```

2.  **Ejecutar el script**:

    ```bash
    python mlflow_hyperparameter_tuning.py
    ```

-----

## Referencia de la API

#### Realizar una Predicción (Ejemplo con `curl`)

  * **Request (`POST /predict`)**

    ```bash
    curl -X POST "http://localhost:8800/predict" \
      -H "Content-Type: application/json" \
      -d '{
        "features": [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]
      }'
    ```

  * **Response**

    ```json
    {
      "prediction": 1,
      "probability": 0.85,
      "model_version": "3"
    }
    ```

#### Realizar una Predicción (Ejemplo con `Python`)

```python
import requests
import json

# Datos de ejemplo (13 características)
sample_data = {
    "features": [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]
}

# Hacer la petición a la API
response = requests.post(
    "http://localhost:8800/predict",
    json=sample_data
)

if response.status_code == 200:
    result = response.json()
    print(f"Predicción: {result['prediction']}")
    print(f"Probabilidad: {result['probability']:.2%}")
    print(f"Versión del modelo: {result['model_version']}")
else:
    print(f"Error: {response.status_code}")
    print(response.json())
```

#### Otros Endpoints Disponibles

  * `GET /`: Devuelve información básica de la API.
  * `GET /health`: Realiza un chequeo de salud del servicio.
  * `GET /model/info`: Devuelve metadatos del modelo actualmente cargado en memoria.
  * `POST /model/reload`: Fuerza la recarga del último modelo registrado en MLflow.

La documentación interactiva completa de la API está disponible en `http://localhost:8800/docs`.

-----

## Personalización

### Agregar Nuevos DAGs

1.  Crea tu archivo Python en la carpeta `airflow/dags/`.
2.  El DAG aparecerá automáticamente en la UI de Airflow después de unos momentos.

### Modificar la API

1.  Edita el archivo `dockerfiles/fastapi/app.py` con tu nueva lógica.
2.  Reconstruye la imagen del contenedor de FastAPI: `docker-compose build fastapi`.
3.  Reinicia el servicio para aplicar los cambios: `docker-compose restart fastapi`.

### Configurar Conexiones de Airflow

Para agregar conexiones de forma declarativa, edita el archivo `airflow/config/connections.yaml`.

-----

## Comandos Útiles

### Gestión de Servicios

```bash
# Levantar todos los servicios en segundo plano
docker-compose --profile all up -d

# Detener todos los servicios
docker-compose --profile all down

# Ver el estado de los contenedores
docker-compose ps

# Ver los logs de un servicio específico en tiempo real
docker-compose logs -f [nombre-del-servicio]

# Detener y eliminar volúmenes (¡CUIDADO! Borra todos los datos)
docker-compose --profile all down --volumes
```

### Debugging con Airflow CLI

```bash
# Listar todos los DAGs disponibles
docker-compose run airflow-cli dags list

# Ver la información detallada de un DAG
docker-compose run airflow-cli dags show mlops_pipeline

# Pausar un DAG
docker-compose run airflow-cli dags pause mlops_pipeline

# Reanudar un DAG
docker-compose run airflow-cli dags unpause mlops_pipeline
```

-----

## Solución de Problemas

  * **Servicios no inician o fallan al levantar**:

    1.  **Verifica que Docker esté activo** y que los contenedores se hayan creado. El comando `ps` te mostrará el estado de cada uno.
        ```bash
        docker-compose ps
        ```
    2.  **Revisa los logs** del servicio que está fallando para ver el mensaje de error específico.
        ```bash
        docker-compose logs [nombre-del-servicio]
        ```
    3.  **Confirma que los puertos** definidos en el archivo `.env` (ej. `8080`, `5000`, etc.) no estén siendo utilizados por otro programa en tu sistema.

  * **Errores de permisos en Linux/macOS**:

      * Asegúrate de que la variable `AIRFLOW_UID` en el archivo `.env` coincida con tu ID de usuario (`id -u`). Adicionalmente, puedes reasignar permisos a la carpeta de Airflow:
        ```bash
        sudo chown -R $(id -u):$(id -g) airflow/
        ```

  * **MLflow no se conecta a MinIO**:

    1.  Verifica que el contenedor de `minio` esté en estado `running` con `docker-compose ps`.
    2.  Revisa que las variables de entorno de S3 en tu archivo `.env` sean correctas.
    3.  Verifica la conectividad de red con el contenedor: `curl http://localhost:9000`.

  * **DAGs no aparecen en la UI de Airflow**:

    1.  **Verifica si hay errores de sintaxis** en los logs del contenedor `airflow-scheduler`.
    2.  **Asegúrate de que el archivo del DAG** esté dentro de la carpeta `airflow/dags/`.
    3.  **Lista los DAGs desde el CLI** para confirmar si Airflow los ha detectado:
        ```bash
        docker-compose run airflow-cli dags list
        ```
    4.  **Para eliminar los DAGs de ejemplo** que vienen con Airflow, puedes usar el CLI:
        ```bash
        docker-compose run airflow-cli dags delete [dag_id_ejemplo]
        ```

-----

## Contribuciones

Las contribuciones a este proyecto son bienvenidas.

1.  Realiza un Fork del repositorio.
2.  Crea una nueva rama para tu funcionalidad (`git checkout -b feature/nueva-funcionalidad`).
3.  Realiza un Commit con sus cambios (`git commit -m 'Agrega nueva funcionalidad'`).
4.  Haz un Push a la rama (`git push origin feature/nueva-funcionalidad`).
5.  Abre un Pull Request.

#### Ideas para mejorar:

  - Agregar más tipos de modelos al pipeline.
  - Implementar monitoreo de servicios con Grafana y Prometheus.
  - Agregar tests automatizados para la API y los componentes del pipeline.
  - Implementar un flujo de CI/CD con GitHub Actions.
  - Agregar más visualizaciones y reportes en MLflow.

-----

## Licencia

Este proyecto se distribuye bajo la Licencia Apache 2.0. Consulte el archivo `LICENSE` para más detalles.


