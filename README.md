# Proyecto Final de MLOps - Pipeline de Predicción de Enfermedades Cardíacas

## Equipo

  * Sebastian Biagiola
  * Erlin Rey
  * Santiago Casado
  * Daniel Bazán
  * Jonathan Matias Borda

-----


Este repositorio contiene un pipeline completo de MLOps para la clasificación de enfermedades cardíacas, utilizando Docker, Apache Airflow, MLflow y FastAPI.

### Tabla de Contenidos
1.  [Características](#características)
2.  [Arquitectura del Sistema](#arquitectura-del-sistema)
3.  [Estructura del Proyecto](#estructura-del-proyecto)
4.  [Requisitos Previos](#requisitos-previos)
5.  [Instalación](#instalación)
6.  [Uso del Sistema](#uso-del-sistema)
7.  [Descripción del Pipeline de Machine Learning](#descripción-del-pipeline-de-machine-learning)
8.  [Referencia de la API](#referencia-de-la-api)
9.  [Solución de Problemas](#solución-de-problemas)
10. [Contribuciones](#contribuciones)
11. [Equipo](#equipo)

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

````

---

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

---

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

## Requisitos Previos

* **Docker**: v24.0 o superior.
* **Docker Compose**: v2.0 o superior (incluido en Docker Desktop).
* **Recursos Mínimos**: 4GB RAM (8GB recomendados), 2 CPU cores (4 recomendados).

Para verificar su instalación, ejecute:
```bash
docker --version
docker-compose --version
````

-----

## Instalación

1.  **Clonar el repositorio**

    ```bash
    git clone [https://github.com/joniborda/MLOps.git](https://github.com/joniborda/MLOps.git)
    cd MLOps
    ```

2.  **Configurar variables de entorno**

    ```bash
    cp env.example .env
    ```

    En sistemas Linux o macOS, es necesario configurar el UID del usuario actual para evitar problemas de permisos con Airflow:

    ```bash
    echo "AIRFLOW_UID=$(id -u)" >> .env
    ```

3.  **Levantar todos los servicios**

    ```bash
    docker-compose --profile all up -d
    ```

4.  **Verificar el estado de los contenedores**

    ```bash
    docker-compose ps
    ```

    Todos los servicios deberían aparecer con el estado `running` o `healthy`.

-----

## Uso del Sistema

#### Acceso a los Servicios

Una vez iniciados los contenedores, las interfaces de los servicios están disponibles en las siguientes URLs:

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

-----

## Descripción del Pipeline de Machine Learning

El DAG `mlops_pipeline` ejecuta las siguientes etapas de forma orquestada:

```
Setup MLflow
    ↓
Extract Data
    ↓
Preprocess Data
    ↓
┌─────────────┬─────────────┬─────────────┐
│   Train     │   Train     │   Train     │
│ Random      │  Logistic   │     SVM     │
│  Forest     │ Regression  │             │
└─────┬───────┴──────┬──────┴──────┬──────┘
      ↓              ↓             ↓
  Evaluate RF   Evaluate LR   Evaluate SVM
      │              │             │
      └──────────┬───┴─────────────┘
                 ↓
         Select Best Model
                 ↓
        Cleanup Old Models
```

  * **Setup MLflow**: Configura el servidor de tracking y el nombre del experimento.
  * **Extract Data**: Carga el dataset de enfermedades cardíacas desde un archivo fuente.
  * **Preprocess Data**: Realiza la limpieza, escalado de características y división en conjuntos de entrenamiento y prueba.
  * **Train Models**: Entrena tres modelos distintos en paralelo.
  * **Evaluate Models**: Calcula métricas de rendimiento (accuracy, precision, recall, F1-score) para cada modelo.
  * **Select Best Model**: Compara los modelos según sus métricas y registra el mejor en el Model Registry de MLflow.
  * **Cleanup**: Mantiene solo las últimas 5 versiones del modelo registrado para evitar la acumulación de artefactos.

-----

## Referencia de la API

#### Realizar una Predicción

Endpoint para obtener una predicción a partir de un conjunto de características.

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

#### Otros Endpoints Disponibles

  * `GET /`: Devuelve información básica de la API.
  * `GET /health`: Realiza un chequeo de salud del servicio.
  * `GET /model/info`: Devuelve metadatos del modelo actualmente cargado en memoria.
  * `POST /model/reload`: Fuerza la recarga del último modelo registrado en MLflow.

La documentación interactiva completa de la API está disponible en `http://localhost:8800/docs`.

-----

## Solución de Problemas

  * **Contenedores no inician**: Verifique los logs de un servicio específico con `docker-compose logs [nombre-del-servicio]`.
  * **Errores de permisos en Linux/macOS**: Asegúrese de que la variable `AIRFLOW_UID` en el archivo `.env` coincida con su ID de usuario (`id -u`). Adicionalmente, puede reasignar permisos con `sudo chown -R $(id -u):$(id -g) airflow/`.
  * **Puertos en uso**: Si un puerto ya está ocupado, edite el archivo `.env` y asigne un puerto diferente a la variable correspondiente (ej. `AIRFLOW_PORT=8081`).
  * **Limpieza total del entorno**: Para eliminar todos los contenedores, redes y volúmenes (esto borrará todos los datos), ejecute:
    ```bash
    docker-compose --profile all down --volumes
    ```

-----

## Contribuciones

Las contribuciones a este proyecto son bienvenidas. Para ello, por favor siga estos pasos:

1.  Realice un Fork del repositorio.
2.  Cree una nueva rama para su funcionalidad (`git checkout -b feature/nueva-funcionalidad`).
3.  Realice un Commit con sus cambios (`git commit -m 'Agrega nueva funcionalidad'`).
4.  Haga un Push a la rama (`git push origin feature/nueva-funcionalidad`).
5.  Abra un Pull Request.

-----


## Licencia

Este proyecto se distribuye bajo la Licencia Apache 2.0. Consulte el archivo `LICENSE` para más detalles.

