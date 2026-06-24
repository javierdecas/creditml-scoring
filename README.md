# CreditML — Pipeline MLOps para Predicción de Aprobación de Préstamos

> Proyecto académico de MLOps para credit scoring con un pipeline local reproducible.

**Equipo:** Javier de Castro Santiago

---

## Descripción

CreditML es una implementación de un pipeline básico de MLOps para la predicción de aprobación de préstamos. El repositorio cubre la ingestión y validación de datos, el preprocesamiento, el entrenamiento de un modelo de clasificación, y un servicio REST para inferencia.

El servicio expone una API FastAPI para solicitudes individuales de scoring y endpoints de estado. La solución está diseñada para ejecutarse localmente con Docker y tiene soporte de infraestructura declarativa compatible con LocalStack para S3/ECR.

---

## Funcionalidad implementada

- Validación y limpieza de datos de crédito con `src.data.ingesta`
- Preprocesamiento de datos con `src.data.preprocesamiento`
- Entrenamiento de modelo con `src.models.entrenamiento`
- Registro de métricas de entrenamiento en MLflow (`mlruns/`)
- Servicio REST con FastAPI en `src.api.main`
- Endpoints disponibles:
  - `GET /health`
  - `GET /metrics`
  - `POST /predict`

---

## Requisitos previos

- Python 3.11
- Docker
- Docker Compose
- Terraform (solo si se usa la configuración en `infrastructure/`)

---

## Inicio rápido (local)

### 1. Clonar el repositorio

```bash
git clone https://github.com/javierdecas/creditml-scoring.git
cd creditml-scoring
```

### 2. Crear y activar un entorno virtual

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows
# .venv\Scripts\activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements-dev.txt
```

### 4. Colocar el dataset

Copiar el archivo `credit_approval.csv` a:

```text
data/raw/credit_approval.csv
```

### 5. Ejecutar los pasos locales de datos y entrenamiento

```bash
python -m src.data.ingesta
python -m src.data.preprocesamiento
python -m src.models.entrenamiento
```

Al finalizar, el artefacto del modelo se guarda como `models/credit_model.pkl`.

### 6. Levantar la API con Docker Compose

```bash
docker-compose up --build
```

La API quedará accesible en:

```text
http://localhost:8000
```

---

## API disponible

### GET /health

Verifica que el servicio está activo y si el modelo está cargado.

```bash
curl http://localhost:8000/health
```

### GET /metrics

Devuelve metadatos del modelo y del servicio.

```bash
curl http://localhost:8000/metrics
```

### POST /predict

Realiza una inferencia de scoring con los campos requeridos.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "income": 45000.0,
    "employment_status": "employed",
    "loan_amount": 25000.0,
    "credit_history_length": 8,
    "existing_debt": 5000.0,
    "payment_history": "excellent"
  }'
```

---

## Estructura principal del repositorio

```text
creditml-scoring/
├── .github/
│   └── workflows/
│       └── ci.yml
├── docker/
│   └── Dockerfile.api
├── infrastructure/
│   └── main.tf
├── src/
│   ├── api/
│   │   ├── main.py
│   │   └── schemas.py
│   ├── data/
│   │   ├── ingesta.py
│   │   └── preprocesamiento.py
│   └── models/
│       ├── entrenamiento.py
│       └── evaluacion.py
├── tests/
│   └── test_api.py
├── docker-compose.yml
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

---

## Pipeline y pruebas

El flujo de CI actual en `.github/workflows/ci.yml` cubre:

- instalación de dependencias
- linting con `flake8`
- pruebas con `pytest`
- build de imagen Docker para la API

La configuración actual no incluye despliegue automático en AWS, notificaciones por Slack/PagerDuty ni un pipeline de CD completo.

---

## Infraestructura local

El archivo `infrastructure/main.tf` contiene configuración de Terraform para aprovisionar recursos LocalStack compatibles con AWS:

- bucket S3 local
- repositorio ECR local

El servicio de Docker Compose arranca `api` y `localstack`, pero la API no requiere AWS para responder a `/predict`.

---

## Alcance real del MVP

### Implementado

- ingestión y validación de datos de crédito
- preprocesamiento y codificación categórica
- entrenamiento y exportación de modelo con `scikit-learn`
- servicio REST FastAPI con `POST /predict`
- endpoints de estado `/health` y `/metrics`
- pruebas unitarias básicas de API

### No implementado

- endpoint de lote `/predict-batch`
- endpoint de explicabilidad `/risk-profile`
- detección automática de model drift
- alertas a Slack o PagerDuty
- despliegue real en AWS Lambda

---

## Referencias

- [FastAPI](https://fastapi.tiangolo.com/)
- [MLflow](https://mlflow.org/docs/latest/index.html)
- [LocalStack](https://docs.localstack.cloud/)
- [Terraform AWS Provider](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)
