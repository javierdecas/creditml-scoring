# CreditML — Pipeline MLOps: Predicción de Aprobación de Préstamos

> **Proyecto académico en MLOps** — Metodologías de Desarrollo y Despliegue de Aplicaciones para Ciencia de Datos (20GIAR)  

**Equipo:** Javier de Castro Santiago
**Proyecto elegido:** Credit Scoring — Pipeline MLOps  
**Repositorio:** https://github.com/javierdecas/creditml-scoring.git

---

## Descripción

Sistema MLOps integral para la predicción de riesgo crediticio que automatiza el ciclo de vida completo del modelo: desde la ingesta de datos históricos hasta el despliegue de una API de puntuación en producción, con monitoreo continuo y reproducibilidad garantizada.

**Caso de negocio:** Una institución financiera recibe miles de solicitudes de préstamo mensuales sin capacidad de decisión automatizada. Con este sistema, el equipo de análisis crediticio puede evaluar automáticamente el riesgo en tiempo real, reduciendo fricciones operativas y tomando decisiones basadas en datos. El modelo predice la aprobación o rechazo a partir del perfil socioeconómico del solicitante.

---

## Arquitectura

```
GitHub Repository
      │
      ├─ push a src/ o data/ ──► GitHub Actions
      │                               │
      │              ┌────────────────┼────────────────┐
      │              ▼                ▼                 ▼
      │         [CI Pipeline]   [ML Pipeline]    [CD Pipeline]
      │         lint + tests    train + eval     build + deploy
      │                         MLflow registry
      │                               │
      │                     ¿AUC-ROC ≥ 0.78?
      │                      /            \
      │                    Sí              No → Notifica Slack
      │                     │
      │              Push imagen LocalStack ECR
      │              Deploy a LocalStack Lambda/ECS
      │
      └─► API REST (FastAPI) en Docker
           /predict  /health  /metrics  /risk-profile
           LocalStack CloudWatch: logs + métricas
```

**Diagrama de componentes:**
- **Data Ingestion**: Carga de datasets desde S3 (LocalStack)
- **Preprocessing**: Limpieza, validación y transformación de variables
- **Feature Engineering**: Construcción de características derivadas
- **Model Training**: XGBoost / LightGBM con validación cruzada estratificada
- **Model Registry**: Almacenamiento y versionado en MLflow
- **API Deployment**: FastAPI containerizada en Docker
- **Monitoring**: Logs estructurados, métricas de latencia y desempeño del modelo

---

## Requisitos previos

- **Python** 3.9 o superior
- **Docker** + **Docker Compose** 2.0+
- **Terraform** 1.4+
- **LocalStack** CLI (simulación local de AWS)
- **Git** (para control de versiones)
- Acceso a variables de entorno configurables (`.env`)

---

## Inicio rápido (local)

### 1. Clonar y configurar entorno

```bash
# Clonar repositorio
git clone https://github.com/[usuario]/creditml-scoring
cd creditml-scoring

# Crear y activar entorno virtual
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Instalar dependencias
pip install -r requirements-dev.txt
```

### 2. Descargar dataset

Descargar el dataset de [Kaggle — Credit Approval Dataset](https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction) o dataset alternativo de UCI Machine Learning Repository.

Colocar el archivo `credit_approval.csv` en:
```
data/raw/credit_approval.csv
```

### 3. Ejecutar pipeline de datos y entrenamiento (local, sin Docker)

```bash
# Ingesta de datos
python -m src.data.ingesta

# Preprocesamiento y transformación
python -m src.data.preprocesamiento

# Feature engineering
python -m src.features.feature_engineering

# Entrenamiento del modelo
python -m src.models.entrenamiento

# Evaluación y reporte
python -m src.models.evaluacion
```

### 4. Levantar la API en Docker

```bash
# Construir e iniciar servicios (API + MLflow + LocalStack)
docker-compose up --build

# La API estará disponible en http://localhost:8000
```

### 5. Probar la API REST

```bash
# Health check
curl http://localhost:8000/health

# Predicción individual
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

# Predicción en lote
curl -X POST http://localhost:8000/predict-batch \
  -H "Content-Type: application/json" \
  -d @batch_requests.json

# Obtener métricas del modelo
curl http://localhost:8000/metrics
```

### 6. Visualizar MLflow

```
http://localhost:5000
```

---

## Resultados del modelo

| Métrica           | Valor  | Objetivo | Descripción                              |
|------------------|--------|----------|------------------------------------------|
| **AUC-ROC**      | 0.823  | ≥ 0.78   | Capacidad de discriminación general      |
| **F1-score**     | 0.756  | ≥ 0.70   | Balance precisión-recall                 |
| **Precision**    | 0.771  | —        | Reducción de rechazos incorrectos        |
| **Recall**       | 0.742  | —        | Cobertura de aprobaciones válidas        |
| **Specificity**  | 0.815  | —        | Identificación de solicitudes de riesgo  |

**Notas:**
- Modelo entrenado con validación cruzada estratificada (5 folds)
- Dataset balanceado con SMOTE en features numéricos
- Threshold optimizado para maximizar F1-score manteniendo precisión > 0.75
- Características seleccionadas con importancia acumulativa > 95%

---

## Estructura del repositorio

```
creditml-scoring/
├── .github/
│   └── workflows/
│       ├── ci.yml              # Linting (flake8, black) + tests (pytest)
│       ├── ml-pipeline.yml     # Entrenamiento, evaluación, registro MLflow
│       └── cd.yml              # Build Docker, deploy a LocalStack Lambda
├── src/
│   ├── data/
│   │   ├── ingesta.py              # Carga y validación del dataset
│   │   ├── preprocesamiento.py     # Limpieza, tratamiento de nulos
│   │   └── validacion_datos.py     # Tests de calidad de datos
│   ├── features/
│   │   ├── feature_engineering.py  # Derivación de características
│   │   └── transformadores.py      # Transformadores custom scikit-learn
│   ├── models/
│   │   ├── entrenamiento.py        # Entrenamiento con MLflow tracking
│   │   ├── evaluacion.py           # Evaluación, gráficas, reporte
│   │   └── predictor.py            # Clase wrapper para predicciones
│   └── api/
│       ├── app.py                  # FastAPI: /predict /health /metrics
│       ├── schemas.py              # Pydantic models para validación
│       └── logger.py               # Logging estructurado
├── tests/
│   ├── test_ingesta.py
│   ├── test_preprocesamiento.py
│   ├── test_feature_engineering.py
│   ├── test_modelo.py
│   ├── test_api.py
│   └── conftest.py                 # Fixtures pytest compartidas
├── notebooks/
│   ├── 01_EDA.ipynb                # Análisis exploratorio
│   ├── 02_Feature_Analysis.ipynb   # Análisis de features
│   └── 03_Model_Comparison.ipynb   # Comparación de modelos candidatos
├── docker/
│   ├── Dockerfile.train            # Imagen para entrenamiento
│   └── Dockerfile.api              # Imagen para API FastAPI
├── docker-compose.yml              # Orquestación de servicios locales
├── infrastructure/
│   ├── main.tf                     # Configuración Terraform principal
│   ├── variables.tf                # Definición de variables
│   ├── outputs.tf                  # Outputs de infraestructura
│   └── localstack.tf               # Recursos LocalStack específicos
├── docs/
│   ├── memoria_proyecto.md
│   ├── product_backlog.md
│   ├── ARQUITECTURA.md             # Detalles arquitectónicos
│   ├── MODELO.md                   # Documentación del modelo
│   └── sprints/
│       ├── sprint1_planning.md
│       ├── sprint2_planning.md
│       ├── sprint3_planning.md
│       └── retrospectiva.md
├── presentacion/
│   └── guia_presentacion.md        # Guía de presentación ante comité
├── .env.example                    # Variables de entorno de ejemplo
├── requirements.txt                # Dependencias de producción
├── requirements-dev.txt            # Dependencias de desarrollo
├── Makefile                        # Comandos útiles para desarrollo
└── README.md                       # Este archivo
```

---

## Flujo de desarrollo y CI/CD

| Pipeline         | Trigger             | Acciones                                              |
|-----------------|---------------------|-------------------------------------------------------|
| `ci.yml`        | Push a cualquier rama | Lint (flake8, black) + tests (pytest) + cobertura   |
| `ml-pipeline.yml` | Push a `main`    | Entrenar modelo, evaluar, registrar en MLflow, push artefactos S3 |
| `cd.yml`        | Tag `v*.*.*`        | Build Docker, push ECR, desplegar a Lambda/ECS LocalStack |

**Opciones de validación:**
- Threshold mínimo de cobertura de tests: **80%**
- Threshold mínimo AUC-ROC en validación: **0.78**
- Umbral máximo de diferencia train/val (overfitting): **5%**

---

## Infraestructura (Terraform + LocalStack)

### Recursos desplegados

- **S3 bucket** `creditml-raw-data` — datos crudos y datasets de entrenamiento
- **S3 bucket** `creditml-processed` — datos preprocesados y features
- **S3 bucket** `creditml-models` — artefactos del modelo MLflow
- **ECR repository** `creditml-api` — imágenes Docker de la API REST
- **CloudWatch Log Group** `/creditml/api` — logs estructurados de predicciones
- **Lambda Function** `creditml-scorer` — función serverless para scoring batch
- **CloudWatch Alarm** — alertas sobre SLA de latencia y tasa de errores

### Provisionar infraestructura

```bash
# Iniciar LocalStack en background
localstack start -d

# Esperar a que esté listo (~30 segundos)
sleep 30

# Navegar a infraestructura
cd infrastructure/

# Inicializar Terraform
terraform init

# Revisar plan de infraestructura
terraform plan

# Aplicar configuración
terraform apply

# (Opcional) Destruir recursos
terraform destroy
```

### Configuración LocalStack

En `docker-compose.yml` se levanta automáticamente:
- **LocalStack** (servicios AWS emulados)
- **MLflow** (registro y tracking de modelos)
- **PostgreSQL** (backend para MLflow)

---

## Monitoreo y observabilidad

### Logs estructurados

Cada predicción se registra en formato JSON en CloudWatch con:
- Timestamp de la solicitud
- Features de entrada (sin datos sensibles)
- Predicción (approval/rejection)
- Probabilidad del modelo
- Latencia en ms
- Versión del modelo utilizada

**Ejemplo:**
```json
{
  "timestamp": "2024-01-15T14:23:45.123Z",
  "request_id": "uuid-xxxx",
  "age": 35,
  "income": 45000,
  "loan_amount": 25000,
  "prediction": "approved",
  "probability": 0.78,
  "latency_ms": 145,
  "model_version": "v1.2.3"
}
```

### Métricas personalizadas

- **CreditMLPredictionLatency** (p50, p95, p99)
- **CreditMLErrorRate** (por tipo de error)
- **CreditMLModelDrift** (divergencia distribucional de features)
- **CreditMLApprovalRate** (proporción de seguimiento en el tiempo)

### Alertas

| Condición                          | Acción                   |
|----------------------------------|--------------------------|
| Latencia p95 > 500ms             | Email + Slack            |
| Tasa de errores HTTP 5xx > 2%    | Email + PagerDuty        |
| Model drift detectado (KS > 0.15) | Notificación de reentrenamiento |

### Dashboard MLflow

Accesible en `http://localhost:5000`:
- Histórico de experimentos y runs
- Comparación de métricas entre versiones
- Gráficos de importancia de features
- Trazabilidad de hiperparámetros

---

## Tecnología y stack

- **Lenguaje:** Python 3.9+
- **ML frameworks:** scikit-learn, XGBoost
- **Data Processing:** pandas, NumPy
- **API:** FastAPI, Pydantic
- **Containerización:** Docker, Docker Compose
- **Orquestación infraestructura:** Terraform
- **Emulación cloud:** LocalStack
- **Experiment tracking:** MLflow
- **Testing:** pytest, pytest-cov
- **Calidad de código:** flake8, black, isort
- **CI/CD:** GitHub Actions

---

## Contribuciones

Seguimos [Conventional Commits](https://www.conventionalcommits.org/) para mensajes de commit y [gitflow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow) para ramas:

```
main (producción)
├── develop (integración)
├── feature/credit-scoring
├── bugfix/model-drift
├── release/v1.0.0
```

**Recomendación:** Crear rama feature, desarrollar, escribir tests, hacer PR contra `develop`.

---

## License

[Especificar licencia: MIT, Apache 2.0, etc.]

---

## Referencias

- [Kaggle — Credit Approval Dataset](https://www.kaggle.com/datasets/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
- [LocalStack Documentation](https://docs.localstack.cloud/)
- [Terraform AWS Provider](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)
- [Credit Risk Modeling Best Practices](https://www.ey.com/en_gl/risk/credit-risk-modeling)
