# Product Backlog — CreditML

**Proyecto:** Predicción de Aprobación de Préstamos (Credit Scoring)  
**Versión:** 1.0  
**Fecha:** Enero 2024  
**Propietario del producto:** [Nombre responsable]

---

## Resumen ejecutivo

El backlog se estructura en **3 sprints** de 2 semanas cada uno, con **6 User Stories** de máxima prioridad que cubren el ciclo de vida completo del sistema MLOps: desde ingesta y preprocesamiento de datos, hasta despliegue y monitoreo de la API REST en producción.

**Objetivos alcanzables:**
- **Sprint 1:** Pipeline de datos (ingesta, preprocesamiento, feature engineering)
- **Sprint 2:** Modelo, API REST y testing
- **Sprint 3:** CI/CD, despliegue y monitoreo

---

## Sprint 1: Foundation — Data Pipeline & Feature Engineering

**Duración:** 2 semanas  
**Objetivo:** Implementar el pipeline completo de ingesta, preprocesamiento y feature engineering.

### US-001: Ingesta, validación y preprocesamiento de datos

**Como** ingeniero de datos  
**quiero** cargar, validar y limpiar automáticamente el dataset de solicitudes de préstamo  
**para que** garantice que los datos son consistentes y de calidad antes del modelado

**Story Points:** 8  
**Prioridad:** Crítica

**Descripción:**
Implementar los módulos `src/data/ingesta.py` y `src/data/preprocesamiento.py` que:

**Ingesta:**
- Lee el archivo CSV desde `data/raw/credit_approval.csv`
- Valida tipos de datos e índices
- Detecta y documenta valores faltantes
- Genera reporte de calidad de datos
- Persiste datos cargados en S3 (LocalStack)

**Preprocesamiento:**
- Imputa valores faltantes (media para numéricos, moda para categóricos)
- Detecta y maneja outliers (IQR method)
- Normaliza variables numéricas (StandardScaler)
- Codifica variables categóricas (LabelEncoder/OneHotEncoder)
- Persiste transformadores sklearn para consistencia train-serve
- Genera reporte de distribuciones antes/después

**Criterios de aceptación:**
- [ ] Scripts ejecutables: `python -m src.data.ingesta` y `python -m src.data.preprocesamiento`
- [ ] Reporte de validación en formato JSON con estadísticas descriptivas
- [ ] Cero valores NaN en datos de salida
- [ ] Distribuciones normalizadas (media ≈ 0, std ≈ 1 para features numéricos)
- [ ] Transformadores persistidos en `models/artifacts/`
- [ ] Cobertura de tests ≥ 80%
- [ ] Logs estructurados en consola y archivo
- [ ] Documentación de variables en docstrings

**Tareas técnicas:**
- Implementar funciones de lectura con manejo de excepciones
- Agregar validaciones de esquema con `pydantic`
- Crear clase Preprocessor heredable de pipeline sklearn
- Tests de casos edge (100% nulos, valores extremos)
- Generar visualizaciones antes/después (seaborn)

**Definición done:**
- Código mergeado a `develop`
- Tests pasando con cobertura ≥ 80%
- Datos preprocesados reproducibles
- Documentación actualizada
- Ejecutable sin errores en entorno local

---

### US-002: Feature Engineering y selección de características

**Como** científico de datos  
**quiero** derivar nuevas características del dataset original y seleccionar las más predictivas  
**para que** mejore la capacidad predictiva del modelo

**Story Points:** 5  
**Prioridad:** Alta

**Descripción:**
Implementar el módulo `src/features/feature_engineering.py` que:
- Crea ratios: `debt_to_income`, `loan_to_income`, `credit_age_ratio`
- Aplica transformaciones polinómicas y logarítmicas a variables sesgadas
- Construye variables dummy para categorías de mayor importancia
- Realiza selección de features usando `SelectKBest` y análisis de correlación
- Genera matriz de correlación y gráficos de importancia
- Reduce dimensionalidad manteniendo 95% de varianza (PCA opcional)
- Documenta justificación de cada feature creada

**Criterios de aceptación:**
- [ ] Script ejecutable: `python -m src.features.feature_engineering`
- [ ] Mínimo 5 nuevas features derivadas documentadas
- [ ] Matriz de correlación sin multicolinealidad (VIF < 5)
- [ ] Features seleccionadas explicadas en `docs/FEATURES.md`
- [ ] Tests validando rango y distribución de nuevas features
- [ ] Visualización de importancia relativa generada
- [ ] Cobertura ≥ 80%

**Tareas técnicas:**
- Implementar transformadores custom (sklearn compatible)
- Análisis VIF y detección de multicolinealidad
- Generar reportes HTML con visualizaciones
- Tests parametrizados para múltiples datasets

**Definición done:**
- Código mergeado
- Feature set final documentado
- Cobertura ≥ 80%
- Notebook exploratoria actualizada (`02_Feature_Analysis.ipynb`)

---

## Sprint 2: ML Model & API REST

**Duración:** 2 semanas  
**Objetivo:** Entrenar modelo, implementar API REST e integración con MLflow.

### US-003: Entrenamiento, evaluación y registro del modelo

**Como** científico de datos  
**quiero** entrenar un modelo de clasificación balanceado con validación cruzada y registrarlo en MLflow  
**para que** tenga traceabilidad de experimentos, reproducibilidad y governance del mejor modelo

**Story Points:** 8  
**Prioridad:** Crítica

**Descripción:**
Implementar el módulo `src/models/entrenamiento.py` que:
- Entrena modelo XGBoost con validación cruzada estratificada (5-fold)
- Balancea dataset con SMOTE en train set
- Registra cada experimento en MLflow (hiperparámetros, métricas, artefactos)
- Optimiza threshold de decisión para maximizar F1-score
- Genera matriz de confusión, curvas ROC, PR, importancia de features
- Persiste modelo entrenado en formato pickle y ONNX
- Documenta versión del modelo con timestamp y git commit hash
- Evalúa con reporte de métricas (AUC-ROC, F1, Precision, Recall, Specificity)

**Criterios de aceptación:**
- [ ] Script ejecutable: `python -m src.models.entrenamiento`
- [ ] Modelo entrenado con AUC-ROC ≥ 0.78 en validación cruzada
- [ ] Diferencia train-test < 5% (validación sin overfitting)
- [ ] MLflow UI muestra mínimo 5 experimentos y comparación de runs
- [ ] Modelo persistido en `models/artifacts/best_model.pkl` y `best_model.onnx`
- [ ] Tests validando reproducibilidad (same seed = same results)
- [ ] Reporte de evaluación en `reports/model_report.html`
- [ ] Cobertura de tests ≥ 80%

**Tareas técnicas:**
- Implementar pipeline sklearn completo con XGBoost
- Integración con MLflow Tracking API
- Grid search de hiperparámetros críticos
- Generación de gráficos de evaluación con matplotlib/seaborn
- Tests unitarios de modelos con diferentes semillas

**Definición done:**
- Código mergeado
- Modelo entrenado y registrado en MLflow con AUC-ROC ≥ 0.78
- Cobertura ≥ 80%
- Documentación de arquitectura del modelo en `docs/MODELO.md`

---

### US-004: API REST para predicciones y testing

**Como** analista crediticio  
**quiero** usar una API REST para obtener predicciones de aprobación en tiempo real y que sea confiable  
**para que** pueda integrarla en mi flujo de análisis de solicitudes

**Story Points:** 8  
**Prioridad:** Crítica

**Descripción:**
Implementar el módulo `src/api/app.py` (FastAPI) con endpoints:
- `POST /predict` — Predicción individual con score, probabilidad y explicabilidad
- `POST /predict-batch` — Predicción en lote (máx 1000 registros)
- `GET /health` — Health check con versión del modelo
- `GET /metrics` — Métricas de desempeño del modelo en JSON
- `GET /risk-profile/{loan_id}` — Explicabilidad con SHAP values
- Validación de inputs con Pydantic schemas y esquema JSON
- Manejo de errores con códigos HTTP correctos (400, 500, 503)
- CORS habilitado
- Logging estructurado JSON de cada predicción
- Rate limiting: máx 100 req/min por IP
- Documentación automática en `/docs` (Swagger)

**Criterios de aceptación:**
- [ ] API ejecutable: `python -m src.api.app`
- [ ] Todos los endpoints responden con status 200/400/500/503 apropiados
- [ ] Validación de inputs correcta (rechaza datos inválidos)
- [ ] Latencia p95 < 500ms para predicción individual
- [ ] Tests de integración pasando (endpoint testing completo)
- [ ] Unit tests de schemas y lógica de predicción
- [ ] Cobertura de tests ≥ 80%
- [ ] Swagger docs accesible en `/docs`
- [ ] Logging en formato JSON persistido en archivo

**Tareas técnicas:**
- Implementar schemas Pydantic para validation
- Cargar modelo persistido en startup con caché
- Implementar middleware de logging  y rate limiting
- Agregar integración SHAP básica
- Tests con `TestClient` de FastAPI
- Fixtures pytest compartidas

**Definición done:**
- Código mergeado
- API testeada localmente sin errores
- Cobertura ≥ 80%
- Documentación de API en `docs/API.md`

---

## Sprint 3: CI/CD, Deployment & Monitoring

**Duración:** 2 semanas  
**Objetivo:** Automatizar CI/CD, desplegar en LocalStack e implementar monitoreo.

### US-005: CI/CD Pipeline con GitHub Actions

**Como** desarrollador  
**quiero** que cada push ejecute automáticamente linting, tests, entrenamiento y evaluación del modelo  
**para que** garantice la calidad del código y el modelo antes de deployment

**Story Points:** 8  
**Prioridad:** Alta

**Descripción:**
Implementar flujos de GitHub Actions (`.github/workflows/`):

**ci.yml** (en cada push a cualquier rama):
- Linting: flake8, black check, isort
- Tests unitarios e integración con pytest
- Cobertura mínima 80%
- Build documentación

**ml-pipeline.yml** (en push a `main`):
- Entrenar modelo en dataset completo
- Evaluar con AUC-ROC ≥ 0.78
- Si cumple: registrar en MLflow y persistir artefactos
- Si no cumple: notificar Slack y fallar build

**cd.yml** (en tag `v*.*.*`):
- Build imagen Docker de la API
- Push a ECR (LocalStack)
- Deploy a Lambda con IaC
- Smoke tests en staging

**Criterios de aceptación:**
- [ ] `.github/workflows/` contiene 3 archivos YAML válidos
- [ ] Pipeline ci.yml ejecuta en < 5 minutos
- [ ] Pipeline ml-pipeline.yml ejecuta en < 20 minutos
- [ ] Cobertura mínima: 80% (falla si < 80%)
- [ ] Tests de integración pasando
- [ ] Notificaciones Slack configuradas y funcionando
- [ ] Documentación de pipelines en `docs/CI_CD.md`
- [ ] Secrets configurados en GitHub (AWS creds, Slack webhook)

**Tareas técnicas:**
- Crear archivos YAML con sintaxis y validación correctas
- Configurar matrix testing (Python 3.9+)
- Implementar smoke tests post-deployment
- Agregar badges de status en README
- Configurar permiso de token de GitHub

**Definición done:**
- Código mergeado
- Pipelines ejecutándose sin errores
- Todos los pasos pasando
- Documentación completa

---

### US-006: Despliegue con Terraform + LocalStack y Monitoreo

**Como** DevOps engineer  
**quiero** desplegar la infraestructura completa en LocalStack con IaC y monitorear el sistema  
**para que** sea reproducible, escalable y observable

**Story Points:** 8  
**Prioridad:** Alta

**Descripción:**

**Infraestructura (Terraform):**
Configurar `infrastructure/main.tf` que provisiona:
- S3 buckets: `creditml-raw-data`, `creditml-processed`, `creditml-models`
- ECR repository: `creditml-api`
- CloudWatch Log Group: `/creditml/api`
- Lambda function `creditml-scorer` con triggers S3
- CloudWatch Alarms para latencia, error rate
- Secrets Manager para credenciales
- Outputs de recursos creados

Scripts y documentación:
- `scripts/setup-localstack.sh` — Iniciar y configurar LocalStack
- `infrastructure/terraform.tfvars` — Configuración dinámica
- `infrastructure/outputs.tf` — Exposición de recursos

**Observabilidad:**
- Logs estructurados (JSON) en CloudWatch con timestamp, request_id, latencia
- Métricas personalizadas: `CreditMLPredictionLatency`, `CreditMLErrorRate`, `CreditMLApprovalRate`
- Dashboard CloudWatch con visualización de métricas
- Alertas: latencia p95 > 500ms → Email + Slack, error rate > 2% → PagerDuty
- Detección de model drift (KS test > 0.15)

**Criterios de aceptación:**
- [ ] `terraform init && terraform plan && terraform apply` sin errores
- [ ] Todos los recursos visibles en LocalStack UI
- [ ] S3 buckets con datos de ejemplo
- [ ] Lambda ejecutable desde CLI e integrado con S3
- [ ] Logs en CloudWatch accesibles y en formato JSON
- [ ] Dashboard CloudWatch creado con métricas clave
- [ ] Alertas configuradas y testeadas
- [ ] 30 días de histórico de logs disponible
- [ ] Documentación en `docs/INFRAESTRUCTURA.md` y `docs/MONITOREO.md`
- [ ] Scripts ejecutables y testeados

**Tareas técnicas:**
- Escribir módulos Terraform reutilizables
- Parametrizar valores en variables.tf
- Integración LocalStack con AWS provider
- Implementar logger centralizado en `src/api/logger.py`
- CloudWatch metrics con boto3
- Crear métricas custom de drift
- Tests de infraestructura
- Documentar procedimientos operativos

**Definición done:**
- Código mergeado
- Infraestructura desplegada correctamente
- Logs fluyendo a CloudWatch
- Alertas funcionando
- Dashboard visible
- Documentación completa de operaciones

---

## Backlog futuro (Sprint 4+)

- **US-007:** Explicabilidad SHAP avanzada con dashboard interactivo
- **US-008:** Reentrenamiento automático cuando se detecta model drift
- **US-009:** A/B testing de nuevas versiones del modelo
- **US-010:** API de análisis rápido del solicitante (enrichment con datos externos)
- **US-011:** Dashboard de desempeño para stakeholders no técnicos

---

## Criterios de definición de "Done" estándar

Todo User Story debe cumplir:

1. ✅ **Código**
   - Mergeado a `develop` (o `main` si es hotfix)
   - Sigue estándar PEP 8 / black formatting
   - Cobertura de tests ≥ 80%

2. ✅ **Testing**
   - Tests unitarios e integración pasando
   - Ejecutables con `pytest` sin warnings
   - Cases edge coverage

3. ✅ **Documentación**
   - Docstrings en funciones públicas
   - README o docs/ actualizado
   - Ejemplos de uso si aplica

4. ✅ **CI/CD**
   - Pipeline de CI pasando
   - No hay deuda técnica crítica (flake8, bandit)

5. ✅ **Revisión**
   - PR revisado y aprobado por mínimo 1 revisor
   - Feedback incorporado

---

## Notas de la gerencia de producto

- **Timeline:** Sprints de 2 semanas fijos; retrospectivas cada viernes
- **Prioridad:** Primero datos → Modelo → API → Deployment
- **Riesgos identificados:** 
  - Datos imbalanceados (mitiga: SMOTE)
  - Overfitting (mitiga: validación cruzada + early stopping)
  - Latencia API (mitiga: caching, model quantization)
- **Dependencias externas:** Dataset en Kaggle
- **Escalabilidad futura:** Soporte multi-modelo, A/B testing, reentrenamiento automático

---

**Historial de cambios:**

| Versión | Fecha      | Cambios                          |
|---------|------------|---------------------------------|
| 1.0     | 01/01/2024 | Backlog inicial (6 US / 3 sprints) |
