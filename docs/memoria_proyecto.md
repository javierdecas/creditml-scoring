# Memoria Técnica — CreditML

## 1. Resumen

Este proyecto desarrolla un pipeline MLOps para la predicción de aprobación de préstamos. El objetivo es automatizar la ingestión de datos, el preprocesamiento, el entrenamiento del modelo, el despliegue de una API REST y la emulación de servicios cloud en local con LocalStack.

El sistema utiliza un dataset tabular de solicitudes de crédito con variables como `age`, `income`, `employment_status`, `loan_amount`, `credit_history_length`, `existing_debt` y `payment_history`. A partir de estas características se entrena un modelo de clasificación binaria que estima si una solicitud será aprobada o rechazada.

La solución se ha construido con Python, scikit-learn, FastAPI, Docker, MLflow, Terraform y LocalStack. Se ha diseñado para que sea reproducible en entornos locales y compatible con una futura transición a una infraestructura cloud real.

## 2. Arquitectura

### 2.1 Visión general

La arquitectura del proyecto sigue un patrón de pipeline clásico de MLOps dividido en capas:

- **Ingesta de datos**: carga y validación del dataset bruto.
- **Preprocesamiento**: limpieza, imputación y codificación de variables.
- **Entrenamiento**: modelo supervisado entrenado con validación cruzada y control de calidad.
- **Despliegue**: API REST containerizada que expone el servicio de predicción.
- **Infraestructura local**: emulación de servicios AWS con LocalStack para S3 y ECR.
- **Observabilidad**: métricas y health check de la API.

### 2.2 Componentes principales

- `src/data/ingesta.py`: carga CSV, valida esquema, completitud y tipos, y guarda datos validados.
- `src/data/preprocesamiento.py`: imputa nulos y realiza codificación one-hot de variables categóricas.
- `src/models/entrenamiento.py`: entrena un Random Forest, registra métricas en MLflow y guarda el modelo en `models/`.
- `src/api/main.py`: API FastAPI que expone `/health` y `/predict`.
- `docker/Dockerfile.api`: define la imagen de la API.
- `docker-compose.yml`: levanta la API y LocalStack.
- `infrastructure/main.tf`: provisiona recursos S3 y ECR en LocalStack.

### 2.3 LocalStack en el flujo

LocalStack se emplea como entorno de emulación cloud local. Sus beneficios para el proyecto son:

- Permite simular buckets S3 para datos y artefactos sin necesidad de AWS real.
- Emula un repositorio ECR para validar el build de imágenes Docker.
- Facilita el desarrollo y las pruebas de infraestructura como código en un entorno reproducible.

El `docker-compose.yml` levanta LocalStack en el puerto `4566`, y Terraform se configura para conectarse a ese endpoint en lugar de la nube real.

### 2.4 Diagrama lógico

1. El dataset se coloca en `data/raw/credit_approval.csv`.
2. `src/data/ingesta.py` valida y guarda en `data/validated/`.
3. `src/data/preprocesamiento.py` limpia y transforma en `data/processed/`.
4. `src/models/entrenamiento.py` entrena el modelo y lo guarda en `models/`.
5. `docker/Dockerfile.api` construye la imagen de la API.
6. `docker-compose.yml` ejecuta la API y LocalStack.
7. La API responde a `/health` y `/predict`.

## 3. Decisiones Técnicas

### 3.1 Selección de stack

- **Python 3.9**: compatibilidad amplia con bibliotecas de ciencia de datos y MLOps.
- **scikit-learn**: librería robusta para clasificación supervisada y pipelines de preprocesamiento.
- **FastAPI**: framework ligero y rápido para exponer la inferencia como servicio.
- **Docker / Docker Compose**: contenedorización y orquestación local.
- **MLflow**: seguimiento de experimentos y registro de modelos.
- **Terraform**: infraestructura como código, adaptada a LocalStack.
- **LocalStack**: emulación de servicios AWS para desarrollo local.

### 3.2 Diseño del modelo

Se seleccionó un **Random Forest** como primer modelo de producción por su equilibrio entre interpretabilidad, estabilidad y tolerancia a características heterogéneas. Los parámetros se fijaron en valores moderados para evitar overfitting en un dataset de tamaño académico.

La evaluación considera métricas de clasificación binaria relevantes para credit scoring:

- AUC-ROC
- F1-score
- Precision
- Recall

### 3.3 Preprocesamiento

La transformación de datos incluye:

- Imputación de valores faltantes con media en variables numéricas.
- Imputación de valores faltantes con moda en variables categóricas.
- One-hot encoding de `employment_status` y `payment_history`.

Esta aproximación garantiza que la API pueda recibir nuevos registros en formato tabular y aplicar la misma transformación que el modelo entrenado.

### 3.4 Infraestructura local y despliegue

El uso de LocalStack se justificó por:

- la posibilidad de practicar IaC con Terraform sin coste cloud,
- la capacidad de generar buckets S3 y repositorios ECR locales,
- la preservación de la arquitectura de un pipeline MLOps real.

Terraform se configuró con endpoints locales y opciones de validación de credenciales deshabilitadas, lo que permite ejecutar `terraform init`, `plan` y `apply` contra LocalStack.

### 3.5 Pruebas y calidad

Se diseñaron tests con `pytest` para:

- validar la función de preprocesamiento con datos sintéticos,
- verificar el comportamiento de la API en `/health` y `/predict`,
- asegurar la robustez frente a entradas inválidas y errores de preprocesamiento.

Los tests utilizan mocks para evitar la dependencia de archivos o modelos reales durante la ejecución.

## 4. Metodología Ágil

### 4.1 Organización del trabajo

El desarrollo se planteó en sprints cortos de dos semanas, con una priorización clara:

- Sprint 1: ingesta, preprocesamiento y feature engineering.
- Sprint 2: entrenamiento del modelo, API y tests.
- Sprint 3: CI/CD, LocalStack e infraestructura.

Se trabajó siguiendo principios de Scrum ligero, con entregas incrementales y revisiones periódicas de avance.

### 4.2 User stories y enfoque iterativo

El backlog se estructuró en historias de usuario que cubren:

- la validación de datos,
- la calidad del modelo,
- la interfaz de predicción,
- la automatización de tests,
- el despliegue local.

Cada historia definió criterios de aceptación claros y una definición de "done" que incluye código, tests y documentación.

### 4.3 Integración continua

Se configuró un pipeline de CI en GitHub Actions con las siguientes etapas:

- checkout del código,
- instalación de dependencias,
- linting con `flake8`,
- ejecución de tests con `pytest` y cobertura,
- build de la imagen Docker de la API.

Este pipeline permite detectar errores tempranos y mantener la integridad del repositorio en cada push a `main`.

### 4.4 Lecciones aprendidas

- La emulación con LocalStack facilita la práctica de MLOps sin necesidad de recursos cloud reales.
- Es crucial definir un contrato claro entre preprocesamiento y API para evitar desviaciones entre entrenamiento e inferencia.
- La automatización de tests en la API mejora la confianza en el servicio de inferencia.

## 5. Conclusión

El proyecto demuestra un pipeline MLOps local completo para credit scoring. Se ha logrado una solución reproducible que integra datos, modelo, API y emulación cloud. La arquitectura permite evolucionar hacia un entorno cloud real en el futuro, manteniendo la trazabilidad de experimentos y la calidad de código en el núcleo de la solución.

---

## 6. Referencias

- FastAPI, documentación oficial.
- scikit-learn, documentación de Random Forest y métricas de clasificación.
- MLflow, tracking y model registry.
- LocalStack, emulación de AWS para desarrollo local.
- Terraform, infraestructura como código.
