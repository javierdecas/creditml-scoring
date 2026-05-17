"""Entrenamiento del modelo con MLflow tracking."""

import logging
import mlflow
import mlflow.sklearn
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, f1_score, classification_report

from src.models.evaluacion import supera_umbrales

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = Path("data/processed/credit_approval_processed.csv")
MODEL_PATH = Path("models/credit_model.pkl")

MLFLOW_EXPERIMENT = "creditml-scoring"
TARGET_COL = "approved"

HIPERPARAMETROS = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "random_state": 42,
    "n_jobs": -1,
}


def cargar_datos(ruta: Path = DATA_PATH):
    df = pd.read_csv(ruta)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y


def entrenar(X_train, y_train, params: dict | None = None):
    if params is None:
        params = HIPERPARAMETROS

    with mlflow.start_run(nested=True):
        for nombre, valor in params.items():
            mlflow.log_param(nombre, valor)

        modelo = RandomForestClassifier(**params)
        modelo.fit(X_train, y_train)

        if len(set(y_train)) > 1:
            y_pred_proba = modelo.predict_proba(X_train)[:, 1]
            y_pred = modelo.predict(X_train)
            auc = roc_auc_score(y_train, y_pred_proba)
            f1 = f1_score(y_train, y_pred)
            mlflow.log_metric("auc", auc)
            mlflow.log_metric("f1", f1)
        else:
            logger.warning("No hay suficientes clases para calcular métricas AUC/F1.")

        mlflow.sklearn.log_model(modelo, artifact_path="model")

    return modelo


def main() -> None:
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    X, y = cargar_datos()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info("Train: %d | Test: %d", len(X_train), len(X_test))

    with mlflow.start_run(run_name="random-forest-v1"):
        # Log parámetros
        mlflow.log_params(HIPERPARAMETROS)

        # Entrenar
        modelo = entrenar(X_train, y_train, HIPERPARAMETROS)

        # Evaluar
        y_pred_proba = modelo.predict_proba(X_test)[:, 1]
        y_pred = modelo.predict(X_test)

        auc = roc_auc_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_metric("auc_roc", auc)
        mlflow.log_metric("f1_score", f1)

        # Cross-validation
        cv_auc = cross_val_score(modelo, X, y, cv=5, scoring="roc_auc").mean()
        mlflow.log_metric("cv_auc_mean", cv_auc)

        logger.info("AUC-ROC: %.4f | F1: %.4f | CV-AUC: %.4f", auc, f1, cv_auc)
        logger.info("\n%s", classification_report(y_test, y_pred))

        # Guardar modelo
        mlflow.sklearn.log_model(modelo, artifact_path="model")
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(modelo, MODEL_PATH)
        mlflow.log_artifact(str(MODEL_PATH))

        # Gate de calidad — solo pasa a CD si supera umbrales
        if not supera_umbrales(auc=auc, f1=f1):
            raise ValueError(
                f"Modelo no supera umbrales mínimos (AUC={auc:.3f}, F1={f1:.3f}). "
                "No se procede al despliegue."
            )

        logger.info("Modelo aprobado para despliegue.")


if __name__ == "__main__":
    main()
