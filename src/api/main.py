"""API REST de predicción de aprobación de préstamos — FastAPI."""

import logging
import time
import os
import joblib
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from src.api.schemas import ClienteInput, PrediccionOutput
from src.data.preprocesamiento import preprocesar

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH    = Path(os.getenv("MODEL_PATH", "models/credit_model.pkl"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")

app = FastAPI(
    title="CreditML — Predicción de Aprobación de Préstamos",
    description="API de predicción de aprobación de solicitudes de préstamo. Proyecto académico MLOps.",
    version=MODEL_VERSION,
)

# Cargar modelo al arrancar (una sola vez)
try:
    modelo = joblib.load(MODEL_PATH)
    logger.info("Modelo cargado desde %s (versión %s)", MODEL_PATH, MODEL_VERSION)
except FileNotFoundError:
    logger.error("Modelo no encontrado en %s", MODEL_PATH)
    modelo = None


def _nivel_riesgo(proba: float) -> str:
    if proba >= 0.65:
        return "ALTO"
    elif proba >= 0.35:
        return "MEDIO"
    return "BAJO"


@app.get("/health")
def health_check():
    return {"status": "ok", "modelo_cargado": modelo is not None, "version": MODEL_VERSION}


@app.post("/predict", response_model=PrediccionOutput)
def predict(cliente: ClienteInput):
    if modelo is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    inicio = time.time()

    # Convertir input a DataFrame y preprocesar
    df_input = pd.DataFrame([cliente.model_dump()])
    try:
        df_proc = preprocesar(df_input)
        # Alinear columnas con las del modelo (puede haber dummies nuevos/faltantes)
        cols_modelo = modelo.feature_names_in_ if hasattr(modelo, "feature_names_in_") else df_proc.columns
        df_proc = df_proc.reindex(columns=cols_modelo, fill_value=0)
    except Exception as e:
        logger.error("Error en preprocesamiento: %s", e)
        raise HTTPException(status_code=422, detail=f"Error preprocesando datos: {e}")

    proba = float(modelo.predict_proba(df_proc)[0, 1])
    prediccion = proba >= 0.5
    nivel = _nivel_riesgo(proba)

    latencia_ms = (time.time() - inicio) * 1000
    logger.info(
        "Predicción: approved_proba=%.3f nivel=%s latencia=%.1fms",
        proba, nivel, latencia_ms,
    )

    return PrediccionOutput(
        approved_probability=round(proba, 4),
        approved_prediction=prediccion,
        risk_level=nivel,
        model_version=MODEL_VERSION,
    )