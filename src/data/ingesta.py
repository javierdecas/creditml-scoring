"""Carga y validación inicial del dataset de credit approval."""

import logging
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW_PATH = Path("data/raw/credit_approval.csv")
VALIDATED_PATH = Path("data/validated/credit_approval_validated.csv")

EXPECTED_COLUMNS = {
    "age", "income", "employment_status", "loan_amount",
    "credit_history_length", "existing_debt", "payment_history", "approved",
}


def cargar_datos(ruta: Path = RAW_PATH) -> pd.DataFrame:
    logger.info("Cargando dataset desde %s", ruta)
    df = pd.read_csv(ruta)
    logger.info("Dataset cargado: %d filas, %d columnas", len(df), df.shape[1])
    return df


def validar_esquema(df: pd.DataFrame) -> bool:
    columnas_faltantes = EXPECTED_COLUMNS - set(df.columns)
    columnas_extra = set(df.columns) - EXPECTED_COLUMNS
    if columnas_faltantes or columnas_extra:
        if columnas_faltantes:
            logger.warning("Columnas faltantes en el dataset: %s", columnas_faltantes)
        if columnas_extra:
            logger.warning("Columnas extra en el dataset: %s", columnas_extra)
        return False

    logger.info("Esquema validado: todas las columnas presentes")
    return True


def validar_completitud(df: pd.DataFrame, umbral: float = 0.95) -> bool:
    completitud = 1 - df.isnull().to_numpy().mean()
    if completitud < umbral:
        logger.warning("Completitud general < %.0f%%: %.2f%%", umbral * 100, completitud * 100)
        return False

    logger.info("Completitud general OK: %.2f%% > %.0f%%", completitud * 100, umbral * 100)
    return True


def validar_tipos(df: pd.DataFrame) -> bool:
    numeric_cols = ["age", "income", "loan_amount", "credit_history_length", "existing_debt"]
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            logger.warning("'%s' debe ser numérico", col)
            return False
    if not pd.api.types.is_object_dtype(df["employment_status"]):
        logger.warning("'employment_status' debe ser categórico")
        return False
    if not pd.api.types.is_object_dtype(df["payment_history"]):
        logger.warning("'payment_history' debe ser categórico")
        return False
    if not pd.api.types.is_numeric_dtype(df["approved"]):
        logger.warning("'approved' debe ser numérico (0/1)")
        return False

    logger.info("Tipos de datos validados")
    return True


def guardar_validado(df: pd.DataFrame, ruta: Path | str = VALIDATED_PATH) -> None:
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ruta, index=False)
    logger.info("Dataset validado guardado en %s", ruta)


def main() -> None:
    df = cargar_datos()
    validar_esquema(df)
    validar_completitud(df)
    validar_tipos(df)
    guardar_validado(df)
    logger.info("Ingesta completada: %d registros validados", len(df))


if __name__ == "__main__":
    main()