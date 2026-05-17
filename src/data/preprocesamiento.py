"""Limpieza y transformaciones del dataset de credit approval."""

import logging
import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

INPUT_PATH = Path("data/validated/credit_approval_validated.csv")
OUTPUT_PATH = Path("data/processed/credit_approval_processed.csv")

NUMERIC_COLS = ["age", "income", "loan_amount", "credit_history_length", "existing_debt"]
CATEGORICAL_COLS = ["employment_status", "payment_history"]
COLS_TO_DROP = []  # Agregar si hay columnas innecesarias como ID


def imputar_nulos(df: pd.DataFrame) -> pd.DataFrame:
    """Imputa valores faltantes: media para numéricos, moda para categóricos."""
    df = df.copy()
    # Numéricos
    imputer_num = SimpleImputer(strategy="mean")
    df[NUMERIC_COLS] = imputer_num.fit_transform(df[NUMERIC_COLS])
    # Categóricos
    imputer_cat = SimpleImputer(strategy="most_frequent")
    df[CATEGORICAL_COLS] = imputer_cat.fit_transform(df[CATEGORICAL_COLS])
    n_nulos_total = df.isnull().sum().sum()
    if n_nulos_total > 0:
        logger.warning("Quedan %d valores nulos después de imputación", n_nulos_total)
    else:
        logger.info("Imputación completada: no hay valores nulos")
    return df


def codificar_categoricas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
    logger.info("Variables categóricas one-hot encoded: %s", CATEGORICAL_COLS)
    return df


def eliminar_columnas(df: pd.DataFrame, cols: list = COLS_TO_DROP) -> pd.DataFrame:
    return df.drop(columns=[c for c in cols if c in df.columns])


def preprocesar(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Iniciando preprocesamiento: %d filas", len(df))
    df = imputar_nulos(df)
    df = codificar_categoricas(df)
    df = eliminar_columnas(df)
    logger.info("Preprocesamiento completo: %d filas, %d columnas", df.shape[0], df.shape[1])
    return df


def main() -> None:
    df_raw = pd.read_csv(INPUT_PATH)
    df_proc = preprocesar(df_raw)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_proc.to_csv(OUTPUT_PATH, index=False)
    logger.info("Guardado en %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
