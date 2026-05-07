"""Tests para módulo ingesta.py - carga y validación de datos."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock
from src.data.ingesta import (
    cargar_datos,
    validar_esquema,
    validar_completitud,
    validar_tipos,
    guardar_validado,
)


class TestCargarDatos:
    """Tests para función cargar_datos."""

    def test_cargar_datos_exitoso(self):
        """Test carga exitosa de CSV."""
        csv_content = "age,income,loan_amount,credit_history_length,existing_debt,employment_status,payment_history,approved\n25,40000,20000,5,5000,employed,good,1"
        with patch("builtins.open", mock_open(read_data=csv_content)):
            with patch("src.data.ingesta.pd.read_csv") as mock_read:
                mock_read.return_value = pd.DataFrame({
                    'age': [25],
                    'income': [40000],
                    'loan_amount': [20000],
                    'credit_history_length': [5],
                    'existing_debt': [5000],
                    'employment_status': ['employed'],
                    'payment_history': ['good'],
                    'approved': [1]
                })
                df = cargar_datos("data/raw/test.csv")
                assert len(df) == 1
                assert list(df.columns) == [
                    'age', 'income', 'loan_amount', 'credit_history_length',
                    'existing_debt', 'employment_status', 'payment_history', 'approved'
                ]

    def test_cargar_datos_archivo_no_existe(self):
        """Test error cuando archivo no existe."""
        with patch("src.data.ingesta.pd.read_csv", side_effect=FileNotFoundError("File not found")):
            with pytest.raises(FileNotFoundError):
                cargar_datos("data/raw/inexistente.csv")


class TestValidarEsquema:
    """Tests para función validar_esquema."""

    def test_esquema_valido(self):
        """Test validación con esquema correcto."""
        df = pd.DataFrame({
            'age': [25, 35],
            'income': [40000, 50000],
            'loan_amount': [20000, 25000],
            'credit_history_length': [5, 8],
            'existing_debt': [5000, 3000],
            'employment_status': ['employed', 'unemployed'],
            'payment_history': ['good', 'fair'],
            'approved': [1, 0]
        })
        assert validar_esquema(df) is True

    def test_esquema_falta_columna(self):
        """Test validación falla si falta columna."""
        df = pd.DataFrame({
            'age': [25],
            'income': [40000],
            # Faltan otras columnas
        })
        assert validar_esquema(df) is False

    def test_esquema_columnas_extra(self):
        """Test validación falla si hay columnas extra."""
        df = pd.DataFrame({
            'age': [25],
            'income': [40000],
            'loan_amount': [20000],
            'credit_history_length': [5],
            'existing_debt': [5000],
            'employment_status': ['employed'],
            'payment_history': ['good'],
            'approved': [1],
            'columna_extra': [999]  # Columna no esperada
        })
        assert validar_esquema(df) is False


class TestValidarCompletitud:
    """Tests para función validar_completitud."""

    def test_completitud_alta(self):
        """Test validación exitosa con alto nivel de completitud."""
        df = pd.DataFrame({
            'age': [25, 35, np.nan],
            'income': [40000, 50000, 60000],
            'loan_amount': [20000, 25000, 30000],
            'credit_history_length': [5, 8, 10],
            'existing_debt': [5000, 3000, 2000],
            'employment_status': ['employed', 'unemployed', 'employed'],
            'payment_history': ['good', 'fair', 'excellent'],
            'approved': [1, 0, 1]
        })
        # 1 valor nulo de 8 columnas * 3 filas = 24 valores = 23/24 = 95.8% (> 95%)
        assert validar_completitud(df) is True

    def test_completitud_baja(self):
        """Test validación falla con bajo nivel de completitud."""
        df = pd.DataFrame({
            'age': [25, np.nan, np.nan],
            'income': [40000, np.nan, np.nan],
            'loan_amount': [20000, np.nan, np.nan],
            'credit_history_length': [5, np.nan, np.nan],
            'existing_debt': [5000, np.nan, np.nan],
            'employment_status': ['employed', np.nan, np.nan],
            'payment_history': ['good', np.nan, np.nan],
            'approved': [1, np.nan, np.nan]
        })
        # 16 valores nulos de 24 valores = 8/24 = 33.3% (< 95%)
        assert validar_completitud(df) is False


class TestValidarTipos:
    """Tests para función validar_tipos."""

    def test_tipos_correctos(self):
        """Test validación con tipos correctos."""
        df = pd.DataFrame({
            'age': [25, 35],
            'income': [40000.0, 50000.0],
            'loan_amount': [20000.0, 25000.0],
            'credit_history_length': [5, 8],
            'existing_debt': [5000.0, 3000.0],
            'employment_status': ['employed', 'unemployed'],
            'payment_history': ['good', 'fair'],
            'approved': [1, 0]
        })
        assert validar_tipos(df) is True

    def test_tipos_incorrectos(self):
        """Test validación falla con tipos incorrectos."""
        df = pd.DataFrame({
            'age': ['25', '35'],  # string en lugar de int/float
            'income': [40000.0, 50000.0],
            'loan_amount': [20000.0, 25000.0],
            'credit_history_length': [5, 8],
            'existing_debt': [5000.0, 3000.0],
            'employment_status': ['employed', 'unemployed'],
            'payment_history': ['good', 'fair'],
            'approved': [1, 0]
        })
        # age es object (string), no numérico
        assert validar_tipos(df) is False


class TestGuardarValidado:
    """Tests para función guardar_validado."""

    def test_guardar_validado_exitoso(self):
        """Test guardado exitoso de datos validados."""
        df = pd.DataFrame({
            'age': [25],
            'income': [40000],
            'loan_amount': [20000],
            'credit_history_length': [5],
            'existing_debt': [5000],
            'employment_status': ['employed'],
            'payment_history': ['good'],
            'approved': [1]
        })
        with patch("pathlib.Path.mkdir"):
            with patch("src.data.ingesta.pd.DataFrame.to_csv") as mock_to_csv:
                guardar_validado(df, "data/validated/test.csv")
                mock_to_csv.assert_called_once()

    def test_guardar_validado_crea_directorio(self):
        """Test que guardarValidado crea directorio si no existe."""
        df = pd.DataFrame({'age': [25]})
        with patch("pathlib.Path.mkdir") as mock_mkdir:
            with patch("src.data.ingesta.pd.DataFrame.to_csv"):
                guardar_validado(df, "data/validated/test.csv")
                # Verifica que mkdir fue llamado (parents=True, exist_ok=True)
                mock_mkdir.assert_called_once()
