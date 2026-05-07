import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from src.data.preprocesamiento import preprocesar, imputar_nulos, codificar_categoricas


class TestImputarNulos:
    def test_imputar_nulos_numericos(self):
        """Test imputación de nulos en variables numéricas con media."""
        df = pd.DataFrame({
            'age': [25, np.nan, 35],
            'income': [40000, 50000, np.nan],
            'loan_amount': [20000, 25000, 30000],
            'credit_history_length': [5, 10, 8],
            'existing_debt': [5000, 3000, 2000],
            'employment_status': ['employed', 'unemployed', 'employed'],
            'payment_history': ['good', 'fair', 'excellent']
        })
        result = imputar_nulos(df)
        assert result['age'].isna().sum() == 0
        assert result['income'].isna().sum() == 0
        assert result['age'].iloc[1] == 30.0  # Media de 25 y 35
        assert result['income'].iloc[2] == 45000.0  # Media de 40000 y 50000

    def test_imputar_nulos_categoricos(self):
        """Test imputación de nulos en variables categóricas con moda."""
        df = pd.DataFrame({
            'age': [25, 30, 35],
            'income': [40000, 50000, 60000],
            'loan_amount': [20000, 25000, 30000],
            'credit_history_length': [5, 10, 8],
            'existing_debt': [5000, 3000, 2000],
            'employment_status': ['employed', np.nan, 'employed'],
            'payment_history': ['good', 'fair', np.nan]
        })
        result = imputar_nulos(df)
        assert result['employment_status'].isna().sum() == 0
        assert result['payment_history'].isna().sum() == 0
        assert result['employment_status'].iloc[1] == 'employed'  # Moda


class TestCodificarCategoricas:
    def test_codificar_categoricas(self):
        """Test one-hot encoding de variables categóricas."""
        df = pd.DataFrame({
            'age': [25, 30, 35],
            'income': [40000, 50000, 60000],
            'loan_amount': [20000, 25000, 30000],
            'credit_history_length': [5, 10, 8],
            'existing_debt': [5000, 3000, 2000],
            'employment_status': ['employed', 'unemployed', 'employed'],
            'payment_history': ['good', 'fair', 'excellent']
        })
        result = codificar_categoricas(df)
        # Verificar que se crearon dummies (drop_first=True elimina la primera categoría)
        assert 'employment_status_unemployed' in result.columns or 'employment_status_employed' in result.columns
        assert 'payment_history_fair' in result.columns or 'payment_history_good' in result.columns
        # Verificar que columnas originales fueron eliminadas
        assert 'employment_status' not in result.columns
        assert 'payment_history' not in result.columns


class TestPreprocesar:
    def test_preprocesar_completo(self):
        """Test función preprocesar completa."""
        df = pd.DataFrame({
            'age': [25, np.nan, 35],
            'income': [40000, 50000, np.nan],
            'loan_amount': [20000, 25000, 30000],
            'credit_history_length': [5, 10, 8],
            'existing_debt': [5000, 3000, 2000],
            'employment_status': ['employed', 'unemployed', 'employed'],
            'payment_history': ['good', 'fair', 'excellent']
        })
        result = preprocesar(df)
        # Verificar que no hay nulos
        assert result.isna().sum().sum() == 0
        # Verificar que hay columnas dummy para categorías (con drop_first=True habrá menos categorías)
        dummy_cols = [col for col in result.columns if any(cat + '_' in col for cat in ['employment_status', 'payment_history'])]
        assert len(dummy_cols) > 0, f"No se encontraron columnas dummy. Columnas presentes: {list(result.columns)}"
        assert any('payment_history_' in col for col in result.columns)
        # Verificar que columnas originales numéricas están presentes
        assert 'age' in result.columns
        assert 'income' in result.columns

    def test_preprocesar_sin_cambios(self):
        """Test preprocesar con datos ya limpios."""
        df = pd.DataFrame({
            'age': [25, 30, 35],
            'income': [40000, 50000, 60000],
            'loan_amount': [20000, 25000, 30000],
            'credit_history_length': [5, 10, 8],
            'existing_debt': [5000, 3000, 2000],
            'employment_status': ['employed', 'unemployed', 'employed'],
            'payment_history': ['good', 'fair', 'excellent']
        })
        result = preprocesar(df)
        # Debería crear dummies igual
        assert any('employment_status_' in col for col in result.columns)
        assert result.shape[1] > df.shape[1]  # Más columnas por dummies