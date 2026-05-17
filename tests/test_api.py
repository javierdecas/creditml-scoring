import pytest
import pandas as pd
from unittest.mock import patch
from fastapi.testclient import TestClient
from src.api.main import app


@pytest.fixture
def client():
    """Fixture para TestClient de FastAPI."""
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_ok(self, client):
        """Test /health devuelve estado ok."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "modelo_cargado" in data


class TestPredictEndpoint:
    def test_predict_datos_validos(self, client):
        """Test /predict con datos válidos y modelo mockeado."""
        # Patchear en el módulo main
        with patch('src.api.main.preprocesar') as mock_preprocesar:

            # Configurar mocks
            mock_preprocesar.return_value = pd.DataFrame({'dummy': [1]})

            data = {
                "age": 35,
                "income": 45000.0,
                "employment_status": "employed",
                "loan_amount": 25000.0,
                "credit_history_length": 8,
                "existing_debt": 5000.0,
                "payment_history": "excellent"
            }
            response = client.post("/predict", json=data)
            assert response.status_code == 200
            result = response.json()
            assert "approved_probability" in result
            assert "approved_prediction" in result
            assert "risk_level" in result
            assert "model_version" in result

    def test_predict_datos_invalidos(self, client):
        """Test /predict con datos inválidos (edad negativa)."""
        data = {
            "age": -5,  # Inválido
            "income": 45000.0,
            "employment_status": "employed",
            "loan_amount": 25000.0,
            "credit_history_length": 8,
            "existing_debt": 5000.0,
            "payment_history": "excellent"
        }
        response = client.post("/predict", json=data)
        assert response.status_code == 422  # Validación Pydantic

    def test_predict_error_preprocesamiento(self, client):
        """Test /predict con error en preprocesamiento."""
        with patch('src.api.main.preprocesar', side_effect=Exception("Error de preprocesamiento")):

            data = {
                "age": 35,
                "income": 45000.0,
                "employment_status": "employed",
                "loan_amount": 25000.0,
                "credit_history_length": 8,
                "existing_debt": 5000.0,
                "payment_history": "excellent"
            }
            response = client.post("/predict", json=data)
            assert response.status_code == 422
            assert "Error preprocesando datos" in response.json()["detail"]
