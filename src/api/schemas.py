"""Schemas de entrada y salida de la API."""

from pydantic import BaseModel, Field


class ClienteInput(BaseModel):
    age: int = Field(..., ge=18, le=100, description="Edad del solicitante")
    income: float = Field(..., gt=0, description="Ingreso anual en €")
    employment_status: str = Field(..., description="Estado laboral: employed, unemployed, self-employed, etc.")
    loan_amount: float = Field(..., gt=0, description="Monto del préstamo solicitado en €")
    credit_history_length: int = Field(..., ge=0, description="Años de historial crediticio")
    existing_debt: float = Field(..., ge=0, description="Deuda existente en €")
    payment_history: str = Field(..., description="Historial de pagos: excellent, good, fair, poor")

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 35,
                "income": 45000.0,
                "employment_status": "employed",
                "loan_amount": 25000.0,
                "credit_history_length": 8,
                "existing_debt": 5000.0,
                "payment_history": "excellent",
            }
        }
    }


class PrediccionOutput(BaseModel):
    approved_probability: float = Field(..., description="Probabilidad de aprobación (0-1)")
    approved_prediction: bool = Field(..., description="True si la solicitud es aprobada")
    risk_level: str = Field(..., description="ALTO | MEDIO | BAJO")
    model_version: str = Field(..., description="Versión del modelo usado")
