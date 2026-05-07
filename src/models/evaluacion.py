"""Umbrales de calidad del modelo."""

AUC_MINIMO = 0.78
F1_MINIMO  = 0.70


def supera_umbrales(auc: float, f1: float) -> bool:
    """Devuelve True si el modelo supera los umbrales mínimos para despliegue."""
    return auc >= AUC_MINIMO and f1 >= F1_MINIMO