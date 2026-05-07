"""Script para generar dataset sintético de credit approval."""

import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

# Configuración
n_records = 1000
output_path = Path("data/raw/credit_approval.csv")

# Generar datos
data = {
    "age": np.random.randint(18, 80, n_records),
    "income": np.random.normal(50000, 20000, n_records).astype(int),
    "employment_status": np.random.choice(["employed", "unemployed", "self-employed"], n_records),
    "loan_amount": np.random.normal(30000, 15000, n_records).astype(int),
    "credit_history_length": np.random.randint(0, 30, n_records),
    "existing_debt": np.random.normal(10000, 8000, n_records).astype(int),
    "payment_history": np.random.choice(["poor", "fair", "good", "excellent"], n_records),
}

df = pd.DataFrame(data)

# Generar variable objetivo (approval) basada en lógica simplificada
df["approved"] = (
    (df["age"] >= 25) & 
    (df["income"] > 30000) & 
    (df["loan_amount"] < df["income"]) &
    (df["payment_history"].isin(["good", "excellent"]))
).astype(int)

# Agregar ruido (algumas aprobaciones/rechazos aleatorios para no ser perfectamente determinístico)
noise_idx = np.random.choice(df.index, size=int(0.1 * len(df)), replace=False)
df.loc[noise_idx, "approved"] = 1 - df.loc[noise_idx, "approved"]

# Crear directorio si no existe
output_path.parent.mkdir(parents=True, exist_ok=True)

# Guardar
df.to_csv(output_path, index=False)
print(f"Dataset generado: {output_path}")
print(f"Registros: {len(df)}")
print(f"Distribución de aprobaciones:\n{df['approved'].value_counts()}")
print(f"\nPrimeras filas:\n{df.head()}")
