"""Tests para módulo entrenamiento.py - entrenamiento del modelo."""

import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from sklearn.ensemble import RandomForestClassifier
from src.models.entrenamiento import cargar_datos, entrenar, main


class TestEntrenar:
    """Tests para función entrenar."""

    def test_entrenar_retorna_modelo(self):
        """Test que entrenar retorna un modelo RandomForest."""
        X = pd.DataFrame({
            'age': [25, 35, 45, 55],
            'income': [40000, 50000, 60000, 70000],
            'loan_amount': [20000, 25000, 30000, 35000],
            'credit_history_length': [5, 8, 10, 12],
            'existing_debt': [5000, 3000, 2000, 1000],
        })
        y = pd.Series([1, 0, 1, 0])

        with patch('mlflow.log_param'):
            with patch('mlflow.log_metric'):
                with patch('mlflow.sklearn.log_model'):
                    modelo = entrenar(X, y)

        assert isinstance(modelo, RandomForestClassifier)

    def test_entrenar_registra_mlflow_params(self):
        """Test que entrenar registra parámetros en MLflow."""
        X = pd.DataFrame({
            'age': [25, 35],
            'income': [40000, 50000],
            'loan_amount': [20000, 25000],
            'credit_history_length': [5, 8],
            'existing_debt': [5000, 3000],
        })
        y = pd.Series([1, 0])

        with patch('mlflow.log_param') as mock_log_param:
            with patch('mlflow.log_metric'):
                with patch('mlflow.sklearn.log_model'):
                    entrenar(X, y)

            # Verificar que se registraron parámetros del modelo
            param_calls = [c[0][0] for c in mock_log_param.call_args_list]
            assert 'n_estimators' in param_calls or any('param' in str(c) for c in param_calls)

    def test_entrenar_registra_mlflow_metricas(self):
        """Test que entrenar registra métricas (AUC, F1) en MLflow."""
        X = pd.DataFrame({
            'age': [25, 35, 45, 55],
            'income': [40000, 50000, 60000, 70000],
            'loan_amount': [20000, 25000, 30000, 35000],
            'credit_history_length': [5, 8, 10, 12],
            'existing_debt': [5000, 3000, 2000, 1000],
        })
        y = pd.Series([1, 0, 1, 0])

        with patch('mlflow.log_param'):
            with patch('mlflow.log_metric') as mock_log_metric:
                with patch('mlflow.sklearn.log_model'):
                    entrenar(X, y)

            # Verificar que se registraron métricas
            metric_names = [c[0][0] for c in mock_log_metric.call_args_list]
            assert any('auc' in m.lower() or 'f1' in m.lower() for m in metric_names)

    def test_entrenar_con_datos_desbalanceados(self):
        """Test entrenar con datos muy desbalanceados."""
        # Mayoría de clase 1, minoría de clase 0
        X = pd.DataFrame({
            'age': [25, 35, 45, 55, 65, 75],
            'income': [40000, 50000, 60000, 70000, 80000, 90000],
            'loan_amount': [20000, 25000, 30000, 35000, 40000, 45000],
            'credit_history_length': [5, 8, 10, 12, 15, 18],
            'existing_debt': [5000, 3000, 2000, 1000, 500, 100],
        })
        y = pd.Series([1, 1, 1, 1, 0, 0])

        with patch('mlflow.log_param'):
            with patch('mlflow.log_metric'):
                with patch('mlflow.sklearn.log_model'):
                    modelo = entrenar(X, y)

        assert isinstance(modelo, RandomForestClassifier)
        # Verificar que el modelo hace predicciones
        pred = modelo.predict(X)
        assert len(pred) == len(X)

    def test_entrenar_genera_predicciones(self):
        """Test que el modelo entrenado genera predicciones."""
        X = pd.DataFrame({
            'age': [25, 35, 45, 55],
            'income': [40000, 50000, 60000, 70000],
            'loan_amount': [20000, 25000, 30000, 35000],
            'credit_history_length': [5, 8, 10, 12],
            'existing_debt': [5000, 3000, 2000, 1000],
        })
        y = pd.Series([1, 0, 1, 0])

        with patch('mlflow.log_param'):
            with patch('mlflow.log_metric'):
                with patch('mlflow.sklearn.log_model'):
                    modelo = entrenar(X, y)

        # Generar predicciones
        prob = modelo.predict_proba(X)
        assert prob.shape == (4, 2)
        assert all(0 <= p <= 1 for p in prob.flatten())

    def test_entrenar_mantiene_feature_names(self):
        """Test que el modelo entrenado guarde nombres de features."""
        feature_names = ['age', 'income', 'loan_amount', 'credit_history_length', 'existing_debt']
        X = pd.DataFrame({col: [25, 35, 45, 55] for col in feature_names})
        y = pd.Series([1, 0, 1, 0])

        with patch('mlflow.log_param'):
            with patch('mlflow.log_metric'):
                with patch('mlflow.sklearn.log_model'):
                    modelo = entrenar(X, y)

        # RandomForest automáticamente guarda feature_names_in_
        assert hasattr(modelo, 'feature_names_in_')
        assert list(modelo.feature_names_in_) == feature_names

    def test_cargar_datos_lee_csv(self):
        df = pd.DataFrame({
            'age': [25],
            'income': [40000],
            'loan_amount': [20000],
            'credit_history_length': [5],
            'existing_debt': [5000],
            'approved': [1]
        })

        with patch('src.models.entrenamiento.pd.read_csv', return_value=df) as mock_read:
            X, y = cargar_datos(Path('dummy.csv'))

        mock_read.assert_called_once_with(Path('dummy.csv'))
        assert list(X.columns) == ['age', 'income', 'loan_amount', 'credit_history_length', 'existing_debt']
        assert y.name == 'approved'

    def test_main_entrena_y_guarda_modelo(self):
        X = pd.DataFrame({
            'age': list(range(10)),
            'income': [40000 + i * 1000 for i in range(10)],
            'loan_amount': [20000 + i * 500 for i in range(10)],
            'credit_history_length': [5, 8, 10, 12, 6, 15, 7, 9, 11, 13],
            'existing_debt': [5000, 3000, 2000, 1000, 600, 700, 800, 900, 1000, 1100],
        })
        y = pd.Series([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], name='approved')

        with patch('src.models.entrenamiento.cargar_datos', return_value=(X, y)) as mock_cargar:
            with patch('src.models.entrenamiento.train_test_split', return_value=(X.iloc[:8], X.iloc[8:], y.iloc[:8], y.iloc[8:])) as mock_split:
                with patch('src.models.entrenamiento.cross_val_score', return_value=np.array([0.9, 0.9, 0.9, 0.9, 0.9])) as mock_cv:
                    with patch('src.models.entrenamiento.mlflow.set_experiment') as mock_set_exp:
                        cm = MagicMock()
                        cm.__enter__.return_value = cm
                        cm.__exit__.return_value = None
                        with patch('src.models.entrenamiento.mlflow.start_run', return_value=cm):
                            with patch('src.models.entrenamiento.mlflow.log_params') as mock_log_params:
                                with patch('src.models.entrenamiento.mlflow.log_metric') as mock_log_metric:
                                    with patch('src.models.entrenamiento.mlflow.sklearn.log_model') as mock_log_model:
                                        with patch('src.models.entrenamiento.mlflow.log_artifact') as mock_log_artifact:
                                            with patch('src.models.entrenamiento.joblib.dump') as mock_joblib_dump:
                                                with patch('src.models.entrenamiento.Path.mkdir'):
                                                    with patch('src.models.entrenamiento.supera_umbrales', return_value=True) as mock_supera:
                                                        main()

        mock_cargar.assert_called_once()
        mock_split.assert_called_once()
        mock_cv.assert_called_once()
        mock_set_exp.assert_called_once()
        mock_log_params.assert_called_once()
        assert mock_log_metric.call_count >= 2
        mock_log_model.assert_called()
        mock_joblib_dump.assert_called_once()
        mock_log_artifact.assert_called_once()
        mock_supera.assert_called_once()
