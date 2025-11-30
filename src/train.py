"""
Обучение моделей: логистическая регрессия, random forest и т. п.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score
from pathlib import Path
import joblib
import yaml
import mlflow
import mlflow.sklearn


def prepare_features(df: pd.DataFrame, target_column: str):
    """Разделение на признаки и целевую переменную"""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

def save_model(model, model_name: str, output_dir: str):
    """Сохранение обученной модели."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    model_path = output_path / f'{model_name}.pkl'
    joblib.dump(model, model_path)
    print(f"\nМодель сохранена: {model_path}")
    return model_path


def main():
    base_dir = Path(__file__).parent.parent
    
    # Настройка MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("titanic-survival-prediction")
    
    params_path = base_dir / 'params.yaml'
    with open(params_path, 'r') as f:
        params = yaml.safe_load(f)
        train_params = params['train']
    
    train_df = pd.read_csv(base_dir / 'data/processed/train.csv')
    test_df = pd.read_csv(base_dir / 'data/processed/test.csv')
    
    # Подготовка признаков
    X_train, y_train = prepare_features(train_df, 'Survived')
    X_test, y_test = prepare_features(test_df, 'Survived')
        
    # Обучение Random Forest
    with mlflow.start_run(run_name="random_forest"):
        mlflow.log_param("model", "RandomForest")
        mlflow.log_param("n_estimators", train_params['n_estimators'])
        mlflow.log_param("random_state", train_params['random_state'])
        
        rf_model = RandomForestClassifier(
            n_estimators=train_params['n_estimators'],
            random_state=train_params['random_state']
        )
        rf_model.fit(X_train, y_train)
        
        # Предсказания
        y_test_pred_rf = rf_model.predict(X_test)
        
        # Метрики
        rf_accuracy = accuracy_score(y_test, y_test_pred_rf)
        
        # Логирование метрик
        mlflow.log_metric("rf_test_accuracy", rf_accuracy)
        
        # Сохранение модели
        model_path = base_dir / 'model.pkl'
        joblib.dump(rf_model, model_path)
        mlflow.log_artifact(str(model_path))
        
if __name__ == '__main__':
    main()

