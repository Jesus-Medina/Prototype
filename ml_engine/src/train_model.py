import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
from pathlib import Path

def train_model():
    # Cargar datos etiquetados (solo para entrenamiento)
    data_path = Path(__file__).parent.parent / "data" / "processed" / "dwdm_labeled.csv"
    df = pd.read_csv(data_path)
    
    # Features y target
    # ml_engine/src/train_model.py
    features = [
    'potencia_dbm', 
    'ber', 
    'osnr', 
    'temperatura', 
    'edad_cable'  
    ]
    X = df[features]
    y = df['estado']
    
    # Entrenar modelo
    # Cambia el clasificador para priorizar casos críticos:
    model = RandomForestClassifier(
        n_estimators=200,
        class_weight={"critical": 10, "warning": 2, "safe": 1},  # Peso 5x mayor para críticos
        random_state=42
    )
    model.fit(X, y)
    
    # Guardar modelo
    model_path = Path(__file__).parent.parent / "models" / "fiber_model.pkl"
    model_path.parent.mkdir(exist_ok=True)
    joblib.dump(model, model_path)
    print(f"✅ Modelo entrenado y guardado en: {model_path}")

if __name__ == "__main__":
    train_model()