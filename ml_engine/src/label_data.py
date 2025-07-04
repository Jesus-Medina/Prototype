import pandas as pd
import numpy as np
from pathlib import Path

def label_data(raw_data_path: str, labeled_data_path: str):
    """
    Añade etiquetas 'estado' basadas en reglas técnicas DWDM mejoradas.
    Incluye potencia, BER, OSNR y temperatura con umbrales configurables.
    """
    # Cargar datos
    df = pd.read_csv(raw_data_path)
    
    # Umbrales técnicos configurables (ajusta según especificaciones de tu equipo DWDM)
    UMBRALES = {
        'potencia': {
            'safe': -18,      # dBm (potencia óptima)
            'critical': -30    # dBm (pérdida extrema)
        },
        'ber': {
            'safe': 1e-12,    # BER óptimo
            'critical': 1e-8   # BER inaceptable
        },
        'osnr': {
            'safe': 30,        # dB (OSNR mínimo recomendado)
            'critical': 25     # dB (OSNR peligroso)
        },
        'temperatura': {
            'safe': 70,        # °C (temperatura máxima normal)
            'critical': 85     # °C (temperatura de riesgo)
        }
    }
    
    # Condiciones MEJORADAS (sin solapamientos)
    conditions = [
        # 1. CRÍTICO: CUALQUIER métrica en zona crítica
        (df['potencia_dbm'] < UMBRALES['potencia']['critical']) |
        (df['ber'] > UMBRALES['ber']['critical']) |
        (df['osnr'] < UMBRALES['osnr']['critical']) |
        (df['temperatura'] > UMBRALES['temperatura']['critical']),
        
        # 2. SEGURO: TODAS las métricas en zona segura
        (df['potencia_dbm'] >= UMBRALES['potencia']['safe']) &
        (df['ber'] <= UMBRALES['ber']['safe']) &
        (df['osnr'] >= UMBRALES['osnr']['safe']) &
        (df['temperatura'] <= UMBRALES['temperatura']['safe']),
        
        # 3. WARNING: ALGUNA métrica en zona de advertencia (pero ninguna crítica)
        (
            ((df['potencia_dbm'] < UMBRALES['potencia']['safe']) & (df['potencia_dbm'] >= UMBRALES['potencia']['critical'])) |
            ((df['ber'] > UMBRALES['ber']['safe']) & (df['ber'] <= UMBRALES['ber']['critical'])) |
            ((df['osnr'] < UMBRALES['osnr']['safe']) & (df['osnr'] >= UMBRALES['osnr']['critical'])) |
            ((df['temperatura'] > UMBRALES['temperatura']['safe']) & (df['temperatura'] <= UMBRALES['temperatura']['critical']))
        ) & ~(  # Excluir casos que ya son críticos
            (df['potencia_dbm'] < UMBRALES['potencia']['critical']) |
            (df['ber'] > UMBRALES['ber']['critical']) |
            (df['osnr'] < UMBRALES['osnr']['critical']) |
            (df['temperatura'] > UMBRALES['temperatura']['critical'])
        )
    ]
    
    choices = ['critical', 'safe', 'warning']
    df['estado'] = np.select(conditions, choices, default='unknown')
    
    # Guardar datos etiquetados
    Path(labeled_data_path).parent.mkdir(exist_ok=True)
    df.to_csv(labeled_data_path, index=False)
    print(f"✅ Datos etiquetados guardados en: {labeled_data_path}")
    print(f"📊 Distribución de estados:\n{df['estado'].value_counts()}")
    print("🔍 -------------------------------------------------------")
    print(df[['potencia_dbm', 'ber', 'osnr', 'temperatura', 'estado']].sample(10))
    print("🔍 Ejemplos de estados críticos:")
    print(df[df["temperatura"] > 85][["temperatura", "estado"]].head())
    print("\n🔍 Validación de umbrales de potencia:")
    print(df[(df['potencia_dbm'] < -30)][['potencia_dbm', 'estado']].head(10))
    print(df["estado"].unique())  # Verifica si hay valores inesperados

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    raw_data_path = project_root / "ml_engine" / "data" / "raw" / "dwdm_raw_data.csv"
    labeled_data_path = project_root / "ml_engine" / "data" / "processed" / "dwdm_labeled.csv"
    
    label_data(raw_data_path=str(raw_data_path), labeled_data_path=str(labeled_data_path))
    