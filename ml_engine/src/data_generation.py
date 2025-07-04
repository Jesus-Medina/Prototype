import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict

def generate_strict_dwdm_data(save_path: str, num_samples: int = 10000) -> None:
    """
    Genera datos DWDM con distribución estricta:
    - 70% safe
    - 25% warning
    - 5% critical
    Con mezcla aleatoria en el CSV final.
    """
    np.random.seed(42)
    
    PARAMS = {
        'safe': {
            'temperatura': (30, 45),
            'potencia': (-12, -18),
            'osnr': (35, 40),
            'ber': (1e-16, 1e-13),
            'edad_cable': (1, 4)
        },
        'warning': {
            'temperatura': (50, 70),
            'potencia': (-19, -24),
            'osnr': (25, 32),
            'ber': (1e-10, 1e-8),
            'edad_cable': (5, 9)
        },
        'critical': {
            'temperatura': (75, 100),
            'potencia': (-26, -40),
            'osnr': (15, 22),
            'ber': (1e-6, 1e-4),
            'edad_cable': (10, 20)
        }
    }

    n_safe = int(num_samples * 0.7)
    n_warning = int(num_samples * 0.25)
    n_critical = num_samples - n_safe - n_warning

    def generate_with_bounds(params, n):
        temp_mean = np.mean(params['temperatura'])
        temp_std = (params['temperatura'][1] - params['temperatura'][0])/6
        temperatura = np.random.normal(temp_mean, temp_std, n)
        temperatura = np.clip(temperatura, *params['temperatura'])
        
        return {
            'temperatura': temperatura,
            'potencia': np.random.uniform(*params['potencia'], n),
            'osnr': np.random.uniform(*params['osnr'], n),
            'ber': 10 ** np.random.uniform(
                np.log10(params['ber'][0]),
                np.log10(params['ber'][1]),
                n
            ),
            'edad_cable': np.random.randint(*params['edad_cable'], n),
            'estado': [params['estado']] * n
        }

    safe_params = PARAMS['safe'].copy()
    safe_params['estado'] = 'safe'
    safe_data = generate_with_bounds(safe_params, n_safe)

    warning_params = PARAMS['warning'].copy()
    warning_params['estado'] = 'warning'
    warning_data = generate_with_bounds(warning_params, n_warning)

    critical_params = PARAMS['critical'].copy()
    critical_params['estado'] = 'critical'
    critical_data = generate_with_bounds(critical_params, n_critical)

    combined_data = {
        'timestamp': pd.date_range(end=datetime.now(), periods=num_samples, freq='h'),
        'cable_id': [f"DWDM-{i:05d}" for i in range(num_samples)],
        'estado': safe_data['estado'] + warning_data['estado'] + critical_data['estado'],
        'edad_cable': np.concatenate([
            safe_data['edad_cable'], 
            warning_data['edad_cable'], 
            critical_data['edad_cable']
        ]),
        'potencia_dbm': np.concatenate([
            safe_data['potencia'], 
            warning_data['potencia'], 
            critical_data['potencia']
        ]),
        'temperatura': np.concatenate([
            safe_data['temperatura'], 
            warning_data['temperatura'], 
            critical_data['temperatura']
        ]),
        'osnr': np.concatenate([
            safe_data['osnr'], 
            warning_data['osnr'], 
            critical_data['osnr']
        ]),
        'ber': np.concatenate([
            safe_data['ber'], 
            warning_data['ber'], 
            critical_data['ber']
        ])
    }

    df = pd.DataFrame(combined_data)

    # ✅ MEZCLA ALEATORIA AQUÍ
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Casos de prueba
    test_cases = pd.DataFrame({
        'timestamp': [df['timestamp'].iloc[0]] * 3,
        'cable_id': ['TEST-SAFE', 'TEST-WARNING', 'TEST-CRITICAL'],
        'estado': ['safe', 'warning', 'critical'],
        'edad_cable': [3, 8, 15],
        'potencia_dbm': [-15.0, -22.5, -35.0],
        'temperatura': [40.0, 60.0, 90.0],
        'osnr': [38.0, 28.0, 18.0],
        'ber': [1e-14, 1e-9, 1e-5]
    })

    df = pd.concat([df, test_cases], ignore_index=True)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)

    print(f"✅ Datos generados en: {save_path}")
    print(f"📊 Distribución real obtenida:")
    total = len(df)
    for state in ['safe', 'warning', 'critical']:
        count = (df['estado'] == state).sum()
        print(f"{state.upper()}: {count} muestras ({count/total*100:.1f}%)")

    print("\n🔍 Validación de rangos:")
    for state in ['safe', 'warning', 'critical']:
        subset = df[df['estado'] == state]
        print(f"\n{state.upper()}:")
        print(f"Temperatura: {subset['temperatura'].min():.1f}-{subset['temperatura'].max():.1f}")
        print(f"Potencia: {subset['potencia_dbm'].min():.1f}-{subset['potencia_dbm'].max():.1f}")
        print(f"OSNR: {subset['osnr'].min():.1f}-{subset['osnr'].max():.1f}")
        print(f"BER: {subset['ber'].min():.1e}-{subset['ber'].max():.1e}")
        print(f"Edad cable: {subset['edad_cable'].min()}-{subset['edad_cable'].max()}")

if __name__ == "__main__":
    output_path = "ml_engine/data/raw/dwdm_raw_data.csv"
    generate_strict_dwdm_data(save_path=output_path, num_samples=10000)
