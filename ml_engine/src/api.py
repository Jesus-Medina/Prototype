import random 
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
from pydantic import BaseModel
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
from typing import List, Dict, Any

# Configuración inicial
warnings.filterwarnings("ignore", category=UserWarning)

app = FastAPI(
    title="API de Predicción DWDM",
    description="API para predecir el estado de fibras ópticas DWDM basado en métricas técnicas",
    version="1.0.0"
)

# Configura CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar modelo
model_path = Path(__file__).parent.parent / "models" / "fiber_model.pkl"
model = joblib.load(model_path)

class FiberRequest(BaseModel):
    potencia_dbm: float
    ber: float
    osnr: float
    temperatura: float
    edad_cable: int

# Ruta al archivo CSV
DATA_PATH = Path(__file__).parent.parent / "data" / "processed" / "dwdm_labeled.csv"

print(f"Intentando leer CSV desde: {DATA_PATH}")
print(f"El archivo existe: {DATA_PATH.exists()}")

# Definición de regiones mineras con coordenadas en tierra
REGIONES_CHILE = {
    "Antofagasta": {
        "coords": [-23.650, -69.400],  # Movido más al este
        "radio": 0.5,
        "conexiones": ["Atacama", "Tarapacá"],
        "es_hub": False,
        "area_segura": {
            "lat_min": -24.0, "lat_max": -23.0,
            "lng_min": -70.0, "lng_max": -68.5
        }
    },
    "Atacama": {
        "coords": [-27.366, -69.832],  # Ajustado hacia el este
        "radio": 0.4,
        "conexiones": ["Antofagasta", "Coquimbo"],
        "es_hub": True,
        "area_segura": {
            "lat_min": -28.0, "lat_max": -26.5,
            "lng_min": -70.5, "lng_max": -69.0
        }
    },
    "Coquimbo": {
        "coords": [-30.000, -70.500],  # Ajustado hacia el este
        "radio": 0.3,
        "conexiones": ["Atacama", "Valparaíso"],
        "es_hub": False,
        "area_segura": {
            "lat_min": -31.0, "lat_max": -29.5,
            "lng_min": -71.5, "lng_max": -70.0
        }
    },
    "Tarapacá": {
        "coords": [-20.217, -69.333],  # Movido más al este
        "radio": 0.4,
        "conexiones": ["Antofagasta"],
        "es_hub": True,
        "area_segura": {
            "lat_min": -21.0, "lat_max": -19.5,
            "lng_min": -70.0, "lng_max": -68.5
        }
    },
    "Valparaíso": {
        "coords": [-33.045, -71.120],  # Ajustado hacia el este
        "radio": 0.3,
        "conexiones": ["Coquimbo"],
        "es_hub": True,
        "area_segura": {
            "lat_min": -33.5, "lat_max": -32.5,
            "lng_min": -71.5, "lng_max": -70.5
        }
    },
    "Santiago": {
        "coords": [-33.4489, -70.2693],  # Ajustado hacia el este
        "radio": 0.2,
        "conexiones": ["Valparaíso"],
        "es_hub": True,
        "area_segura": {
            "lat_min": -34.0, "lat_max": -33.0,
            "lng_min": -71.0, "lng_max": -70.0
        }
    },
    "Arica": {
        "coords": [-18.478, -69.812],  # Movido más al este
        "radio": 0.3,
        "conexiones": ["Tarapacá"],
        "es_hub": True,
        "area_segura": {
            "lat_min": -19.0, "lat_max": -18.0,
            "lng_min": -70.5, "lng_max": -69.0
        }
    },
    "Copiapó": {
        "coords": [-27.366, -69.932],  # Ajustado hacia el este
        "radio": 0.3,
        "conexiones": [],
        "es_hub": True,
        "area_segura": {
            "lat_min": -28.0, "lat_max": -27.0,
            "lng_min": -70.5, "lng_max": -69.5
        }
    }
}

def generar_coordenada_segura(region):
    """Genera coordenadas dentro del área segura definida para la región"""
    area = region["area_segura"]
    lat = random.uniform(area["lat_min"], area["lat_max"])
    lng = random.uniform(area["lng_min"], area["lng_max"])
    return [lat, lng]

def generate_coordinates(cable_id: str):
    """Genera coordenadas realistas para zonas mineras de Chile evitando el agua"""
    regiones = list(REGIONES_CHILE.keys())
    region_seleccionada = regiones[hash(cable_id) % len(regiones)]
    region = REGIONES_CHILE[region_seleccionada]
    
    # Solo el primer cable de cada región será el hub (si la región lo permite)
    if not hasattr(generate_coordinates, 'hubs_por_region'):
        generate_coordinates.hubs_por_region = {}
    
    is_hub = False
    if region_seleccionada not in generate_coordinates.hubs_por_region and region["es_hub"]:
        generate_coordinates.hubs_por_region[region_seleccionada] = {
            'id': cable_id,
            'connections_drawn': set()
        }
        is_hub = True
    
    # Generar coordenadas dentro del área segura
    coordinates = generar_coordenada_segura(region)
    
    # Para conexiones no duplicadas
    connections_to_include = []
    if is_hub:
        for connected_region in region["conexiones"]:
            if connected_region not in generate_coordinates.hubs_por_region.get(region_seleccionada, {}).get('connections_drawn', set()):
                connections_to_include.append(connected_region)
                if region_seleccionada not in generate_coordinates.hubs_por_region:
                    generate_coordinates.hubs_por_region[region_seleccionada] = {'connections_drawn': set()}
                generate_coordinates.hubs_por_region[region_seleccionada]['connections_drawn'].add(connected_region)
    
    return {
        "coordinates": coordinates,
        "region": region_seleccionada,
        "connections": connections_to_include,
        "is_hub": is_hub
    }

@app.get("/api/cables", response_model=List[Dict[str, Any]])
async def get_cables(limit: int = 50):
    """Endpoint para obtener cables distribuidos en zonas mineras de Chile"""
    try:
        # Limpiar el registro de hubs y conexiones para cada nueva llamada
        if hasattr(generate_coordinates, 'hubs_por_region'):
            delattr(generate_coordinates, 'hubs_por_region')
            
        df = pd.read_csv(
            DATA_PATH,
            dtype={
                'cable_id': str,
                'potencia_dbm': float,
                'ber': float,
                'osnr': float,
                'temperatura': float,
                'edad_cable': int,
                'estado': str
            },
            parse_dates=['timestamp']
        )
        
        df = df.dropna()
        df = df[df["estado"].str.lower().isin(["safe", "warning", "critical"])]
        
        # Cambio clave: en lugar de .tail() usamos .sample() para obtener registros aleatorios
        df = df.sample(n=min(limit, len(df)), random_state=42)  # random_state para reproducibilidad

        cables = []
        for _, row in df.iterrows():
            try:
                geo_data = generate_coordinates(row["cable_id"])
                
                cables.append({
                    "id": str(row["cable_id"]),
                    "station": f"Estación {row['cable_id']} ({geo_data['region']})",
                    "coordinates": geo_data["coordinates"],
                    "region": geo_data["region"],
                    "connections": geo_data["connections"],
                    "is_hub": geo_data["is_hub"],
                    "age": int(row["edad_cable"]),
                    "power": float(row["potencia_dbm"]),
                    "temperature": float(row["temperatura"]),
                    "osnr": float(row["osnr"]),
                    "ber": float(row["ber"]),
                    "status": str(row["estado"]).lower()
                })
            except (ValueError, TypeError) as e:
                print(f"Error procesando fila {row['cable_id']}: {str(e)}")
                continue
        
        return cables
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al leer el CSV: {str(e)}. Ruta intentada: {DATA_PATH}"
        )


@app.post("/predict", response_model=Dict[str, Any])
async def predict_status(request: FiberRequest):
    """
    Endpoint para predecir el estado de la fibra óptica.
    """
    try:
        if request.ber > 1e-6:
            return {
                "prediction": "critical",
                "confidence": 1.0,
                "probabilities": {"critical": 1.0, "warning": 0.0, "safe": 0.0}
            }
        if request.potencia_dbm < -40:
            return {
                "status": "success",
                "prediction": "critical",
                "confidence": 1.0,
                "probabilities": {"critical": 1.0, "warning": 0.0, "safe": 0.0},
                "metrics": {
                    "potencia_dbm": request.potencia_dbm,
                    "ber_log10": np.log10(request.ber),
                    "osnr": request.osnr,
                    "temperatura": request.temperatura,
                    "edad_cable": request.edad_cable
                }
            }
        
        input_data = [[
            request.potencia_dbm,
            np.log10(request.ber),
            request.osnr,
            request.temperatura,
            request.edad_cable
        ]]
        
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        
        return {
            "status": "success",
            "prediction": prediction,
            "confidence": float(np.max(probabilities)),
            "probabilities": dict(zip(model.classes_, [float(p) for p in probabilities])),
            "metrics": {
                "potencia_dbm": request.potencia_dbm,
                "ber_log10": np.log10(request.ber),
                "osnr": request.osnr,
                "temperatura": request.temperatura,
                "edad_cable": request.edad_cable
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "details": "Verifique los valores de entrada y vuelva a intentar"
        }

@app.get("/model_info")
async def get_model_info():
    """Endpoint para obtener información del modelo cargado"""
    return {
        "model_type": str(type(model)),
        "features": ['potencia_dbm', 'ber_log10', 'osnr', 'temperatura', 'edad_cable'],
        "classes": model.classes_.tolist()
    }

@app.get("/regiones")
async def get_regiones_info():
    """Endpoint para obtener información de las regiones y sus conexiones"""
    return {
        "regiones": [
            {
                "nombre": nombre,
                "coords": data["coords"],
                "conexiones": data["conexiones"],
                "es_hub": data["es_hub"]
            }
            for nombre, data in REGIONES_CHILE.items()
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="debug"
    )