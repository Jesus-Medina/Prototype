import requests

# testing_api.py
data = {
    "potencia_dbm": -55.0,
    "ber": 1e-6,
    "osnr": 30.0,
    "temperatura": 125.0,
    "edad_cable": 5
}
response = requests.post("http://localhost:8000/predict", json=data)
print(response.json())  # Debería devolver {"prediction": "critical"} 