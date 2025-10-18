import joblib
from sklearn.linear_model import LogisticRegression
import numpy as np
import os

# Crear carpeta "models" si no existe
os.makedirs("models", exist_ok=True)

# Datos falsos (13 features)
X = np.random.rand(100, 13)
y = np.random.randint(0, 2, 100)

# Entrenamiento del modelo dummy
model = LogisticRegression()
model.fit(X, y)

# Guardar modelo
joblib.dump(model, "models/model.pkl")

print("✅ Modelo dummy guardado correctamente en 'models/model.pkl'")
