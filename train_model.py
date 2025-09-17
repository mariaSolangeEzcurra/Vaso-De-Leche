import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import unicodedata

# ==============================
# Función para normalizar nombres de archivo
# ==============================
def normalizar_nombre(cadena: str) -> str:
    """
    Convierte una cadena a minúsculas, reemplaza espacios por guiones bajos
    y elimina acentos y caracteres especiales (como la ñ).
    """
    cadena = cadena.lower().replace(" ", "_")
    cadena = unicodedata.normalize("NFKD", cadena).encode("ascii", "ignore").decode("utf-8")
    return cadena

# ==============================
# Cargar dataset
# ==============================
df = pd.read_csv("beneficiarios_limpio.csv")

# Columnas que vamos a predecir
categorias = [
    "BENEFICIARIOS DE 0-6 AÑOS",
    "BENEFICIARIOS GESTANTES",
    "BENEFICIARIOS LACTANTES",
    "BENEFICIARIOS DISCAPACITADOS",
    "BENEFICIARIOS ANCIANOS"
]

# ==============================
# Entrenar un modelo por cada categoría
# ==============================
for cat in categorias:
    print(f"🔄 Entrenando modelo para: {cat}")

    # Usamos como features las demás categorías
    X = df[[c for c in categorias if c != cat]]
    y = df[cat]

    # Dividir datos en train y test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Entrenar modelo
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Guardar modelo con nombre normalizado
    filename = f"modelo_{normalizar_nombre(cat)}.pkl"
    joblib.dump(model, filename)

    # Reporte simple
    score = model.score(X_test, y_test)
    print(f"✅ Modelo para {cat} guardado como {filename} (R² en test = {score:.3f})")

print("🎉 Todos los modelos fueron entrenados y guardados correctamente.")
