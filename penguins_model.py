import json
import joblib
import time
import platform
import sklearn
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Directorio de salida
output_dir = Path("models/penguins")
output_dir.mkdir(parents=True, exist_ok=True)

# Dataset Penguins
penguins = sns.load_dataset("penguins").dropna()

# Features y target
X = penguins[["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]]
y = penguins["species"]
target_names = y.unique().tolist()

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Shape entrenamiento:", X_train.shape)
print("Shape prueba:", X_test.shape)

# Pipeline
pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, multi_class="multinomial"))
])
pipeline.fit(X_train, y_train)

# Evaluación
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f" Accuracy en test: {acc:.4f}")
print("\nReporte de clasificación:\n")
print(classification_report(y_test, y_pred, target_names=target_names))

# Guardar modelo
model_path = output_dir / "penguins_model.pkl"
joblib.dump(pipeline, model_path)

# Guardar metadatos
meta = {
    "model_name": "penguins_logreg_pipeline",
    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "features": X.columns.tolist(),
    "target_names": target_names,
    "metrics": {"accuracy": acc},
    "sklearn_version": sklearn.__version__,
    "python_version": platform.python_version(),
    "pipeline": ["SimpleImputer(median)", "StandardScaler()", "LogisticRegression(max_iter=1000)"],
    "model_version": "1.0.0"
}
meta_path = output_dir / "penguins_meta.json"
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)

print(f" Modelo guardado en: {model_path}")
print(f" Metadatos guardados en: {meta_path}")

# Ejemplo de predicción
sample = [[39.1, 18.7, 181.0, 3750.0]]  # Adelie
pred_idx = pipeline.predict(sample)[0]
proba = pipeline.predict_proba(sample)[0]
print("Predicción (label):", pred_idx)
print("Probabilidades:", dict(zip(pipeline.classes_, proba)))

