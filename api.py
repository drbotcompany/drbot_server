from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


MODEL_PATH = Path(__file__).with_name("drbot_modelos_enfermedad.joblib")

app = FastAPI(title="DRBOT · Predicción por síntomas")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class SymptomRequest(BaseModel):
    symptoms: list[str]
    model: str | None = None


def load_artifact():
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="No hay modelo entrenado. Entrena primero desde app.py.",
        )
    try:
        artifact = joblib.load(MODEL_PATH)
        if "models" not in artifact:
            raise HTTPException(
                status_code=503,
                detail="El archivo guardado es de una versión anterior. Vuelve a entrenar ambos modelos desde app.py.",
            )
        return artifact
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"No se pudo cargar el modelo guardado: {exc}",
        )


def readable_symptom(name: str) -> str:
    return name.replace("_", " ").strip().capitalize()


@app.get("/health")
def health():
    return {
        "api": "ok",
        "modelo_disponible": MODEL_PATH.exists(),
    }


@app.get("/schema")
def schema():
    artifact = load_artifact()
    return {
        "symptoms": [
            {
                "name": name,
                "label": readable_symptom(name),
            }
            for name in artifact["feature_cols"]
        ],
        "target": artifact.get("target", "diseases"),
        "models": list(artifact["models"].keys()),
        "default_model": artifact["default_model"],
    }


@app.post("/predict")
def predict(request: SymptomRequest):
    artifact = load_artifact()
    feature_cols = artifact["feature_cols"]
    selected = set(request.symptoms)

    unknown = sorted(selected.difference(feature_cols))
    if unknown:
        raise HTTPException(
            status_code=422,
            detail="Síntomas desconocidos: "
            + ", ".join(readable_symptom(name) for name in unknown),
        )

    if not selected:
        raise HTTPException(
            status_code=422,
            detail="Selecciona al menos un síntoma.",
        )

    row = {
        column: int(column in selected)
        for column in feature_cols
    }
    patient = pd.DataFrame([row], columns=feature_cols)
    model_name = request.model or artifact["default_model"]
    if model_name not in artifact["models"]:
        raise HTTPException(
            status_code=422,
            detail="El modelo seleccionado no está disponible. Actualiza la página y vuelve a elegirlo.",
        )
    model = artifact["models"][model_name]

    try:
        disease = str(model.predict(patient)[0])
        probabilities = model.predict_proba(patient)[0]
        confidence = float(probabilities.max())
        top_indexes = probabilities.argsort()[-3:][::-1]
        top_predictions = [
            {
                "disease": str(model.classes_[index]),
                "confidence": float(probabilities[index]),
            }
            for index in top_indexes
        ]
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"No se pudo calcular la predicción: {exc}",
        )

    return {
        "disease": disease,
        "confidence": confidence,
        "top_predictions": top_predictions,
        "model": model_name,
    }

