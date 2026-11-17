from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


st.set_page_config(
    page_title="DRBOT · Predicción por síntomas",
    page_icon="🩺",
    layout="wide",
)
st.title("🩺 DRBOT · Predicción orientativa por síntomas")
st.warning(
    "Proyecto académico. El resultado es una clasificación experimental, "
    "no un diagnóstico ni una recomendación médica."
)

MODEL_PATH = Path(__file__).with_name("drbot_modelos_enfermedad.joblib")
TARGET_ALIASES = ["diseases", "disease", "enfermedad"]
RANDOM_STATE = 42
EVAL_CASES_PER_DISEASE = 5


def find_target(columns):
    lookup = {str(column).strip().casefold(): column for column in columns}
    return next((lookup[name] for name in TARGET_ALIASES if name in lookup), None)


def load_dataset(uploaded):
    uploaded.seek(0)
    header = pd.read_csv(uploaded, nrows=0)
    target_col = find_target(header.columns)

    if target_col is None:
        raise ValueError(
            "No encuentro la columna diseases. Carga el archivo data.csv "
            "del dataset Symptoms to Diseases."
        )

    feature_cols = [column for column in header.columns if column != target_col]
    if not feature_cols:
        raise ValueError("El CSV no contiene columnas de síntomas.")

    uploaded.seek(0)
    data = pd.read_csv(
        uploaded,
        dtype={column: "uint8" for column in feature_cols},
        low_memory=False,
    )
    data[target_col] = data[target_col].astype("string").str.strip()
    data = data.dropna(subset=[target_col])
    data = data[data[target_col] != ""].copy()

    if data.empty:
        raise ValueError("El CSV no contiene filas válidas.")

    invalid_values = {
        column: sorted(set(data[column].dropna().unique()) - {0, 1})
        for column in feature_cols
    }
    invalid_values = {column: values for column, values in invalid_values.items() if values}
    if invalid_values:
        raise ValueError(
            "Las columnas de síntomas deben ser binarias (0/1). "
            f"Valores no válidos en: {', '.join(invalid_values)}"
        )

    data[feature_cols] = data[feature_cols].fillna(0).astype("uint8")
    return data, feature_cols, target_col


def readable_symptom(name):
    return str(name).replace("_", " ").strip().capitalize()


def make_artifact(models, default_model, feature_cols, labels):
    return {
        "models": models,
        "default_model": default_model,
        "feature_cols": list(feature_cols),
        "labels": list(labels),
        "target": "diseases",
        "dataset": "Symptoms to Diseases",
    }


with st.sidebar:
    page = st.radio("Sección", ["Datos", "Entrenamiento y evaluación"])

uploaded = st.file_uploader(
    "Carga data.csv del dataset Symptoms to Diseases",
    type="csv",
)

if uploaded is None:
    st.info("Carga data.csv para inspeccionar los síntomas y entrenar el modelo.")
    st.stop()

try:
    data, feature_cols, target_col = load_dataset(uploaded)
except Exception as exc:
    st.error(f"No se pudo preparar el CSV: {exc}")
    st.stop()

labels = sorted(data[target_col].astype(str).unique().tolist())
st.caption(
    f"{len(data):,} casos · {len(feature_cols)} síntomas · {len(labels)} enfermedades"
)

if page == "Datos":
    st.subheader("Distribución de enfermedades")
    st.write(
        "Entradas: síntomas codificados con 0 (ausente) y 1 (presente)."
    )
    st.write(f"Objetivo que se predice: **{target_col}**")
    st.bar_chart(data[target_col].value_counts().head(30))
    st.dataframe(data.head(20), use_container_width=True)

elif page == "Entrenamiento y evaluación":
    test_pct = st.slider(
        "Porcentaje reservado para test",
        min_value=10,
        max_value=30,
        value=20,
        step=5,
    )
    n_estimators = st.slider(
        "Número de árboles del Random Forest",
        min_value=100,
        max_value=300,
        value=200,
        step=50,
    )

    if st.button("Entrenar y evaluar", type="primary"):
        st.session_state.pop("model_metrics", None)
        progress = st.progress(0, text="Preparando los datos…")
        training_status = st.empty()
        try:
            X = data[feature_cols]
            y = data[target_col].astype(str)

            if y.value_counts().min() < 2:
                raise ValueError(
                    "Cada enfermedad necesita al menos dos ejemplos "
                    "para separar entrenamiento y test."
                )

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_pct / 100,
                random_state=RANDOM_STATE,
                stratify=y,
            )

            # Limita la evaluación a cinco casos por enfermedad para que las
            # métricas sean rápidas. El ajuste final usa todo el CSV.
            evaluation_index = y_test.groupby(y_test, sort=False).head(
                EVAL_CASES_PER_DISEASE
            ).index
            X_eval = X_test.loc[evaluation_index]
            y_eval = y_test.loc[evaluation_index]
            st.caption(
                f"Evaluación rápida: {len(y_eval):,} casos de test "
                f"(hasta {EVAL_CASES_PER_DISEASE} por enfermedad)."
            )

            model_name = "Random Forest"
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_features="sqrt",
                max_depth=5,
                class_weight="balanced",
                n_jobs=1,
                random_state=RANDOM_STATE,
            )
            training_status.info("Ajustando Random Forest con los datos de entrenamiento…")
            model.fit(X_train, y_train)
            training_status.info("Calculando métricas de Random Forest…")
            predictions = model.predict(X_eval)
            metrics = pd.DataFrame(
                [{
                    "Accuracy": accuracy_score(y_eval, predictions),
                    "F1 macro": f1_score(
                        y_eval,
                        predictions,
                        average="macro",
                        zero_division=0,
                    ),
                }],
                index=[model_name],
            )
            progress.progress(55, text="Evaluación de Random Forest completada")

            st.subheader("Resultados de evaluación")
            st.dataframe(metrics.style.format("{:.3f}"), use_container_width=True)
            st.session_state["model_metrics"] = metrics

            # Ajusta el clasificador final con todas las filas.
            training_status.info("Ajustando Random Forest con todos los datos…")
            model.fit(X, y)
            progress.progress(90, text="Random Forest final listo")

            models = {model_name: model}
            default_model = model_name
            artifact = make_artifact(models, default_model, feature_cols, labels)
            training_status.info("Guardando Random Forest para la API…")
            joblib.dump(artifact, MODEL_PATH, compress=3)
            progress.progress(100, text="Entrenamiento y guardado completados")

            st.success(
                f"Random Forest se guardó en {MODEL_PATH}."
            )
        except Exception as exc:
            training_status.error(f"No se pudo completar el entrenamiento: {exc}")

    if st.session_state.get("model_metrics") is not None:
        st.subheader("Resultados de Random Forest")
        st.dataframe(
            st.session_state.model_metrics.style.format("{:.3f}"),
            use_container_width=True,
        )

