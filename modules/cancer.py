"""
MediGuard360 – Module Cancer / Complications Chimiothérapie
============================================================
Prédit la probabilité de complication grave (ex. Neutropénie fébrile)
lors d'une cure de chimiothérapie.

Algorithme : XGBoost Classifier entraîné sur des données synthétiques.

Entrées  : température, intensité douleur, nausées, niveau de fatigue
Sorties  : probabilité de complication [0-100 %] + type de complication probable
"""

from __future__ import annotations

import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import os
import logging

logger = logging.getLogger(__name__)

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "cancer_xgb.joblib")

# ---------------------------------------------------------------------------
# Données synthétiques d'entraînement
# ---------------------------------------------------------------------------

def _generate_synthetic_data(n: int = 600, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """
    Génère des données synthétiques représentant des patients en chimio.

    Colonnes :
        0 – temperature      : température corporelle en °C [35.0-41.5]
        1 – pain_intensity   : intensité douleur            [1-10]
        2 – nausea           : présence de nausées          [0/1]
        3 – fatigue_level    : niveau de fatigue            [0-10]

    Label : 0 = pas de complication, 1 = complication détectée
    """
    rng = np.random.default_rng(seed)

    temperature    = rng.uniform(35.0, 41.5, n)
    pain_intensity = rng.uniform(1, 10, n)
    nausea         = rng.integers(0, 2, n)
    fatigue_level  = rng.uniform(0, 10, n)

    # Heuristique de labellisation médicale (données réelles remplaceraient cela)
    fever_flag   = (temperature > 38.3).astype(float)    # fièvre neutropénique
    pain_flag    = (pain_intensity > 7).astype(float)
    fatigue_flag = (fatigue_level > 7).astype(float)

    risk = fever_flag * 40 + pain_flag * 25 + nausea * 15 + fatigue_flag * 20
    labels = (risk > 45).astype(int)

    X = np.column_stack([temperature, pain_intensity, nausea, fatigue_level])
    return X, labels


# ---------------------------------------------------------------------------
# Pipeline XGBoost
# ---------------------------------------------------------------------------

def _build_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            eval_metric="logloss",
            random_state=42,
        )),
    ])


def _load_or_train_model() -> Pipeline:
    if os.path.exists(_MODEL_PATH):
        logger.info("Cancer : chargement du modèle depuis %s", _MODEL_PATH)
        return joblib.load(_MODEL_PATH)

    logger.info("Cancer : entraînement du modèle XGBoost…")
    X, y = _generate_synthetic_data()
    pipeline = _build_pipeline()
    pipeline.fit(X, y)

    os.makedirs(os.path.dirname(_MODEL_PATH), exist_ok=True)
    joblib.dump(pipeline, _MODEL_PATH)
    logger.info("Cancer : modèle sauvegardé dans %s", _MODEL_PATH)
    return pipeline


_pipeline: Pipeline | None = None


def get_model() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = _load_or_train_model()
    return _pipeline


# ---------------------------------------------------------------------------
# Identification de la complication la plus probable
# ---------------------------------------------------------------------------

def _identify_complication(
    temperature: float,
    pain_intensity: float,
    nausea: bool,
    fatigue_level: float,
    probability: float,
) -> str:
    """
    Règle clinique simplifiée d'identification de la complication probable.
    En production, un classifieur multi-classe remplacerait cette logique.
    """
    if temperature > 38.3 and probability > 0.5:
        return "Neutropénie fébrile"
    if nausea and fatigue_level > 7:
        return "Syndrome de fatigue sévère / Anémie"
    if pain_intensity > 7:
        return "Mucite / Douleur neuropathique"
    if probability > 0.4:
        return "Complication non spécifique (surveillance rapprochée)"
    return "Aucune complication majeure détectée"


# ---------------------------------------------------------------------------
# Fonction de prédiction principale
# ---------------------------------------------------------------------------

def predict_chemo_complication(
    temperature: float,
    pain_intensity: float,
    nausea: bool,
    fatigue_level: float,
) -> dict:
    """
    Évalue le risque de complication lors d'une cure de chimiothérapie.

    Paramètres
    ----------
    temperature    : température corporelle en degrés Celsius
    pain_intensity : intensité de la douleur de 1 (légère) à 10 (insupportable)
    nausea         : True si le patient présente des nausées
    fatigue_level  : niveau de fatigue de 0 (inexistant) à 10 (extrême)

    Retour
    ------
    dict contenant :
        - complication_probability  : float [0-100 %]
        - complication_type         : str, type de complication probable
        - risk_level                : str "LOW" | "MODERATE" | "HIGH" | "CRITICAL"
        - alert                     : dict présent si probabilité > 70 %
    """
    model = get_model()

    # Prétraitement : mise à l'échelle gérée par le pipeline
    X = np.array([[
        float(temperature),
        float(pain_intensity),
        int(nausea),
        float(fatigue_level),
    ]])

    prob_complication: float = float(model.predict_proba(X)[0][1])
    probability_percent = round(prob_complication * 100, 2)

    complication_type = _identify_complication(
        temperature, pain_intensity, nausea, fatigue_level, prob_complication
    )

    if probability_percent <= 30:
        risk_level = "LOW"
    elif probability_percent <= 55:
        risk_level = "MODERATE"
    elif probability_percent <= 70:
        risk_level = "HIGH"
    else:
        risk_level = "CRITICAL"

    result: dict = {
        "module": "cancer_chemo",
        "complication_probability_percent": probability_percent,
        "complication_type": complication_type,
        "risk_level": risk_level,
        "input_features": {
            "temperature_celsius": temperature,
            "pain_intensity": pain_intensity,
            "nausea": nausea,
            "fatigue_level": fatigue_level,
        },
    }

    if probability_percent > 70:
        result["alert"] = {
            "severity": "CRITICAL",
            "message": (
                f"Risque élevé de '{complication_type}' détecté. "
                "Une évaluation médicale urgente est nécessaire."
            ),
            "recommendations": [
                "Hospitalisation immédiate si fièvre > 38,3 °C.",
                "Numération Formule Sanguine (NFS) en urgence.",
                "Antibiothérapie empirique selon le protocole établi.",
                "Hydratation intraveineuse si nausées sévères.",
            ],
        }

    return result
