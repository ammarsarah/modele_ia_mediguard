"""
MediGuard360 – Module Épilepsie
================================
Prédit le risque de crise épileptique à partir d'un formulaire patient.
Algorithme : Random Forest Classifier entraîné sur des données synthétiques.

Entrées  : qualité du sommeil, niveau de stress, oubli de médicaments, auras
Sorties  : score de risque [0-100 %] + alerte critique si > 80 %
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import os
import logging

logger = logging.getLogger(__name__)

# Chemin de sauvegarde du modèle sérialisé
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "epilepsy_rf.joblib")

# ---------------------------------------------------------------------------
# Génération de données synthétiques d'entraînement
# ---------------------------------------------------------------------------

def _generate_synthetic_data(n: int = 500, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Produit un jeu de données synthétiques représentatif.

    Colonnes :
        0 – sleep_quality       : qualité du sommeil [0-10]
        1 – stress_level        : niveau de stress   [0-10]
        2 – medication_missed   : oubli médicament   [0/1]
        3 – aura_present        : présence d'aura    [0/1]

    Label : 0 = risque faible, 1 = risque élevé (> 80 %)
    """
    rng = np.random.default_rng(seed)
    sleep  = rng.uniform(0, 10, n)
    stress = rng.uniform(0, 10, n)
    missed = rng.integers(0, 2, n)   # 0 ou 1
    aura   = rng.integers(0, 2, n)   # 0 ou 1

    # Règle de labellisation heuristique (remplacée par vraies données en prod)
    risk_score = (
        (10 - sleep)  * 3   # mauvais sommeil → +risque
        + stress      * 3   # stress élevé   → +risque
        + missed      * 20  # oubli médic.   → +risque
        + aura        * 25  # aura présente  → +risque fort
    )
    labels = (risk_score > 48).astype(int)  # seuil calibré sur la plage max ≈ 96

    X = np.column_stack([sleep, stress, missed, aura])
    return X, labels


# ---------------------------------------------------------------------------
# Construction / chargement du pipeline scikit-learn
# ---------------------------------------------------------------------------

def _build_pipeline() -> Pipeline:
    """Construit le pipeline RandomForest avec mise à l'échelle."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            random_state=42,
            class_weight="balanced",
        )),
    ])


def _load_or_train_model() -> Pipeline:
    """Charge le modèle depuis le disque ou le (ré)entraîne si absent."""
    if os.path.exists(_MODEL_PATH):
        logger.info("Épilepsie : chargement du modèle depuis %s", _MODEL_PATH)
        return joblib.load(_MODEL_PATH)

    logger.info("Épilepsie : entraînement du modèle Random Forest…")
    X, y = _generate_synthetic_data()
    pipeline = _build_pipeline()
    pipeline.fit(X, y)

    os.makedirs(os.path.dirname(_MODEL_PATH), exist_ok=True)
    joblib.dump(pipeline, _MODEL_PATH)
    logger.info("Épilepsie : modèle sauvegardé dans %s", _MODEL_PATH)
    return pipeline


# Instance unique (chargée à l'import du module)
_pipeline: Pipeline | None = None


def get_model() -> Pipeline:
    """Retourne l'instance du modèle (singleton paresseux)."""
    global _pipeline
    if _pipeline is None:
        _pipeline = _load_or_train_model()
    return _pipeline


# ---------------------------------------------------------------------------
# Fonction de prédiction principale
# ---------------------------------------------------------------------------

def predict_epilepsy_risk(
    sleep_quality: float,
    stress_level: float,
    medication_missed: bool,
    aura_present: bool,
) -> dict:
    """
    Prédit le risque épileptique et retourne un rapport JSON structuré.

    Paramètres
    ----------
    sleep_quality     : qualité du sommeil de 0 (très mauvaise) à 10 (excellente)
    stress_level      : niveau de stress de 0 (nul) à 10 (extrême)
    medication_missed : True si le patient a oublié sa médication
    aura_present      : True si des auras ont été rapportées

    Retour
    ------
    dict contenant :
        - risk_score    : float, probabilité [0-100 %]
        - risk_level    : str, "LOW" | "MODERATE" | "HIGH" | "CRITICAL"
        - alert         : dict présent uniquement si risk_score > 80 %
    """
    model = get_model()

    # Prétraitement : LabelEncoding implicite (booléens → 0/1)
    X = np.array([[
        float(sleep_quality),
        float(stress_level),
        int(medication_missed),
        int(aura_present),
    ]])

    # Probabilité de la classe 1 (risque élevé)
    prob_high_risk: float = float(model.predict_proba(X)[0][1])
    risk_score = round(prob_high_risk * 100, 2)

    # Catégorisation du risque
    if risk_score <= 30:
        risk_level = "LOW"
    elif risk_score <= 60:
        risk_level = "MODERATE"
    elif risk_score <= 80:
        risk_level = "HIGH"
    else:
        risk_level = "CRITICAL"

    result: dict = {
        "module": "epilepsy",
        "risk_score_percent": risk_score,
        "risk_level": risk_level,
        "input_features": {
            "sleep_quality": sleep_quality,
            "stress_level": stress_level,
            "medication_missed": medication_missed,
            "aura_present": aura_present,
        },
    }

    # Alerte critique si risque > 80 %
    if risk_score > 80:
        result["alert"] = {
            "severity": "CRITICAL",
            "message": (
                "Risque de crise épileptique très élevé détecté. "
                "Une consultation médicale immédiate est fortement recommandée."
            ),
            "recommendations": [
                "Repos immédiat dans un environnement calme et sécurisé.",
                "Prendre les médicaments antiépileptiques prescrits sans délai.",
                "Éviter toute activité dangereuse (conduite, piscine, hauteurs).",
                "Contacter votre neurologue ou appeler le 15 (SAMU).",
            ],
        }

    return result
