"""
MediGuard360 – Cerveau IA (FastAPI)
=====================================
Point d'entrée principal de l'API d'intelligence artificielle de MediGuard360.

Architecture des modules :
  - /epilepsy/predict     → Prédiction du risque épileptique (Random Forest)
  - /cancer/predict       → Probabilité de complication chimiothérapie (XGBoost)
  - /mindcare/analyze     → Analyse NLP du journal thérapeutique (Transformers)
  - /vision/analyze       → Analyse d'image médicale IRM/Scanner (ResNet50)
  - /generate-report      → Rapport JSON fusionné (orchestration)

Toutes les entrées/sorties sont en JSON strict.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Configuration du logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("mediguard360")

# ---------------------------------------------------------------------------
# Lifespan – pré-chargement des modèles ML au démarrage
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Pré-chauffe les modèles ML dès le démarrage pour éviter
    la latence sur les premières requêtes.
    """
    logger.info("MediGuard360 IA – démarrage et chargement des modèles…")
    try:
        from modules.epilepsy import get_model as ep_model
        from modules.cancer import get_model as ca_model
        ep_model()
        ca_model()
        logger.info("Modèles ML (épilepsie, cancer) chargés avec succès.")
    except Exception as exc:
        logger.warning("Impossible de pré-charger les modèles ML : %s", exc)

    # Les modèles NLP et Vision sont chargés à la première requête (lru_cache)
    yield
    logger.info("MediGuard360 IA – arrêt propre.")


# ---------------------------------------------------------------------------
# Application FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(
    title="MediGuard360 – API IA",
    description=(
        "Plateforme IA médicale combinant prédiction épileptique, "
        "détection de complications chimiothérapie, analyse NLP de journaux "
        "thérapeutiques et analyse d'images médicales."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Restreindre en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Schémas Pydantic (entrées)
# ---------------------------------------------------------------------------


class EpilepsyInput(BaseModel):
    """Formulaire patient pour le module épilepsie."""
    sleep_quality: float = Field(
        ..., ge=0, le=10,
        description="Qualité du sommeil de 0 (très mauvaise) à 10 (excellente)",
        examples=[6.5],
    )
    stress_level: float = Field(
        ..., ge=0, le=10,
        description="Niveau de stress de 0 (nul) à 10 (extrême)",
        examples=[7.0],
    )
    medication_missed: bool = Field(
        ...,
        description="Le patient a-t-il oublié sa médication ?",
        examples=[False],
    )
    aura_present: bool = Field(
        ...,
        description="Des auras ont-elles été rapportées ?",
        examples=[True],
    )


class CancerInput(BaseModel):
    """Formulaire patient pour le module cancer / complications chimio."""
    temperature: float = Field(
        ..., ge=34.0, le=43.0,
        description="Température corporelle en degrés Celsius",
        examples=[38.5],
    )
    pain_intensity: float = Field(
        ..., ge=1, le=10,
        description="Intensité de la douleur de 1 (légère) à 10 (insupportable)",
        examples=[6.0],
    )
    nausea: bool = Field(
        ...,
        description="Présence de nausées",
        examples=[True],
    )
    fatigue_level: float = Field(
        ..., ge=0, le=10,
        description="Niveau de fatigue de 0 (inexistant) à 10 (extrême)",
        examples=[8.0],
    )


class MindCareInput(BaseModel):
    """Entrée de journal thérapeutique pour le module NLP MindCare."""
    text: str = Field(
        ..., min_length=3, max_length=5000,
        description="Texte libre du journal du patient",
        examples=["Je me sens très fatigué aujourd'hui, le traitement est épuisant."],
    )


class ReportInput(BaseModel):
    """Entrée pour la génération du rapport fusionné."""
    epilepsy: EpilepsyInput
    cancer: CancerInput
    mindcare: MindCareInput


# ---------------------------------------------------------------------------
# Endpoints – Module Épilepsie
# ---------------------------------------------------------------------------


@app.post(
    "/epilepsy/predict",
    summary="Prédiction du risque épileptique",
    tags=["Épilepsie"],
    status_code=status.HTTP_200_OK,
)
def predict_epilepsy(data: EpilepsyInput) -> dict:
    """
    Analyse le formulaire patient et retourne un score de risque de crise épileptique.

    - **sleep_quality** : qualité du sommeil [0-10]
    - **stress_level**  : niveau de stress [0-10]
    - **medication_missed** : oubli de médication
    - **aura_present**  : présence d'auras

    Si le score dépasse 80 %, une alerte critique avec recommandations est incluse.
    """
    try:
        from modules.epilepsy import predict_epilepsy_risk
        return predict_epilepsy_risk(
            sleep_quality=data.sleep_quality,
            stress_level=data.stress_level,
            medication_missed=data.medication_missed,
            aura_present=data.aura_present,
        )
    except Exception as exc:
        logger.exception("Erreur dans /epilepsy/predict")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur interne du module épilepsie : {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# Endpoints – Module Cancer
# ---------------------------------------------------------------------------


@app.post(
    "/cancer/predict",
    summary="Prédiction de complication chimiothérapie",
    tags=["Cancer"],
    status_code=status.HTTP_200_OK,
)
def predict_cancer(data: CancerInput) -> dict:
    """
    Évalue la probabilité de complication grave (ex. Neutropénie fébrile)
    lors d'une cure de chimiothérapie.

    - **temperature**      : température en °C [34-43]
    - **pain_intensity**   : douleur [1-10]
    - **nausea**           : présence de nausées
    - **fatigue_level**    : fatigue [0-10]
    """
    try:
        from modules.cancer import predict_chemo_complication
        return predict_chemo_complication(
            temperature=data.temperature,
            pain_intensity=data.pain_intensity,
            nausea=data.nausea,
            fatigue_level=data.fatigue_level,
        )
    except Exception as exc:
        logger.exception("Erreur dans /cancer/predict")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur interne du module cancer : {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# Endpoints – Module NLP MindCare
# ---------------------------------------------------------------------------


@app.post(
    "/mindcare/analyze",
    summary="Analyse NLP du journal thérapeutique",
    tags=["MindCare NLP"],
    status_code=status.HTTP_200_OK,
)
def analyze_mindcare(data: MindCareInput) -> dict:
    """
    Analyse complète d'une entrée de journal thérapeutique :
    - Détection de l'émotion dominante (tristesse, peur, colère, joie…)
    - Génération d'une réponse empathique et d'une action thérapeutique
    - Détection de mots-clés de détresse / idéation suicidaire → alerte psychologue
    - Résumé en 3 lignes pour le praticien
    """
    try:
        from modules.mindcare import analyze_journal
        return analyze_journal(data.text)
    except Exception as exc:
        logger.exception("Erreur dans /mindcare/analyze")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur interne du module MindCare : {exc}",
        ) from exc


@app.post(
    "/mindcare/summarize",
    summary="Résumé du journal thérapeutique",
    tags=["MindCare NLP"],
    status_code=status.HTTP_200_OK,
)
def summarize_mindcare(data: MindCareInput) -> dict:
    """
    Condense le journal thérapeutique en 3 lignes pour le praticien,
    sans effectuer l'analyse complète.
    """
    try:
        from modules.mindcare import summarize_journal
        return summarize_journal(data.text)
    except Exception as exc:
        logger.exception("Erreur dans /mindcare/summarize")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur interne du module résumé : {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# Endpoints – Module Vision
# ---------------------------------------------------------------------------


@app.post(
    "/vision/analyze",
    summary="Analyse d'image médicale (IRM / Scanner)",
    tags=["Vision"],
    status_code=status.HTTP_200_OK,
)
async def analyze_vision(
    file: UploadFile = File(..., description="Fichier image IRM / Scanner (JPEG, PNG)"),
) -> dict:
    """
    Analyse un fichier IRM ou Scanner avec ResNet50 et retourne :
    - Probabilité d'anomalie [0-100 %]
    - Niveau de risque
    - Heatmap simplifiée (grille 7×7) indiquant les zones d'activation
    """
    allowed_types = {"image/jpeg", "image/png", "image/tiff", "image/bmp"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Type de fichier non supporté : '{file.content_type}'. "
                f"Types acceptés : {', '.join(allowed_types)}"
            ),
        )

    try:
        image_bytes = await file.read()
        from modules.vision import analyze_medical_image
        return analyze_medical_image(image_bytes, filename=file.filename or "image")
    except Exception as exc:
        logger.exception("Erreur dans /vision/analyze")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur interne du module vision : {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# Endpoint Orchestration – /generate-report
# ---------------------------------------------------------------------------


@app.post(
    "/generate-report",
    summary="Génération du rapport médical fusionné",
    tags=["Orchestration"],
    status_code=status.HTTP_200_OK,
)
def generate_report(data: ReportInput) -> dict:
    """
    Fusionne les résultats des modules épilepsie, cancer et MindCare
    en un rapport JSON structuré unique.

    Logique de recommandation automatique :
    - Si stress_level > 7/10 → recommandation exercice de respiration guidée
    - Si l'un des modules retourne un niveau CRITICAL → alerte globale

    Ce rapport est destiné au praticien pour une vue d'ensemble rapide.
    """
    try:
        from modules.epilepsy import predict_epilepsy_risk
        from modules.cancer import predict_chemo_complication
        from modules.mindcare import analyze_sentiment, summarize_journal

        # --- Épilepsie ---
        epilepsy_result = predict_epilepsy_risk(
            sleep_quality=data.epilepsy.sleep_quality,
            stress_level=data.epilepsy.stress_level,
            medication_missed=data.epilepsy.medication_missed,
            aura_present=data.epilepsy.aura_present,
        )

        # --- Cancer ---
        cancer_result = predict_chemo_complication(
            temperature=data.cancer.temperature,
            pain_intensity=data.cancer.pain_intensity,
            nausea=data.cancer.nausea,
            fatigue_level=data.cancer.fatigue_level,
        )

        # --- MindCare (sentiment + résumé) ---
        sentiment = analyze_sentiment(data.mindcare.text)
        summary   = summarize_journal(data.mindcare.text)

        # --- Niveau global de risque ---
        all_levels = [
            epilepsy_result["risk_level"],
            cancer_result["risk_level"],
        ]
        global_risk = "CRITICAL" if "CRITICAL" in all_levels else (
            "HIGH"     if "HIGH"     in all_levels else (
                "MODERATE" if "MODERATE" in all_levels else "LOW"
            )
        )

        # --- Recommandations automatiques ---
        recommendations: list[str] = []

        if data.epilepsy.stress_level > 7:
            recommendations.append(
                "Exercice de respiration guidée (technique 4-7-8) recommandé "
                "pour réduire le niveau de stress élevé."
            )

        if data.epilepsy.medication_missed:
            recommendations.append(
                "Rappeler au patient l'importance de ne pas omettre sa médication antiépileptique."
            )

        if data.cancer.fatigue_level > 7:
            recommendations.append(
                "Fatigue sévère détectée : évaluer l'anémie et adapter le protocole de traitement."
            )

        if sentiment["dominant_emotion"] in ("sadness", "fear"):
            recommendations.append(
                "État émotionnel fragile détecté : orienter vers un suivi psychologique."
            )

        if sentiment["crisis_detected"]:
            recommendations.append(
                "⚠️ URGENCE : Indicateurs de détresse psychologique sévère – "
                "contacter le psychologue référent immédiatement."
            )

        # --- Rapport final ---
        report = {
            "report_version": "1.0",
            "global_risk_level": global_risk,
            "modules": {
                "epilepsy": {
                    "risk_score_percent": epilepsy_result["risk_score_percent"],
                    "risk_level": epilepsy_result["risk_level"],
                    "alert": epilepsy_result.get("alert"),
                },
                "cancer_chemo": {
                    "complication_probability_percent": cancer_result["complication_probability_percent"],
                    "complication_type": cancer_result["complication_type"],
                    "risk_level": cancer_result["risk_level"],
                    "alert": cancer_result.get("alert"),
                },
                "mindcare": {
                    "dominant_emotion": sentiment["dominant_emotion"],
                    "confidence": sentiment["confidence"],
                    "crisis_detected": sentiment["crisis_detected"],
                    "journal_summary": summary.get("summary"),
                },
            },
            "recommendations": recommendations if recommendations else [
                "Aucune recommandation urgente. Maintenir le suivi habituel."
            ],
            "disclaimer": (
                "Ce rapport est généré automatiquement par l'IA MediGuard360. "
                "Il doit être interprété par un professionnel de santé qualifié "
                "et ne constitue pas un diagnostic médical."
            ),
        }

        return report

    except Exception as exc:
        logger.exception("Erreur dans /generate-report")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la génération du rapport : {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# Endpoint de santé
# ---------------------------------------------------------------------------


@app.get("/health", tags=["Système"], summary="Vérification de l'état de l'API")
def health_check() -> dict:
    """Retourne l'état de santé de l'API."""
    return {"status": "ok", "service": "MediGuard360 IA", "version": "1.0.0"}


# ---------------------------------------------------------------------------
# Point d'entrée (développement)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
