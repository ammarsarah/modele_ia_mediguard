"""
MediGuard360 – Module NLP "MindCare"
======================================
Analyse les journaux thérapeutiques des patients avec trois capacités clés :

1. Analyse de sentiment / émotion  – Détecte tristesse, anxiété, colère, joie…
2. Réponse empathique              – Génère une réponse validant l'émotion.
3. Détection de détresse           – Alertes mots-clés suicidaires / crise majeure.
4. Résumé                          – Condense le journal en 3 lignes pour le praticien.

Modèles HuggingFace utilisés :
  - Émotion        : j-hartmann/emotion-english-distilroberta-base
  - Résumé         : sshleifer/distilbart-cnn-6-6
  
  Note : Pour une utilisation en français, remplacer par des modèles francophones
  (ex. tblard/tf-allocine pour le sentiment, moussaKam/barthez pour le résumé).
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mots-clés de détresse critique (FR + EN)
# ---------------------------------------------------------------------------

_CRISIS_KEYWORDS_FR = [
    "suicid", "me tuer", "finir ma vie", "plus envie de vivre",
    "je veux mourir", "mettre fin à", "mourir", "en finir",
    "je ne veux plus", "disparaître", "m'en aller pour toujours",
    "insupportable", "plus aucun espoir", "tout est perdu",
]

_CRISIS_KEYWORDS_EN = [
    "suicid", "kill myself", "end my life", "no reason to live",
    "want to die", "give up on life", "worthless", "hopeless",
    "can't go on", "disappear forever",
]

_ALL_CRISIS_KEYWORDS = _CRISIS_KEYWORDS_FR + _CRISIS_KEYWORDS_EN

# ---------------------------------------------------------------------------
# Réponses empathiques par émotion détectée
# ---------------------------------------------------------------------------

_EMPATHY_TEMPLATES: dict[str, dict] = {
    "fatigue": {
        "response": (
            "Je comprends ta fatigue, c'est un combat courageux que tu mènes. "
            "Prends ce ressenti au sérieux et avance à ton rythme."
        ),
        "action": "Méditation anti-fatigue (respiration + scan corporel, 5 min) recommandée.",
        "resource": "https://mediguard360.app/meditation/anti-fatigue",
    },
    "sadness": {
        "response": (
            "Je comprends ta tristesse, c'est tout à fait normal de ressentir cela "
            "dans cette période difficile. Tu n'es pas seul(e) dans ce combat courageux."
        ),
        "action": "Méditation guidée de pleine conscience (5 min) recommandée.",
        "resource": "https://mediguard360.app/meditation/pleine-conscience",
    },
    "fear": {
        "response": (
            "L'anxiété que tu ressens est une réaction compréhensible. "
            "Permets-toi de respirer profondément – tu peux traverser ce moment."
        ),
        "action": "Exercice de respiration 4-7-8 recommandé.",
        "resource": "https://mediguard360.app/breathing/4-7-8",
    },
    "anger": {
        "response": (
            "Ta colère est légitime – ce que tu traverses est extrêmement difficile. "
            "Exprimer tes émotions est un pas important vers la guérison."
        ),
        "action": "Journalisation émotionnelle guidée recommandée.",
        "resource": "https://mediguard360.app/journal/guided",
    },
    "joy": {
        "response": (
            "C'est magnifique de lire que tu ressens de la joie aujourd'hui ! "
            "Ces moments positifs sont précieux – continue à les cultiver."
        ),
        "action": "Continuer les activités qui génèrent cette joie.",
        "resource": None,
    },
    "disgust": {
        "response": (
            "Je comprends que certaines situations puissent te paraître accablantes. "
            "Parler de ce que tu ressens avec un professionnel peut t'aider."
        ),
        "action": "Session de soutien psychologique recommandée.",
        "resource": "https://mediguard360.app/support/psychologist",
    },
    "surprise": {
        "response": (
            "Il semble que tu aies vécu quelque chose d'inattendu. "
            "Prends le temps de l'assimiler à ton propre rythme."
        ),
        "action": "Exercice de centrage recommandé.",
        "resource": "https://mediguard360.app/meditation/centering",
    },
    "neutral": {
        "response": (
            "Merci de partager ton vécu. Maintenir ce journal est déjà un acte de soin envers toi-même."
        ),
        "action": "Continue à documenter tes ressentis quotidiens.",
        "resource": None,
    },
}

# Mappage des labels HuggingFace → clés du dictionnaire empathique
_LABEL_MAP = {
    "sadness": "sadness",
    "fear": "fear",
    "anger": "anger",
    "joy": "joy",
    "disgust": "disgust",
    "surprise": "surprise",
    "neutral": "neutral",
    # distilbert SST-2 labels
    "POSITIVE": "joy",
    "NEGATIVE": "sadness",
    # cardiffnlp labels
    "Positive": "joy",
    "Negative": "sadness",
    "Neutral": "neutral",
}

# ---------------------------------------------------------------------------
# Chargement paresseux des pipelines HuggingFace
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_emotion_pipeline():
    """Charge le pipeline d'analyse d'émotion (singleton mis en cache)."""
    from transformers import pipeline as hf_pipeline
    logger.info("MindCare : chargement du modèle d'émotion…")
    return hf_pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None,  # retourne tous les scores
    )


@lru_cache(maxsize=1)
def _get_summarization_pipeline():
    """Charge le pipeline de résumé (singleton mis en cache)."""
    from transformers import pipeline as hf_pipeline
    logger.info("MindCare : chargement du modèle de résumé…")
    return hf_pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-6-6",
    )


# ---------------------------------------------------------------------------
# Détection de mots-clés de crise
# ---------------------------------------------------------------------------

def _detect_crisis(text: str) -> bool:
    """
    Retourne True si le texte contient des indicateurs de détresse sévère
    ou d'idéation suicidaire.
    """
    text_lower = text.lower()
    return any(kw in text_lower for kw in _ALL_CRISIS_KEYWORDS)


# ---------------------------------------------------------------------------
# Détection heuristique de la fatigue / tristesse en français
# ---------------------------------------------------------------------------

_FATIGUE_KEYWORDS_FR = ["fatigué", "épuisé", "à bout", "vidé", "sans énergie"]
_SADNESS_KEYWORDS_FR = ["triste", "déprimé", "malheureux", "chagrin", "découragé"]


def _contains_any_keyword(text: str, keywords: list[str]) -> bool:
    """Retourne True si au moins un mot-clé est présent dans le texte."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in keywords)


def _french_keyword_override(text: str) -> Optional[str]:
    """
    Retourne une émotion si des mots-clés français emblématiques sont détectés,
    permettant de pallier un modèle entraîné principalement en anglais.
    """
    text_lower = text.lower()
    if any(kw in text_lower for kw in _FATIGUE_KEYWORDS_FR):
        return "fear"   # fatigue → anxiété/peur traitée de la même façon
    if any(kw in text_lower for kw in _SADNESS_KEYWORDS_FR):
        return "sadness"
    return None


# ---------------------------------------------------------------------------
# Fonctions publiques
# ---------------------------------------------------------------------------

def analyze_sentiment(text: str) -> dict:
    """
    Analyse le sentiment / l'émotion dominante dans le texte fourni.

    Paramètres
    ----------
    text : texte libre du patient (journal thérapeutique)

    Retour
    ------
    dict contenant :
        - dominant_emotion : str  (sadness, fear, anger, joy, disgust, surprise, neutral)
        - confidence       : float [0-1]
        - all_scores       : dict {label: score}
        - crisis_detected  : bool
    """
    # 1. Vérification des mots-clés de crise
    crisis = _detect_crisis(text)

    # 2. Émotion par mots-clés FR (override rapide)
    fr_override = _french_keyword_override(text)

    # 3. Classification via le modèle
    pipe = _get_emotion_pipeline()
    raw_results = pipe(text[:512])  # limite de tokens BERT

    # Le pipeline avec top_k=None retourne [[{label, score}, …]]
    if isinstance(raw_results[0], list):
        scores_list = raw_results[0]
    else:
        scores_list = raw_results

    all_scores = {item["label"]: round(item["score"], 4) for item in scores_list}
    dominant_label = max(all_scores, key=all_scores.get)
    dominant_emotion = _LABEL_MAP.get(dominant_label, dominant_label.lower())

    # Appliquer l'override FR si la confiance du modèle est faible (< 0.5)
    if fr_override and all_scores.get(dominant_label, 0) < 0.5:
        dominant_emotion = fr_override

    return {
        "dominant_emotion": dominant_emotion,
        "confidence": round(all_scores.get(dominant_label, 0), 4),
        "all_scores": all_scores,
        "crisis_detected": crisis,
    }


def generate_empathic_response(emotion: str) -> dict:
    """
    Génère une réponse empathique et une action thérapeutique
    adaptées à l'émotion détectée.

    Paramètres
    ----------
    emotion : émotion dominante (sadness, fear, anger, joy, disgust, surprise, neutral)

    Retour
    ------
    dict contenant :
        - empathic_message : str
        - suggested_action : str
        - resource_link    : str | None
    """
    template = _EMPATHY_TEMPLATES.get(emotion, _EMPATHY_TEMPLATES["neutral"])
    return {
        "empathic_message": template["response"],
        "suggested_action": template["action"],
        "resource_link": template["resource"],
    }


def _format_summary_three_lines(text: str) -> str:
    """
    Formate un texte en exactement 3 lignes pour l'interface praticien.
    """
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return "-\n-\n-"

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean) if s.strip()]
    if len(sentences) >= 3:
        lines = sentences[:3]
    else:
        words = clean.split()
        chunks = np.array_split(np.array(words, dtype=object), 3)
        lines = [" ".join(chunk.tolist()).strip() for chunk in chunks]

    lines = [line if line else "-" for line in lines[:3]]
    while len(lines) < 3:
        lines.append("-")
    return "\n".join(lines)


def summarize_journal(text: str, max_length: int = 130, min_length: int = 40) -> dict:
    """
    Condense le journal thérapeutique en 3 lignes pour le praticien.

    Paramètres
    ----------
    text       : texte complet du journal (≥ 50 mots recommandés)
    max_length : longueur maximale du résumé en tokens (défaut 130)
    min_length : longueur minimale du résumé en tokens (défaut 40)

    Retour
    ------
    dict contenant :
        - summary         : str, résumé condensé
        - original_length : int, nombre de mots dans l'original
        - summary_length  : int, nombre de mots dans le résumé
    """
    pipe = _get_summarization_pipeline()
    word_count = len(text.split())

    # Le modèle de résumé nécessite un texte suffisamment long
    if word_count < 30:
        summary_text = _format_summary_three_lines(text)
        return {
            "summary": summary_text,
            "original_length": word_count,
            "summary_length": len(summary_text.split()),
            "note": "Texte court détecté – résumé formaté automatiquement en 3 lignes.",
        }

    result = pipe(
        text[:1024],
        max_length=max_length,
        min_length=min_length,
        do_sample=False,
    )
    summary_text: str = _format_summary_three_lines(result[0]["summary_text"])

    return {
        "summary": summary_text,
        "original_length": word_count,
        "summary_length": len(summary_text.split()),
    }


def analyze_journal(text: str) -> dict:
    """
    Analyse complète d'une entrée de journal thérapeutique.

    Combine : analyse d'émotion + réponse empathique + détection de crise + résumé.

    Paramètres
    ----------
    text : texte libre du journal du patient

    Retour
    ------
    dict JSON structuré contenant tous les résultats MindCare
    """
    sentiment_result = analyze_sentiment(text)
    emotion = sentiment_result["dominant_emotion"]

    # Exigence fonctionnelle : mots-clés explicites "triste"/"fatigué"
    # déclenchent systématiquement une réponse empathique adaptée.
    if _contains_any_keyword(text, _FATIGUE_KEYWORDS_FR):
        emotion = "fatigue"
        sentiment_result["dominant_emotion"] = "fatigue"
        sentiment_result["context_override"] = "fatigue_keyword"
    elif _contains_any_keyword(text, _SADNESS_KEYWORDS_FR):
        emotion = "sadness"
        sentiment_result["dominant_emotion"] = "sadness"
        sentiment_result["context_override"] = "sadness_keyword"

    empathy_result = generate_empathic_response(emotion)
    summary_result = summarize_journal(text)

    result: dict = {
        "module": "mindcare",
        "sentiment_analysis": sentiment_result,
        "empathic_response": empathy_result,
        "journal_summary": summary_result,
    }

    # Alerte de détresse immédiate
    if sentiment_result["crisis_detected"]:
        result["crisis_alert"] = {
            "severity": "CRITICAL",
            "message": (
                "⚠️ Indicateurs de détresse majeure ou d'idéation suicidaire détectés. "
                "Le psychologue référent doit être alerté immédiatement."
            ),
            "action_required": "Contacter le psychologue référent / équipe de crise.",
            "emergency_contact": "Numéro de crise : 3114 (France – Numéro national de prévention du suicide).",
        }

    return result
