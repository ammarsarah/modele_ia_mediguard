"""
MediGuard360 – Module Vision (Computer Vision)
================================================
Analyse des fichiers IRM / Scanner avec ResNet50 (Transfer Learning).

Pipeline :
  1. Chargement de l'image médicale (JPEG/PNG)
  2. Prétraitement standard ImageNet (resize 224×224, normalisation)
  3. Inférence via ResNet50 pré-entraîné sur ImageNet
  4. Génération d'une heatmap simplifiée (carte d'activation)
  5. Calcul d'un score d'anomalie basé sur l'entropie des activations

AVERTISSEMENT : Ce module est un outil d'aide au diagnostic.
Le score d'anomalie doit TOUJOURS être interprété par un radiologue qualifié.
Il ne remplace en aucun cas un avis médical professionnel.
"""

from __future__ import annotations

import io
import logging
import math
from functools import lru_cache
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chargement paresseux du modèle (singleton)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_resnet_model():
    """
    Charge ResNet50 pré-entraîné sur ImageNet.
    Le modèle est utilisé comme extracteur de caractéristiques.
    En production, remplacer par un modèle fine-tuné sur des données médicales.
    """
    import tensorflow as tf
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.applications.resnet50 import preprocess_input

    logger.info("Vision : chargement du modèle ResNet50…")
    model = ResNet50(weights="imagenet", include_top=True)
    logger.info("Vision : ResNet50 chargé (%d paramètres)", model.count_params())
    return model


@lru_cache(maxsize=1)
def _get_feature_model():
    """
    Retourne un modèle secondaire extrayant les activations
    de la couche 'conv5_block3_out' (avant la classification finale).
    Ces activations servent à générer la heatmap simplifiée.
    """
    import tensorflow as tf
    from tensorflow.keras import Model

    base_model = _get_resnet_model()
    feature_model = Model(
        inputs=base_model.input,
        outputs=base_model.get_layer("conv5_block3_out").output,
    )
    return feature_model


# ---------------------------------------------------------------------------
# Prétraitement de l'image
# ---------------------------------------------------------------------------

def _preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Charge et prétraite une image médicale pour l'inférence ResNet50.

    Étapes :
    - Décodage de l'image (PIL)
    - Conversion en RGB (au cas où l'image soit en niveaux de gris)
    - Redimensionnement en 224×224
    - Application du prétraitement ImageNet (soustraction moyenne, etc.)

    Paramètres
    ----------
    image_bytes : contenu binaire du fichier image

    Retour
    ------
    np.ndarray de forme (1, 224, 224, 3) prêt pour l'inférence
    """
    from PIL import Image
    from tensorflow.keras.applications.resnet50 import preprocess_input

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32)          # (224, 224, 3)
    arr = np.expand_dims(arr, axis=0)              # (1, 224, 224, 3)
    arr = preprocess_input(arr)                    # normalisation ImageNet
    return arr


# ---------------------------------------------------------------------------
# Calcul du score d'anomalie
# ---------------------------------------------------------------------------

def _compute_anomaly_score(predictions: np.ndarray) -> float:
    """
    Dérive un score d'anomalie [0-1] à partir du vecteur de probabilités
    produit par ResNet50.

    Heuristique :
      - Une image médicale typique ne ressemble à aucune classe ImageNet précise.
      - L'entropie de Shannon de la distribution de probabilités est élevée
        quand aucune classe ne domine → image atypique → probabilité d'anomalie haute.
      - On normalise l'entropie maximale (log2(1000) pour 1000 classes) en [0-1].

    Note : Cette heuristique est un placeholder.
    En production, utiliser un modèle fine-tuné sur des annotations radiologiques.
    """
    # Stabilité numérique
    probs = predictions[0] + 1e-10
    probs /= probs.sum()

    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = math.log2(len(probs))  # entropie maximale = log2(nb_classes)
    normalized_entropy = entropy / max_entropy

    # Plus l'entropie est élevée (image atypique), plus le score d'anomalie est élevé
    return round(float(normalized_entropy), 4)


# ---------------------------------------------------------------------------
# Génération de la heatmap simplifiée
# ---------------------------------------------------------------------------

def _generate_heatmap(image_bytes: bytes) -> dict:
    """
    Génère une heatmap simplifiée d'activation basée sur les feature maps
    de la dernière couche convolutive de ResNet50 (pseudo Grad-CAM).

    Retour
    ------
    dict contenant :
        - grid_size      : int, taille de la grille (7×7 pour ResNet50)
        - activation_map : liste 7×7 de valeurs [0-1]
        - peak_region    : dict {row, col, value} coordonnée de la zone la plus active
    """
    feature_model = _get_feature_model()
    X = _preprocess_image(image_bytes)

    feature_maps = feature_model.predict(X, verbose=0)  # (1, 7, 7, 2048)
    # Moyenne des 2048 canaux pour obtenir une carte 7×7
    heatmap_2d = np.mean(feature_maps[0], axis=-1)      # (7, 7)

    # Normalisation [0-1]
    hm_min, hm_max = heatmap_2d.min(), heatmap_2d.max()
    if hm_max > hm_min:
        heatmap_norm = (heatmap_2d - hm_min) / (hm_max - hm_min)
    else:
        heatmap_norm = np.zeros_like(heatmap_2d)

    heatmap_list = [[round(float(v), 4) for v in row] for row in heatmap_norm.tolist()]

    # Zone d'activation maximale
    peak_row, peak_col = np.unravel_index(np.argmax(heatmap_norm), heatmap_norm.shape)

    return {
        "grid_size": 7,
        "activation_map": heatmap_list,
        "peak_region": {
            "row": int(peak_row),
            "col": int(peak_col),
            "normalized_value": round(float(heatmap_norm[peak_row, peak_col]), 4),
            "description": (
                f"Zone d'activation maximale détectée à la cellule "
                f"({int(peak_row)}, {int(peak_col)}) sur une grille 7×7. "
                "Correspond à ~"
                f"{int(peak_row * 32)}-{int((peak_row + 1) * 32)}px (hauteur) × "
                f"{int(peak_col * 32)}-{int((peak_col + 1) * 32)}px (largeur) "
                "sur l'image originale 224×224."
            ),
        },
    }


# ---------------------------------------------------------------------------
# Fonction principale d'analyse
# ---------------------------------------------------------------------------

def analyze_medical_image(image_bytes: bytes, filename: str = "image") -> dict:
    """
    Analyse un fichier IRM / Scanner et retourne un rapport d'anomalie structuré.

    Paramètres
    ----------
    image_bytes : contenu binaire du fichier image (JPEG / PNG / TIFF)
    filename    : nom original du fichier (pour le rapport)

    Retour
    ------
    dict JSON contenant :
        - anomaly_probability_percent : float [0-100 %]
        - risk_level                  : str "LOW" | "MODERATE" | "HIGH" | "CRITICAL"
        - heatmap                     : dict (grille d'activation 7×7)
        - disclaimer                  : str (avertissement médical obligatoire)
    """
    model = _get_resnet_model()
    X = _preprocess_image(image_bytes)

    predictions = model.predict(X, verbose=0)          # (1, 1000)
    anomaly_score = _compute_anomaly_score(predictions)
    anomaly_percent = round(anomaly_score * 100, 2)

    # Heatmap
    heatmap = _generate_heatmap(image_bytes)

    if anomaly_percent <= 25:
        risk_level = "LOW"
        interpretation = "Aucune anomalie significative détectée."
    elif anomaly_percent <= 50:
        risk_level = "MODERATE"
        interpretation = "Anomalie possible – surveillance recommandée."
    elif anomaly_percent <= 75:
        risk_level = "HIGH"
        interpretation = "Anomalie probable – consultation radiologique urgente."
    else:
        risk_level = "CRITICAL"
        interpretation = "Anomalie très probable – avis médical immédiat requis."

    return {
        "module": "vision",
        "filename": filename,
        "anomaly_probability_percent": anomaly_percent,
        "risk_level": risk_level,
        "interpretation": interpretation,
        "heatmap": heatmap,
        "disclaimer": (
            "⚠️ Ce résultat est un outil d'aide au diagnostic destiné au personnel médical. "
            "Il ne remplace pas l'interprétation d'un radiologue qualifié. "
            "Toute décision médicale doit être validée par un professionnel de santé."
        ),
    }
