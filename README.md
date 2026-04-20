# MediGuard360 – Cerveau IA (FastAPI)

Ce dépôt contient le backend IA Python de **MediGuard360** (intégrable avec Angular/Spring Boot).
Le système repose sur l'analyse de **formulaires médicaux**, **textes thérapeutiques** et **images médicales** (sans IoT).

## ✅ Modules inclus

### 1) Module Épilepsie (`/epilepsy/predict`)
- Modèle : **Random Forest** (scikit-learn)
- Entrées : qualité sommeil, stress, oubli médicament, auras
- Sortie : score de risque (%) + niveau de risque + alerte critique si > 80%

### 2) Module Cancer chimio (`/cancer/predict`)
- Modèle : **XGBoost**
- Entrées : température, douleur (1-10), nausées, fatigue
- Sortie : probabilité de complication (%) + type de complication probable

### 3) Module NLP MindCare (`/mindcare/analyze`, `/mindcare/summarize`)
- Analyse d'émotion (Transformers) : tristesse, anxiété/peur, colère, joie…
- Réponse empathique + action proposée
- Détection de mots-clés de détresse majeure (alerte psychologue)
- Résumé thérapeutique formaté en **3 lignes**

### 4) Module Vision (`/vision/analyze`)
- Modèle : **TensorFlow/Keras ResNet50**
- Analyse IRM/Scanner (JPEG/PNG/TIFF/BMP)
- Sortie : probabilité d'anomalie + heatmap simplifiée (grille 7x7)

### 5) Orchestration (`/generate-report`)
- Fusion des scores Épilepsie + Cancer + bilan MindCare
- Rapport JSON structuré global
- Si stress > 7/10 : ajout `recommendation: "Exercice de respiration guidée"`

---

## 📦 Bibliothèques à installer

Le projet utilise les librairies principales suivantes :
- FastAPI, Uvicorn
- NumPy, Pandas, scikit-learn, XGBoost, Joblib
- TensorFlow, Pillow
- Transformers, Torch, SentencePiece
- Pydantic, HTTPX, Pytest

Installation (recommandée via `requirements.txt`) :

```bash
cd /home/runner/work/modele_ia_mediguard/modele_ia_mediguard
python -m pip install -r requirements.txt
```

---

## ▶️ Comment lancer le projet (CMD / terminal)

```bash
cd /home/runner/work/modele_ia_mediguard/modele_ia_mediguard
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

API docs :
- Swagger UI : `http://127.0.0.1:8000/docs`
- ReDoc : `http://127.0.0.1:8000/redoc`

---

## 🧪 Lancer les tests

```bash
cd /home/runner/work/modele_ia_mediguard/modele_ia_mediguard
python -m pytest tests/test_api.py -v
```

---

## ⚠️ Note médicale

Les résultats sont des aides à la décision clinique.  
Ils ne remplacent pas l'évaluation d'un professionnel de santé qualifié.
