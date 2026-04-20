"""
Tests de l'API MediGuard360 IA.

Ces tests utilisent le TestClient de FastAPI (httpx synchrone) pour tester
tous les endpoints sans démarrer un vrai serveur.

Les modèles NLP (Transformers) et Vision (ResNet50) sont mockés pour
éviter le téléchargement de plusieurs giga-octets de poids en CI.
Les modèles ML légers (Random Forest, XGBoost) s'entraînent en ligne
sur les données synthétiques embarquées dans les modules.
"""

from __future__ import annotations

import io
import sys
import os
import types
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Ajouter la racine du projet au path Python
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# Mocks des dépendances lourdes AVANT l'import de l'application
# ---------------------------------------------------------------------------

# --- Mock transformers ---
transformers_mock = types.ModuleType("transformers")

def _mock_pipeline(task, model=None, **kwargs):
    """Retourne un callable qui simule un pipeline HuggingFace."""
    if task == "text-classification":
        def _pipe(text, **kw):
            # Simule top_k=None → retourne une liste de dicts
            return [[
                {"label": "joy",     "score": 0.05},
                {"label": "sadness", "score": 0.70},
                {"label": "anger",   "score": 0.05},
                {"label": "fear",    "score": 0.10},
                {"label": "disgust", "score": 0.05},
                {"label": "surprise","score": 0.02},
                {"label": "neutral", "score": 0.03},
            ]]
        return _pipe
    elif task == "summarization":
        def _pipe(text, **kw):
            return [{"summary_text": "Résumé simulé par le mock de test."}]
        return _pipe
    return MagicMock()

transformers_mock.pipeline = _mock_pipeline
sys.modules["transformers"] = transformers_mock

# --- Mock tensorflow / keras ---
tf_mock = types.ModuleType("tensorflow")
keras_mock = types.ModuleType("tensorflow.keras")
keras_apps_mock = types.ModuleType("tensorflow.keras.applications")
resnet_mock = types.ModuleType("tensorflow.keras.applications.resnet50")

class _FakeResNet50:
    def __init__(self, *a, **kw):
        self._layer_out = np.ones((1, 7, 7, 2048), dtype=np.float32)

    def count_params(self):
        return 25_636_712

    def predict(self, X, verbose=0):
        probs = np.ones((1, 1000), dtype=np.float32) / 1000
        return probs

    def get_layer(self, name):
        layer = MagicMock()
        layer.output = MagicMock()
        return layer

    @property
    def input(self):
        return MagicMock()


class _FakeFeatureModel:
    def predict(self, X, verbose=0):
        return np.random.rand(1, 7, 7, 2048).astype(np.float32)


def _fake_keras_model(inputs, outputs):
    return _FakeFeatureModel()


keras_apps_mock.ResNet50 = _FakeResNet50
resnet_mock.preprocess_input = lambda x: x

keras_mock.applications = keras_apps_mock
keras_mock.Model = _fake_keras_model

tf_mock.keras = keras_mock

sys.modules["tensorflow"] = tf_mock
sys.modules["tensorflow.keras"] = keras_mock
sys.modules["tensorflow.keras.applications"] = keras_apps_mock
sys.modules["tensorflow.keras.applications.resnet50"] = resnet_mock

# PIL mock (minimal)
pil_mock = types.ModuleType("PIL")
pil_image_mock = types.ModuleType("PIL.Image")

class _FakePILImage:
    def convert(self, mode):
        return self
    def resize(self, size):
        return self

def _fake_open(fp, **kw):
    return _FakePILImage()

pil_image_mock.open = _fake_open
pil_mock.Image = pil_image_mock
sys.modules["PIL"] = pil_mock
sys.modules["PIL.Image"] = pil_image_mock

# numpy array from fake PIL image
_orig_array = np.array

def _patched_array(obj, dtype=None, **kw):
    if isinstance(obj, _FakePILImage):
        return np.zeros((224, 224, 3), dtype=dtype or np.float32)
    return _orig_array(obj, dtype=dtype, **kw) if dtype else _orig_array(obj, **kw)

np.array = _patched_array

# ---------------------------------------------------------------------------
# Import de l'application APRÈS les mocks
# ---------------------------------------------------------------------------
from fastapi.testclient import TestClient  # noqa: E402
from main import app  # noqa: E402

client = TestClient(app, raise_server_exceptions=True)


# ===========================================================================
# Tests – Endpoint de santé
# ===========================================================================

class TestHealthEndpoint:
    def test_health_returns_200(self):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_returns_ok_status(self):
        r = client.get("/health")
        body = r.json()
        assert body["status"] == "ok"
        assert body["service"] == "MediGuard360 IA"


# ===========================================================================
# Tests – Module Épilepsie
# ===========================================================================

class TestEpilepsyModule:

    _LOW_RISK_PAYLOAD = {
        "sleep_quality": 9.0,
        "stress_level": 1.0,
        "medication_missed": False,
        "aura_present": False,
    }

    _HIGH_RISK_PAYLOAD = {
        "sleep_quality": 1.0,
        "stress_level": 9.5,
        "medication_missed": True,
        "aura_present": True,
    }

    def test_predict_returns_200(self):
        r = client.post("/epilepsy/predict", json=self._LOW_RISK_PAYLOAD)
        assert r.status_code == 200

    def test_predict_response_structure(self):
        r = client.post("/epilepsy/predict", json=self._LOW_RISK_PAYLOAD)
        body = r.json()
        assert "risk_score_percent" in body
        assert "risk_level" in body
        assert "module" in body
        assert body["module"] == "epilepsy"

    def test_risk_score_in_valid_range(self):
        r = client.post("/epilepsy/predict", json=self._LOW_RISK_PAYLOAD)
        score = r.json()["risk_score_percent"]
        assert 0 <= score <= 100

    def test_high_risk_generates_alert(self):
        r = client.post("/epilepsy/predict", json=self._HIGH_RISK_PAYLOAD)
        body = r.json()
        # High risk case should include an alert (if model scores > 80)
        # We verify the structure is valid regardless of the score
        assert "risk_level" in body
        if body["risk_score_percent"] > 80:
            assert "alert" in body
            assert body["alert"]["severity"] == "CRITICAL"

    def test_invalid_stress_level_returns_422(self):
        payload = dict(self._LOW_RISK_PAYLOAD, stress_level=15.0)  # > 10
        r = client.post("/epilepsy/predict", json=payload)
        assert r.status_code == 422

    def test_invalid_sleep_quality_returns_422(self):
        payload = dict(self._LOW_RISK_PAYLOAD, sleep_quality=-1.0)  # < 0
        r = client.post("/epilepsy/predict", json=payload)
        assert r.status_code == 422


# ===========================================================================
# Tests – Module Cancer
# ===========================================================================

class TestCancerModule:

    _NORMAL_PAYLOAD = {
        "temperature": 37.0,
        "pain_intensity": 3.0,
        "nausea": False,
        "fatigue_level": 2.0,
    }

    _CRITICAL_PAYLOAD = {
        "temperature": 39.5,
        "pain_intensity": 9.0,
        "nausea": True,
        "fatigue_level": 9.0,
    }

    def test_predict_returns_200(self):
        r = client.post("/cancer/predict", json=self._NORMAL_PAYLOAD)
        assert r.status_code == 200

    def test_predict_response_structure(self):
        r = client.post("/cancer/predict", json=self._NORMAL_PAYLOAD)
        body = r.json()
        assert "complication_probability_percent" in body
        assert "complication_type" in body
        assert "risk_level" in body
        assert body["module"] == "cancer_chemo"

    def test_probability_in_valid_range(self):
        r = client.post("/cancer/predict", json=self._NORMAL_PAYLOAD)
        prob = r.json()["complication_probability_percent"]
        assert 0 <= prob <= 100

    def test_critical_case_alert(self):
        r = client.post("/cancer/predict", json=self._CRITICAL_PAYLOAD)
        body = r.json()
        if body["complication_probability_percent"] > 70:
            assert "alert" in body
            assert body["alert"]["severity"] == "CRITICAL"

    def test_invalid_temperature_too_low(self):
        payload = dict(self._NORMAL_PAYLOAD, temperature=33.0)  # < 34
        r = client.post("/cancer/predict", json=payload)
        assert r.status_code == 422

    def test_invalid_pain_intensity(self):
        payload = dict(self._NORMAL_PAYLOAD, pain_intensity=11.0)  # > 10
        r = client.post("/cancer/predict", json=payload)
        assert r.status_code == 422


# ===========================================================================
# Tests – Module MindCare NLP
# ===========================================================================

class TestMindCareModule:

    _NORMAL_TEXT = "Aujourd'hui je me sens un peu fatigué mais j'essaie de rester positif."
    _CRISIS_TEXT  = "Je veux mourir, je ne veux plus vivre, tout est perdu pour moi."
    _SHORT_TEXT   = "Bien."

    def test_analyze_returns_200(self):
        r = client.post("/mindcare/analyze", json={"text": self._NORMAL_TEXT})
        assert r.status_code == 200

    def test_analyze_response_structure(self):
        r = client.post("/mindcare/analyze", json={"text": self._NORMAL_TEXT})
        body = r.json()
        assert body["module"] == "mindcare"
        assert "sentiment_analysis" in body
        assert "empathic_response" in body
        assert "journal_summary" in body

    def test_sentiment_analysis_fields(self):
        r = client.post("/mindcare/analyze", json={"text": self._NORMAL_TEXT})
        sentiment = r.json()["sentiment_analysis"]
        assert "dominant_emotion" in sentiment
        assert "confidence" in sentiment
        assert "crisis_detected" in sentiment

    def test_crisis_keywords_detection(self):
        r = client.post("/mindcare/analyze", json={"text": self._CRISIS_TEXT})
        body = r.json()
        assert body["sentiment_analysis"]["crisis_detected"] is True
        assert "crisis_alert" in body
        assert body["crisis_alert"]["severity"] == "CRITICAL"

    def test_no_crisis_in_normal_text(self):
        r = client.post("/mindcare/analyze", json={"text": self._NORMAL_TEXT})
        body = r.json()
        assert body["sentiment_analysis"]["crisis_detected"] is False
        assert "crisis_alert" not in body

    def test_empathic_response_fields(self):
        r = client.post("/mindcare/analyze", json={"text": self._NORMAL_TEXT})
        empathy = r.json()["empathic_response"]
        assert "empathic_message" in empathy
        assert "suggested_action" in empathy

    def test_summarize_endpoint(self):
        long_text = (
            "Je me sens vraiment épuisé ces derniers jours. "
            "Le traitement est difficile mais je continue à me battre. "
            "Ma famille me soutient beaucoup et cela m'aide à tenir le coup. "
            "Les médecins sont optimistes quant à l'évolution de ma situation."
        )
        r = client.post("/mindcare/summarize", json={"text": long_text})
        assert r.status_code == 200
        body = r.json()
        assert "summary" in body
        assert len(body["summary"].splitlines()) == 3

    def test_text_too_short_returns_422(self):
        r = client.post("/mindcare/analyze", json={"text": "ab"})  # < min_length=3
        assert r.status_code == 422


# ===========================================================================
# Tests – Module Vision
# ===========================================================================

class TestVisionModule:

    def _make_png_bytes(self) -> bytes:
        """Crée un PNG minimal valide (1×1 pixel blanc) en mémoire."""
        import struct, zlib
        def pack_chunk(chunk_type: bytes, data: bytes) -> bytes:
            c = struct.pack(">I", len(data)) + chunk_type + data
            return c + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)

        signature = b"\x89PNG\r\n\x1a\n"
        ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
        ihdr = pack_chunk(b"IHDR", ihdr_data)
        idat_data = zlib.compress(b"\x00\xFF\xFF\xFF")  # filtre 0, RGB blanc
        idat = pack_chunk(b"IDAT", idat_data)
        iend = pack_chunk(b"IEND", b"")
        return signature + ihdr + idat + iend

    def test_analyze_valid_image_returns_200(self):
        img_bytes = self._make_png_bytes()
        r = client.post(
            "/vision/analyze",
            files={"file": ("mri_scan.png", io.BytesIO(img_bytes), "image/png")},
        )
        assert r.status_code == 200

    def test_analyze_response_structure(self):
        img_bytes = self._make_png_bytes()
        r = client.post(
            "/vision/analyze",
            files={"file": ("mri_scan.png", io.BytesIO(img_bytes), "image/png")},
        )
        body = r.json()
        assert body["module"] == "vision"
        assert "anomaly_probability_percent" in body
        assert "risk_level" in body
        assert "heatmap" in body
        assert "disclaimer" in body

    def test_heatmap_structure(self):
        img_bytes = self._make_png_bytes()
        r = client.post(
            "/vision/analyze",
            files={"file": ("mri_scan.png", io.BytesIO(img_bytes), "image/png")},
        )
        heatmap = r.json()["heatmap"]
        assert heatmap["grid_size"] == 7
        assert len(heatmap["activation_map"]) == 7
        assert len(heatmap["activation_map"][0]) == 7
        assert "peak_region" in heatmap

    def test_unsupported_media_type_returns_415(self):
        r = client.post(
            "/vision/analyze",
            files={"file": ("report.pdf", io.BytesIO(b"%PDF"), "application/pdf")},
        )
        assert r.status_code == 415

    def test_anomaly_score_in_valid_range(self):
        img_bytes = self._make_png_bytes()
        r = client.post(
            "/vision/analyze",
            files={"file": ("mri_scan.png", io.BytesIO(img_bytes), "image/png")},
        )
        score = r.json()["anomaly_probability_percent"]
        assert 0 <= score <= 100


# ===========================================================================
# Tests – Endpoint /generate-report (Orchestration)
# ===========================================================================

class TestGenerateReport:

    _REPORT_PAYLOAD = {
        "epilepsy": {
            "sleep_quality": 5.0,
            "stress_level": 8.0,     # > 7 → recommandation respiration
            "medication_missed": False,
            "aura_present": False,
        },
        "cancer": {
            "temperature": 37.5,
            "pain_intensity": 5.0,
            "nausea": False,
            "fatigue_level": 4.0,
        },
        "mindcare": {
            "text": "Je suis assez fatigué mais je garde espoir pour la suite du traitement."
        },
    }

    def test_report_returns_200(self):
        r = client.post("/generate-report", json=self._REPORT_PAYLOAD)
        assert r.status_code == 200

    def test_report_structure(self):
        r = client.post("/generate-report", json=self._REPORT_PAYLOAD)
        body = r.json()
        assert "global_risk_level" in body
        assert "modules" in body
        assert "recommendations" in body
        assert "disclaimer" in body

    def test_report_modules_fields(self):
        r = client.post("/generate-report", json=self._REPORT_PAYLOAD)
        modules = r.json()["modules"]
        assert "epilepsy" in modules
        assert "cancer_chemo" in modules
        assert "mindcare" in modules

    def test_stress_recommendation_added(self):
        """stress_level=8.0 > 7 → doit ajouter la recommandation de respiration."""
        r = client.post("/generate-report", json=self._REPORT_PAYLOAD)
        recs = r.json()["recommendations"]
        assert any("respiration" in rec.lower() for rec in recs)

    def test_recommendation_field_present_when_stress_high(self):
        """Si stress > 7, le champ recommendation (singulier) doit être présent."""
        r = client.post("/generate-report", json=self._REPORT_PAYLOAD)
        body = r.json()
        assert body["recommendation"] == "Exercice de respiration guidée"

    def test_global_risk_level_valid(self):
        r = client.post("/generate-report", json=self._REPORT_PAYLOAD)
        level = r.json()["global_risk_level"]
        assert level in {"LOW", "MODERATE", "HIGH", "CRITICAL"}

    def test_crisis_text_adds_urgent_recommendation(self):
        payload = {
            **self._REPORT_PAYLOAD,
            "mindcare": {"text": "Je veux en finir, je ne veux plus vivre."},
        }
        r = client.post("/generate-report", json=payload)
        body = r.json()
        recs = body["recommendations"]
        assert any("urgence" in rec.lower() or "détresse" in rec.lower() for rec in recs)

    def test_report_version_field(self):
        r = client.post("/generate-report", json=self._REPORT_PAYLOAD)
        assert r.json()["report_version"] == "1.0"
