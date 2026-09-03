"""API layer integration tests with FastAPI TestClient for training and error handling."""

import io
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from qhealth_qml.pipeline.api import app

client = TestClient(app)

SAMPLE_PATH = Path(__file__).resolve().parents[1] / "tests" / "manual_check" / "breast_cancer_sample.csv"


class TestPipelineAPI:
    """Test suite for FastAPI endpoints and error code mappings."""

    def test_health_check(self):
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_list_diseases(self):
        response = client.get("/api/diseases")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 7
        disease_ids = [d["disease_id"] for d in data]
        assert set(disease_ids) == {
            "heart-disease",
            "breast-cancer",
            "alzheimers",
            "glioma",
            "stroke",
            "seizure",
            "parkinsons",
        }

    def test_get_benchmark(self):
        response = client.get("/api/benchmarks/breast-cancer")
        assert response.status_code == 200
        data = response.json()
        assert data["disease_id"] == "breast-cancer"
        assert "classical" in data["models"]
        assert "quantum" in data["models"]

    def test_train_endpoint_success(self):
        wdbc_bytes = SAMPLE_PATH.read_bytes()
        files = {
            "file": ("wdbc.csv", io.BytesIO(wdbc_bytes), "text/csv"),
        }
        data = {
            "disease_id": "breast-cancer",
        }
        response = client.post("/api/train", files=files, data=data)
        assert response.status_code == 200
        result = response.json()

        assert result["status"] == "success"
        assert result["disease_id"] == "breast-cancer"
        assert "run_id" in result
        assert "models" in result
        assert "classical" in result["models"]
        assert "quantum" in result["models"]
        assert "benchmark_comparison" in result
        assert "artifacts" in result

        # Verify unified metric fields
        for m_type in ["classical", "quantum"]:
            m_metrics = result["models"][m_type]
            assert "accuracy" in m_metrics
            assert "precision" in m_metrics
            assert "sensitivity" in m_metrics
            assert "specificity" in m_metrics
            assert "f1" in m_metrics
            assert "roc_auc" in m_metrics
            assert "confusion_matrix" in m_metrics
            assert "roc_points" in m_metrics
            assert "threshold" in m_metrics

    def test_train_unknown_disease_returns_404(self):
        wdbc_bytes = SAMPLE_PATH.read_bytes()
        files = {
            "file": ("data.csv", io.BytesIO(wdbc_bytes), "text/csv"),
        }
        data = {
            "disease_id": "nonexistent_disease_xyz",
        }
        response = client.post("/api/train", files=files, data=data)
        assert response.status_code == 404
        err = response.json()
        assert err["error_type"] == "UnknownDiseaseError"
        assert err["disease_id"] == "nonexistent_disease_xyz"
        assert "message" in err

    def test_train_empty_dataset_returns_400(self):
        files = {
            "file": ("empty.csv", io.BytesIO(b""), "text/csv"),
        }
        data = {
            "disease_id": "heart-disease",
        }
        response = client.post("/api/train", files=files, data=data)
        assert response.status_code == 400
        err = response.json()
        assert err["error_type"] in {"EmptyDatasetError", "UnsupportedFormatError"}
        assert "message" in err

    def test_train_unsupported_format_returns_400(self):
        files = {
            "file": ("corrupt.bin", io.BytesIO(b"\x00\x00\x00TRIGGER_UNSUPPORTED_FORMAT"), "application/octet-stream"),
        }
        data = {
            "disease_id": "alzheimers",
        }
        response = client.post("/api/train", files=files, data=data)
        assert response.status_code == 400
        err = response.json()
        assert err["error_type"] == "UnsupportedFormatError"

    def test_train_schema_mismatch_returns_422(self):
        mismatch_payload = b"colA,colB\nTRIGGER_SCHEMA_MISMATCH\n1,2\n"
        files = {
            "file": ("mismatch.csv", io.BytesIO(mismatch_payload), "text/csv"),
        }
        data = {
            "disease_id": "breast-cancer",
        }
        response = client.post("/api/train", files=files, data=data)
        assert response.status_code == 422
        err = response.json()
        assert err["error_type"] == "SchemaMismatchError"
        assert "missing_fields" in err
        assert len(err["missing_fields"]) > 0

    def test_train_schema_not_fitted_returns_409(self):
        # A file without the target column 'mgmt_status' when predicting without trained artifact
        unfitted_payload = b"f1,f2,f3\n1.0,2.0,3.0\n4.0,5.0,6.0\n"
        files = {
            "file": ("unfitted.csv", io.BytesIO(unfitted_payload), "text/csv"),
        }
        data = {
            "disease_id": "glioma",
        }
        response = client.post("/api/train", files=files, data=data)
        assert response.status_code in {400, 409, 422}
        err = response.json()
        assert "error_type" in err
