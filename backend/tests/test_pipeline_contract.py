"""Contract compliance tests for standardization and model training interfaces."""

import io
import pytest
import numpy as np

from qhealth_qml.pipeline.exceptions import (
    EmptyDatasetError,
    SchemaMismatchError,
    SchemaNotFittedError,
    StandardizationError,
    UnknownDiseaseError,
    UnsupportedFormatError,
)
from qhealth_qml.pipeline.fakes import (
    fake_list_supported_diseases,
    fake_standardize,
    fake_train_disease_models,
)
from qhealth_qml.pipeline.disease_registry import disease_registry


class TestStandardizerContract:
    """Test fake_standardize compliance with the documented contract."""

    def test_fake_standardize_success_all_seven_diseases(self):
        diseases = ["heart-disease", "breast-cancer", "alzheimers", "glioma", "stroke", "seizure", "parkinsons"]
        sample_csv = "feat1,feat2,feat3\n1.0,2.0,3.0\n4.0,5.0,6.0\n7.0,8.0,9.0\n10.0,11.0,12.0\n"

        for d_id in diseases:
            X, y = fake_standardize(sample_csv, d_id)
            assert isinstance(X, np.ndarray), f"X must be numpy array for {d_id}"
            assert isinstance(y, np.ndarray), f"y must be numpy array for {d_id}"
            assert X.dtype == np.float64
            assert len(X) == len(y)
            assert set(np.unique(y)).issubset({0, 1})
            assert len(X) >= 4

    def test_unknown_disease_raises_typed_error(self):
        with pytest.raises(UnknownDiseaseError) as exc_info:
            fake_standardize("a,b\n1,2\n3,4\n", "unsupported-condition")
        assert exc_info.value.disease_id == "unsupported-condition"
        assert issubclass(UnknownDiseaseError, StandardizationError)

    def test_unsupported_format_raises_typed_error(self):
        binary_corrupt = b"\x00\x01\x02\x00\xff"
        with pytest.raises(UnsupportedFormatError) as exc_info:
            fake_standardize(binary_corrupt, "breast-cancer")
        assert issubclass(UnsupportedFormatError, StandardizationError)

    def test_empty_dataset_raises_typed_error(self):
        with pytest.raises(EmptyDatasetError):
            fake_standardize("", "breast-cancer")

        header_only = "col1,col2,col3\n"
        with pytest.raises(EmptyDatasetError):
            fake_standardize(header_only, "breast-cancer")

    def test_schema_mismatch_raises_typed_error(self):
        with pytest.raises(SchemaMismatchError) as exc_info:
            fake_standardize(
                "col1,col2\n1,2\n3,4\n",
                "breast-cancer",
                simulate_missing_columns=["radius_mean", "texture_mean"],
            )
        assert exc_info.value.missing_fields == ["radius_mean", "texture_mean"]
        assert issubclass(SchemaMismatchError, StandardizationError)

    def test_schema_not_fitted_raises_typed_error(self):
        with pytest.raises(SchemaNotFittedError) as exc_info:
            fake_standardize(
                "col1,col2\n1,2\n3,4\n",
                "breast-cancer",
                simulate_unfitted_schema=True,
            )
        assert exc_info.value.disease_id == "breast-cancer"
        assert issubclass(SchemaNotFittedError, StandardizationError)

    def test_standardizer_disease_cross_check(self):
        status = disease_registry.validate_against_standardizer(fake_list_supported_diseases)
        assert all(status.values()), f"Standardizer cross-check failed: {status}"


class TestModelInterfaceContract:
    """Test model interface returning classical and quantum outputs."""

    def test_fake_train_disease_models_interface(self):
        X_train = np.random.randn(30, 6)
        y_train = np.array([0, 1] * 15)
        X_test = np.random.randn(10, 6)
        y_test = np.array([0, 1] * 5)

        results = fake_train_disease_models(X_train, y_train, X_test, y_test)
        assert "classical" in results
        assert "quantum" in results

        for key in ["classical", "quantum"]:
            out = results[key]
            d = out.to_dict() if hasattr(out, "to_dict") else out
            assert "model" in d
            assert "y_pred" in d
            assert "y_prob" in d
            assert "train_time" in d
            assert "infer_time" in d
            assert len(d["y_pred"]) == len(y_test)
            assert len(d["y_prob"]) == len(y_test)
