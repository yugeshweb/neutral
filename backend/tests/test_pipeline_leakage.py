"""Leakage safety tests verifying train-fold-only fitting and clean stratification."""

import numpy as np
import pytest

from qhealth_qml.pipeline.preprocessor import (
    LeakageSafePreprocessor,
    stratified_split,
)


class TestLeakageSafety:
    """Test suite ensuring zero data leakage between train and test folds."""

    def test_stratified_split_preserves_class_ratio(self):
        # 80 zeros and 20 ones (20% positive prevalence)
        X = np.arange(200).reshape(100, 2)
        y = np.array([0] * 80 + [1] * 20)

        X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=0.2, random_state=42)

        assert len(X_train) == 80
        assert len(X_test) == 20
        # Positive ratio in train: 16/80 = 20%
        assert np.mean(y_train == 1) == 0.20
        # Positive ratio in test: 4/20 = 20%
        assert np.mean(y_test == 1) == 0.20

    def test_preprocessing_fitted_strictly_on_train_fold(self):
        # Create X_train with mean = 10.0
        rng = np.random.RandomState(42)
        X_train = rng.normal(loc=10.0, scale=2.0, size=(50, 4))
        # Insert NaNs in train fold
        X_train[0, 0] = np.nan
        X_train[1, 1] = np.nan

        # Create X_test with extreme outlier mean = 1000.0
        X_test = rng.normal(loc=1000.0, scale=2.0, size=(20, 4))
        # Insert NaNs in test fold
        X_test[0, 0] = np.nan

        preprocessor = LeakageSafePreprocessor(n_components=2, random_state=42)
        X_train_trans = preprocessor.fit_transform(X_train)

        # Preprocessor statistics must reflect X_train mean (~10.0), not X_test
        assert np.all(np.abs(preprocessor.imputer_.statistics_ - 10.0) < 1.0)
        assert np.all(np.abs(preprocessor.scaler_.mean_ - 10.0) < 1.0)

        # Record scaler parameters before transforming test fold
        train_mean_before = np.copy(preprocessor.scaler_.mean_)
        train_scale_before = np.copy(preprocessor.scaler_.scale_)
        pca_components_before = np.copy(preprocessor.pca_.components_)

        # Transform test fold
        X_test_trans = preprocessor.transform(X_test)

        # Preprocessor parameters MUST remain completely unchanged
        np.testing.assert_array_equal(preprocessor.scaler_.mean_, train_mean_before)
        np.testing.assert_array_equal(preprocessor.scaler_.scale_, train_scale_before)
        np.testing.assert_array_equal(preprocessor.pca_.components_, pca_components_before)

        # Transformed shapes
        assert X_train_trans.shape == (50, 2)
        assert X_test_trans.shape == (20, 2)
        # Ensure no NaNs remain in transformed outputs
        assert not np.isnan(X_train_trans).any()
        assert not np.isnan(X_test_trans).any()

    def test_transform_before_fit_raises_error(self):
        preprocessor = LeakageSafePreprocessor(n_components=2)
        X = np.random.randn(10, 4)
        with pytest.raises(RuntimeError, match="not been fitted"):
            preprocessor.transform(X)
