"""Leakage-safe split and train-fold-only preprocessing pipeline."""

from typing import Optional, Tuple
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Perform a leakage-safe stratified train/test split.

    Guarantees class distribution balance across train and test folds.
    """
    if len(X) != len(y):
        raise ValueError(f"Length mismatch: X has {len(X)} rows, y has {len(y)} labels")

    # If y is single-class or too small for stratified splitting, fallback to standard split
    unique_classes, counts = np.unique(y, return_counts=True)
    if len(unique_classes) < 2 or np.min(counts) < 2:
        return train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=True
        )

    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
        shuffle=True,
    )


class LeakageSafePreprocessor(BaseEstimator, TransformerMixin):
    """Preprocessing pipeline consisting of Imputation, Scaling, and optional Dimensionality Reduction.

    Strictly fitted on train fold only.
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        impute_strategy: str = "mean",
        random_state: int = 42,
    ):
        self.n_components = n_components
        self.impute_strategy = impute_strategy
        self.random_state = random_state

        self.imputer_: Optional[SimpleImputer] = None
        self.scaler_: Optional[StandardScaler] = None
        self.pca_: Optional[PCA] = None
        self.is_fitted_: bool = False
        self.n_features_in_: int = 0
        self.n_features_out_: int = 0

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "LeakageSafePreprocessor":
        """Fit imputation, scaling, and PCA strictly on train data."""
        X_arr = np.asarray(X, dtype=np.float64)
        self.n_features_in_ = X_arr.shape[1]

        # 1. Fit Imputer
        self.imputer_ = SimpleImputer(strategy=self.impute_strategy)
        X_imp = self.imputer_.fit_transform(X_arr)

        # 2. Fit Scaler
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_imp)

        # 3. Fit PCA if requested and input features exceed components
        if self.n_components is not None and self.n_components > 0:
            target_comp = min(self.n_components, X_arr.shape[0], X_arr.shape[1])
            self.pca_ = PCA(n_components=target_comp, random_state=self.random_state)
            self.pca_.fit(X_scaled)
            self.n_features_out_ = target_comp
        else:
            self.pca_ = None
            self.n_features_out_ = self.n_features_in_

        self.is_fitted_ = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using the parameters learned during train-fold fitting."""
        if not self.is_fitted_:
            raise RuntimeError("Preprocessor has not been fitted yet. Call fit() first.")

        X_arr = np.asarray(X, dtype=np.float64)
        X_imp = self.imputer_.transform(X_arr)
        X_scaled = self.scaler_.transform(X_imp)

        if self.pca_ is not None:
            return self.pca_.transform(X_scaled)
        return X_scaled

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit strictly on X and return transformed array."""
        return self.fit(X, y).transform(X)


def create_and_fit_preprocessor(
    X_train: np.ndarray,
    n_components: Optional[int] = 6,
) -> Tuple[LeakageSafePreprocessor, np.ndarray]:
    """Fit preprocessor strictly on X_train and return fitted preprocessor and transformed X_train."""
    preprocessor = LeakageSafePreprocessor(n_components=n_components)
    X_train_trans = preprocessor.fit_transform(X_train)
    return preprocessor, X_train_trans
