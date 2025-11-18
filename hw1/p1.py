import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import make_regression
from sklearn.datasets import make_classification


def regression_dataset_with_missing_responses(
    n_samples: int, n_features: int, p_missing: float = 0.1
):
    """
    Generate a regression dataset with missing responses.

    Args:
        n_samples: Number of samples.
        n_features: Number of features.
        p_missing: Probability of missing a response.

    Returns:
        A Pandas DataFrame with features and a response column.
    """
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
    )
    # Replace response values by NaN with probability p_missing.
    y = np.where(np.random.rand(*y.shape) < p_missing, np.nan, y)
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    return pd.concat([X, pd.Series(y, name="response")], axis=1)


def classification_dataset_with_missing_features(
    n_samples: int, n_features: int, p_missing: float = 0.1, n_classes: int = 3
):
    """
    Generate a classification dataset with missing features.

    Args:
        n_samples: Number of samples.
        n_features: Number of features.
        p_missing: Probability of missing a feature.
        n_classes: Number of classes.

    Returns:
        A Pandas DataFrame with features and a label column.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_clusters_per_class=1,
    )
    # Replace feature values in X by NaN with probability p_missing.
    X = np.where(np.random.rand(*X.shape) < p_missing, np.nan, X)
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    return pd.concat([X, pd.Series(y, name="label")], axis=1)


class KnnLabelImputer(BaseEstimator, TransformerMixin):
    def __init__(self, k: int = 3):
        self.k = k

    def fit(self, X, y):
        assert y is not None
        # store all values
        self.X_train = np.asarray(X).copy()
        self.y_train = np.asarray(y).copy()
        return self

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X, y)

    def transform(self, X, y):
        """Replace missing responses with the mean of the nearest neighbors' responses.

        Args:
            X: The features.
            y: The original responses.

        Returns:
            The imputed responses.
        """
        assert y is not None
        X = np.asarray(X)
        y = np.asarray(y)
        y_imputed = y.copy()

        # loop through each sample
        for i in range(len(y)):
            if np.isnan(y[i]):  # handle missing response
                distances = []
                for j in range(len(self.y_train)):
                    # skip samples with missing responses
                    if not np.isnan(self.y_train[j]):
                        # calculate Euclidean distance
                        dist = np.sqrt(np.sum((X[i] - self.X_train[j]) ** 2))
                        if dist > 0:
                            distances.append((dist, j))

                # sort to find nearest k
                distances.sort()
                k_nearest = distances[:self.k]

                # average out the responses
                neighbor_responses = [self.y_train[idx] for _, idx in k_nearest]
                if neighbor_responses:
                    y_imputed[i] = float(np.mean(neighbor_responses))

        return y_imputed


class KnnFeatureImputer(BaseEstimator, TransformerMixin):
    def __init__(self, k: int = 3):
        self.k = k

    def fit(self, X, y):
        assert y is not None
        # Store all data for later use
        self.X_train = np.asarray(X).copy()
        self.y_train = np.asarray(y).copy()
        return self

    def fit_transform(self, X, y, **fit_params):
        return self.fit(X, y).transform(X, y)

    def transform(self, X, y):
        """Replace missing features with the mean of the nearest neighbors' features.

        Uses class-aware neighbor selection: only considers samples with the same 
        class label when finding nearest neighbors for feature imputation.

        Args:
            X: The features.
            y: The original responses.

        Returns:
            The imputed features.
        """
        assert y is not None
        X = np.asarray(X)
        y = np.asarray(y)
        X_imputed = X.copy()

        # loop through each sample
        for i in range(len(X)):
            current_class = y[i]

            # handle missing
            has_missing = np.isnan(X[i]).any()
            if not has_missing:
                continue

            # Find distances to training samples with the same class
            distances = []
            for j in range(len(self.y_train)):
                if self.y_train[j] == current_class:  # same class check
                    # calculate distance using only non-missing features
                    valid_mask = ~np.isnan(X[i]) & ~np.isnan(self.X_train[j])
                    if np.any(valid_mask):
                        diff = X[i, valid_mask] - self.X_train[j, valid_mask]
                        dist = np.sqrt(np.sum(diff ** 2))
                        if dist > 0:
                            distances.append((dist, j))

            # sort to find nearest k
            distances.sort()
            k_nearest = distances[:self.k]

            # impute with mean of neighbors for each missing feature
            missing_cols = np.where(np.isnan(X[i]))[0]
            for col_idx in missing_cols:
                neighbor_values = []
                for _, neighbor_idx in k_nearest:
                    neighbor_value = self.X_train[neighbor_idx, col_idx]
                    if not np.isnan(neighbor_value):
                        neighbor_values.append(neighbor_value)

                if neighbor_values:
                    X_imputed[i, col_idx] = float(np.mean(neighbor_values))

        return X_imputed


def knn_impute_missing_responses(X, y, k=3):
    """
    Impute missing labels y using k-NN over features X.
    Returns: pandas Series y_imputed
    """
    imputer = KnnLabelImputer(k=k).fit(X, y)
    return imputer.transform(X, y)


def knn_impute_missing_features(X, y, k=3):
    """
    Impute missing feature values in X using class-aware k-NN within the same label.
    Returns: pandas DataFrame X_imputed
    """
    imputer = KnnFeatureImputer(k=k).fit(X, y)
    return imputer.transform(X, y)
