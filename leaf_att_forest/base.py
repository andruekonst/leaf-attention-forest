import logging
import numpy as np
from typing import NamedTuple, Optional, Tuple
from .forests import *
from scipy.special import softmax as _softmax
import cvxpy as cp
from .leaf_data import _prepare_leaf_data_fast
from .utils import _get_node_depths, _convert_labels_to_probas
from abc import ABC, abstractmethod


class BaseAttentionForest(ABC):
    """Base Attention Forest.
    Preprocesses inputs and trains an underlying forest.
    """
    def __init__(self, params):
        self.params = params
        self.forest = None
        self._after_init()

    def _after_init(self):
        self.onehot_encoder = None

    def _preprocess_target(self, y):
        """Preprocess target.
        Converts classification labels to probabilities.

        Args:
            y: Input labels.

        Returns:
            Encoded labels (probabilities).

        """
        if self.params.task == TaskType.CLASSIFICATION:
            y, self.onehot_encoder = _convert_labels_to_probas(y, self.onehot_encoder)
        return y

    def fit(self, X, y) -> 'BaseAttentionForest':
        """Fit an underlying forest and obtain leaf data.

        Args:
            X: Input vectors.
            y: Input targets.

        Returns:
            Self.

        """
        forest_cls = FORESTS[ForestType(self.params.kind, self.params.task)]
        self.forest = forest_cls(**self.params.forest)
        self.forest.fit(X, y)
        # store training X and y
        self.training_xs = X.copy()
        self.training_y = self._preprocess_target(y.copy())
        # store leaf id for each point in X
        self.training_leaf_ids = self.forest.apply(self.training_xs)
        # collect leaf data
        if hasattr(self.forest, 'get_leaf_data'):
            self.leaf_data_x, self.leaf_data_y = self.forest.get_leaf_data()
        else:
            self.leaf_data_x, self.leaf_data_y = _prepare_leaf_data_fast(
                self.training_xs,
                self.training_y,
                self.training_leaf_ids,
                self.forest.estimators_,
            )
        self.tree_weights = np.ones(self.forest.n_estimators)
        self.static_weights = np.ones(self.forest.n_estimators) / self.forest.n_estimators
        return self

    @abstractmethod
    def optimize_weights(self, X, y_orig) -> 'BaseAttentionForest':
        ...

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        ...

    def predict_original(self, X):
        """Predict with the underlying forest.

        Args:
            X: Input vectors.

        Returns:
            Predictions.

        """
        if self.params.task == TaskType.REGRESSION:
            return self.forest.predict(X)
        elif self.params.task == TaskType.CLASSIFICATION:
            return self.forest.predict_proba(X)
        raise ValueError(f'Unsupported task type in predict_original: "{self.params.task}"')

