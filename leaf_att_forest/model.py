import logging
import numpy as np
from typing import NamedTuple, Optional, Tuple
from .forests import *
from scipy.special import softmax as _softmax
import cvxpy as cp
from .leaf_data import _prepare_leaf_data_fast
from .utils import _get_node_depths, _convert_labels_to_probas
from .base import BaseAttentionForest
from .leaf_data import _prepare_separate_leaf_data_fast


class GLAFParams(NamedTuple):
    """Parameters of Gamma-Leaf Attention Forest.
    """
    kind: ForestKind
    task: TaskType
    loss_ord: int = 2
    tau: float = 1.0
    # GLAF-specific Parameters
    leaf_tau: float = 1.0
    leaf_attention: bool = True
    n_tau: int = 5
    fit_tree_weights: bool = False
    # End of GLAF-specific parameters
    forest: dict = {}


class GammaLeafAttentionForest(BaseAttentionForest):
    """Gamma-Leaf Attention Forest.
    """
    def __init__(self, params: GLAFParams):
        self.params = params
        self.forest = None
        self.w = None
        self.gammas = None
        self._after_init()

    def _leaf_attention_base_fit(self, X, y) -> 'GammaLeafAttentionForest':
        forest_cls = FORESTS[ForestType(self.params.kind, self.params.task)]
        self.forest = forest_cls(**self.params.forest)
        self.forest.fit(X, y)
        # store training X and y
        self.training_xs = X.copy()
        self.training_y = self._preprocess_target(y.copy())
        # store leaf id for each point in X
        self.training_leaf_ids = self.forest.apply(self.training_xs)
        # make a tree-leaf-points correspondence
        
        self.leaf_data_xy = _prepare_separate_leaf_data_fast(
            self.training_xs,
            self.training_y,
            self.training_leaf_ids,
            self.forest.estimators_
        )
        self.tree_weights = np.ones(self.forest.n_estimators)
        return self

    def _get_n_taus(self):
        return self.params.n_tau
    
    def fit(self, x, y):
        # super().fit(x, y)
        self._leaf_attention_base_fit(x, y)
        self.w = np.ones(self.forest.n_estimators) / self.forest.n_estimators
        self.gammas = np.zeros(self._get_n_taus() + 1)
        self.gammas[-1] = 1.0  # by default none of softmax terms are used

    def _make_tau_lists(self):
        tau_list = np.logspace(-self.params.n_tau // 2, self.params.n_tau // 2, self.params.n_tau)
        return (tau_list,)

    def _get_dynamic_weights_sigma_y(self, X) -> Tuple[np.ndarray, np.ndarray]:
        tau_list, *_ = self._make_tau_lists()
        leaf_ids = self.forest.apply(X)
        all_dynamic_weights = []
        all_y = []
        all_x = []
        for cur_x, cur_leaf_ids in zip(X, leaf_ids):
            tree_dynamic_weights = []
            tree_dynamic_y = []
            tree_dynamic_x = []
            for cur_tree_id, cur_leaf_id in enumerate(cur_leaf_ids):
                leaf_xs, leaf_y = self.leaf_data_xy[(cur_tree_id, cur_leaf_id)]
                if self.params.leaf_attention:
                    leaf_att_weights = _softmax(
                        -0.5 * np.linalg.norm(leaf_xs - cur_x[np.newaxis], ord=2, axis=1) ** 2.0 * self.params.leaf_tau
                    )
                else:  # not self.params.leaf_attention
                    n_points_in_leaf = leaf_xs.shape[0]
                    # use uniform weights in case if leaf attention is not enabled
                    leaf_att_weights = np.ones((n_points_in_leaf,), dtype=np.float32) / n_points_in_leaf
                leaf_att_x = np.einsum('i,ij->j', leaf_att_weights, leaf_xs)
                if leaf_y.ndim == 2:
                    leaf_att_y = np.einsum('i,ij->j', leaf_att_weights, leaf_y)
                elif leaf_y.ndim == 1:
                    leaf_att_y = np.dot(leaf_att_weights, leaf_y).reshape((1,))
                else:
                    raise ValueError(f'Wrong number of leaf y dimensions: {leaf_y.ndim}')

                tree_dynamic_weight = cur_x - leaf_att_x
                tree_dynamic_weights.append(tree_dynamic_weight)
                tree_dynamic_y.append(leaf_att_y)
                tree_dynamic_x.append(leaf_att_x)
            x_diffs = np.stack(tree_dynamic_weights, axis=0)
            tree_dynamic_weights = [
                _softmax(-0.5 * (np.linalg.norm(x_diffs, ord=2, axis=1)) * tau)
                for tau in tau_list
            ]
            tree_dynamic_y = np.array(tree_dynamic_y)
            tree_dynamic_x = np.array(tree_dynamic_x)
            all_dynamic_weights.append(tree_dynamic_weights)
            all_x.append(tree_dynamic_x)
            all_y.append(tree_dynamic_y)
        all_dynamic_weights = np.array(all_dynamic_weights)
        all_x = np.array(all_x)
        all_y = np.array(all_y)
        dynamic_sigma_y = np.einsum('ist,ito->iso', all_dynamic_weights, all_y)
        return all_dynamic_weights, all_x, all_y, dynamic_sigma_y
    
    def optimize_weights(self, X, y_orig) -> 'GammaLeafAttentionForest':
        """Estimate optimal weights values based on the given data set.

        Args:
            X: Input vectors.
            y_orig: Input targets.

        Returns:
            Self.

        """
        assert self.forest is not None, "Need to fit before weights optimization"
        _dynamic_weights, dynamic_x, dynamic_y, dynamic_sigma_y = self._get_dynamic_weights_sigma_y(X)
        n_taus = self._get_n_taus()
        n_gammas = n_taus + 1
        gammas = cp.Variable((1, n_gammas))
        n_trees = self.forest.n_estimators
        if self.params.fit_tree_weights:
            w = cp.Variable((1, n_trees))  # static_weights
        else:
            w = cp.Parameter(
                shape=(1, n_trees),
                name='w',
                value=np.ones((1, n_trees)) / n_trees
            )
        
        y = y_orig.copy()

        if dynamic_y.shape[2] == 1:
            dynamic_y = dynamic_y[..., 0]
            dynamic_sigma_y = dynamic_sigma_y[..., 0]
        else:
            raise NotImplementedError()
        
        if self.params.fit_tree_weights:
            # when using trainable tree weights, their sum == gammas[-1]
            multiplier = 1.0
        else:
            # when weights are not trainable, they should be multiplied by gammas[-1]
            multiplier = gammas[0, -1]

        statically_weighted_target = multiplier * cp.sum(cp.multiply(w, dynamic_y), axis=1)
        sigma_weighted_target = cp.sum(cp.multiply(gammas[:, :-1], dynamic_sigma_y), axis=1)
        target_approx = statically_weighted_target + sigma_weighted_target

        loss_terms = target_approx - y
        if self.params.loss_ord == 1:
            min_obj = cp.sum(cp.abs(loss_terms))
        elif self.params.loss_ord == 2:
            min_obj = cp.sum_squares(loss_terms)
        else:
            raise ValueError(f'Wrong loss order: {self.params.loss_ord}')

        constraints = [
            gammas >= 0,
            cp.sum(gammas, axis=1) == 1,
        ]
        if self.params.fit_tree_weights:
            constraints.extend([
                w >= 0,
                # cp.sum(w, axis=1) == 1 - cp.sum(gammas[:, :-1], axis=1),
                cp.sum(w, axis=1) == gammas[:, -1],
            ])
        problem = cp.Problem(cp.Minimize(min_obj), constraints)

        loss_value = self._solve_opt_problem(problem, gammas)

        if gammas.value is None:
            logging.warn(f"Weights optimization error. Using default values.")
        else:
            self.w = w.value.copy().reshape((-1,))
            self.gammas = gammas.value.copy().reshape((-1,))
        return self

    def _solve_opt_problem(self, problem: cp.Problem, gammas: cp.Variable):
        try:
            loss_value = problem.solve()
        except Exception as ex:
            logging.warning(f"Solver error: {ex}")

        if gammas.value is None:
            logging.info(f"Can't solve problem with OSQP. Trying another solver...")
            loss_value = problem.solve(solver=cp.SCS)
        return loss_value


    def predict(self, X) -> np.ndarray:
        """Predict using the optimized weights.

        Args:
            X: Input vectors.

        Returns:
            Predictions.

        """
        assert self.forest is not None, "Need to fit before predict"
        _all_dynamic_weights, all_x, all_y, dynamic_sigma_y = self._get_dynamic_weights_sigma_y(X)

        if self.params.fit_tree_weights:
            # when using trainable tree weights, their sum == gammas[-1]
            multiplier = 1.0
        else:
            # when weights are not trainable, they should be multiplied by gammas[-1]
            multiplier = self.gammas[-1]


        statically_weighted_target = multiplier * np.dot(self.w[np.newaxis], all_y)
        sigma_weighted_target = np.dot(self.gammas[np.newaxis, :-1], dynamic_sigma_y)
        target_approx = statically_weighted_target + sigma_weighted_target
        predictions = target_approx[0]
        return predictions
