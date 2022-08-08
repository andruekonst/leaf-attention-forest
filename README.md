# Gamma-Leaf Forest Attention

## Installation

For the package installation, first install all the requirements and then install the **leaf_att_forest** package.
```
$ pip install -r requirements.txt
$ python setup.py install
```

## Usage

The model interface is *scikit-learn* like, except
it is extended with `optimize_weights` method which can be executed with the same training data as used for an underlying forest training (see [example](notebooks/simple_regression.ipynb)), or with a new data set (see [example](notebooks/simple_warm_start.ipynb)).

Code example for model instantiation:
```
from leaf_att_forest import (
    GLAFParams,
    GammaLeafAttentionForest,
    ForestKind,
    TaskType,
)

model = GammaLeafAttentionForest(
    GLAFParams(
        kind=ForestKind.EXTRA,
        task=TaskType.REGRESSION,
        # Gamma-Leaf Attention Forest Parameters
        leaf_tau=1.0,
        leaf_attention=True,
        n_tau=5,
        fit_tree_weights=True,
        # Base forest parameters
        forest=dict(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=5,
            random_state=12345,
        ),
    )
)
```

After the underlying forest should be trained:
```
model.fit(X_train, y_train)
```

And then weights are optimized:
```
model.optimize_weights(X_train, y_train)
```

In order to estimate weights optimization impact scores for `model.predict_original(X_val)` and `model.predict(X_val)` could be compared.
