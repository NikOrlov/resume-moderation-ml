import numpy as np
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder


class XGBClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_estimators=100,
        eta=0.3,
        gamma=0.0,
        max_depth=6,
        min_child_weight=1.0,
        max_delta_step=0.0,
        subsample=1.0,
        colsample_bytree=1.0,
        scale_pos_weight=1.0,
        verbosity=True,
        seed=0,
        nthread=-1,
    ):
        self.n_estimators = n_estimators
        self.eta = eta
        self.gamma = gamma
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.max_delta_step = max_delta_step
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.scale_pos_weight = scale_pos_weight
        self.seed = seed
        self.nthread = nthread
        self.verbosity = verbosity

        self.classes_ = None
        self.n_classes_ = None
        self.encoder_ = None
        self.booster_ = None
        self.eval_results = None

        self.threshold = 0.5

    def get_xgb_params(self):
        xgb_params = self.get_params(deep=False)
        if self.nthread <= 0:
            xgb_params.pop("nthread", None)
        xgb_params["verbosity"] = 1 if self.verbosity else 0
        return xgb_params

    def fit(self, X, y, sample_weight=None, eval_set=None, eval_metric=None, early_stopping_rounds=None, verbose=False):
        eval_results = {}
        self.classes_ = list(np.unique(y))
        self.n_classes_ = len(self.classes_)

        xgb_options = self.get_xgb_params()
        if self.n_classes_ > 2:
            objective = "multi:softprob"
            xgb_options["num_class"] = self.n_classes_
        else:
            objective = "binary:logistic"
        xgb_options["objective"] = objective

        eval_func = eval_metric if callable(eval_metric) else None
        if eval_metric is not None:
            if callable(eval_metric):
                eval_metric = None
            xgb_options["eval_metric"] = eval_metric

        self.encoder_ = LabelEncoder().fit(y)
        labels = self.encoder_.transform(y)

        evals = []
        if eval_set is not None:
            evals = [
                (xgb.DMatrix(x[0], label=self.encoder_.transform(x[1])), "validation_{}".format(i))
                for i, x in enumerate(eval_set)
            ]

        train_dmatrix = xgb.DMatrix(X, label=labels, weight=sample_weight)
        self.booster_ = xgb.train(
            xgb_options,
            train_dmatrix,
            self.n_estimators,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            evals_result=eval_results,
            feval=eval_func,
            verbose_eval=verbose,
        )
        if eval_results:
            self.eval_results = {
                dataset: np.array(metric_result[eval_metric]) for dataset, metric_result in eval_results.items()
            }

        return self

    @property
    def booster(self):
        if self.booster_ is None:
            raise ValueError("model is not fitted")
        return self.booster_

    def predict(self, X):
        test_dmatrix = xgb.DMatrix(X)
        class_probs = self.booster.predict(test_dmatrix)
        if len(class_probs.shape) > 1:
            column_indexes = np.argmax(class_probs, axis=1)
        else:
            column_indexes = np.repeat(0, X.shape[0])
            column_indexes[class_probs > self.threshold] = 1
        return self.encoder_.inverse_transform(column_indexes)

    def predict_proba(self, X):
        test_dmatrix = xgb.DMatrix(X)
        class_probs = self.booster.predict(test_dmatrix)
        if self.n_classes_ > 2:
            return class_probs
        return np.vstack((1.0 - class_probs, class_probs)).transpose()
