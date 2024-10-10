import copy
import logging
import os

import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, tpe
from sklearn.model_selection import KFold

from resume_moderation_ml.model.classifier import TASKS
from resume_moderation_ml.model.train.config import ModerationConfig
from resume_moderation_ml.model.train.environment import init_train_env
from resume_moderation_ml.model.train.evaluate import cross_validate_model
from resume_moderation_ml.model.train.model import get_task_subjects
from resume_moderation_ml.model.train.source import get_targets
from resume_moderation_ml.model.train.vectorize import get_resume_vectors
from resume_moderation_ml.model.train import cache_obj

logger = logging.getLogger(__name__)

config = ModerationConfig()


def serialize_params(xgb_params):
    params = copy.copy(xgb_params)
    params.pop('nthread', None)
    params.pop('n_estimators', None)

    serialized = []
    for param_name in sorted(params):
        value = params[param_name]
        format_str = '{}:{}' if isinstance(value, int) else '{}:{:.4f}'
        serialized.append(format_str.format(param_name, value))
    return '; '.join(serialized)


def optimize_xgb_params(task_name, X, y, param_space, max_evals=10, n_folds=5, random_state=None):
    trials = Trials()

    def _hp_wrapper(xgb_params):
        logger.info('trying xgb params: [%s]', serialize_params(xgb_params))

        cv = KFold(n_splits=n_folds, random_state=random_state).split(np.arange(X.shape[0]))

        cv_result = cross_validate_model(task_name, X, y, cv, xgb_params)
        return {
            'loss': 1.0 - cv_result['mean'],
            'n_estimators': cv_result['n_estimators'],
            'status': STATUS_OK,
            'mean': cv_result['mean'],
            'std': cv_result['std']
        }

    best_params = fmin(_hp_wrapper, param_space, algo=tpe.suggest, trials=trials, max_evals=max_evals)
    return trials, best_params


def run_params_optimization():
    optimization_config = {
        'param_space': config.hp_param_space,
        'max_evals': config.hp_max_evals,
        'n_folds': config.cv_number_of_folds,
        'random_state': config.cv_seed
    }

    resume_vectors = get_resume_vectors()
    targets = get_targets()

    for task_name in TASKS:
        logger.info('start optimizing hyperparameters for task %s', task_name)
        X, y = get_task_subjects(task_name, resume_vectors, targets)
        trials, best_params = optimize_xgb_params(task_name, X, y, **optimization_config)

        output_path = os.path.join('moderation/resume/hp', task_name)
        cache_obj.save(trials, os.path.join(output_path, 'trials'))
        cache_obj.save(best_params, os.path.join(output_path, 'best'))


if __name__ == '__main__':
    # init_train_env()
    run_params_optimization()
