import numpy as np
from sklearn.model_selection import KFold

from resume_moderation_ml.model.classifier import TASKS
from resume_moderation_ml.model.train.config import resume_moderation_config
from resume_moderation_ml.model.train.logger import setup_logger
from resume_moderation_ml.model.train.model import create_model, get_task_subjects
from resume_moderation_ml.model.train.source import get_targets
from resume_moderation_ml.model.train.vectorize import get_resume_vectors

logger = setup_logger(__name__)


def cross_validate_model(task_name, X, y, cv, xgb_parameters=None):
    cv_result = []

    logger.info("Start cross validation on sample with %d objects", X.shape[0])

    for fold_num, (train, test) in enumerate(cv):
        model = create_model(task_name, xgb_parameters=xgb_parameters)
        model.fit(X[train], y[train], eval_set=[(X[test], y[test])], eval_metric="auc")
        cv_result.append(model.eval_results["validation_0"])
        logger.debug("finished fold %d", fold_num + 1)

    cv_result = np.asarray(cv_result)
    means = cv_result.mean(axis=0)
    best_idx = np.argmax(means)
    std = cv_result[:, best_idx].std()

    logger.info("got average ROC %.5f with std %.5f, ntrees=%d", means[best_idx], std, best_idx + 1)
    # TODO: plug-in average precision on different recall levels
    return {"n_estimators": best_idx + 1, "mean": means[best_idx], "std": std}


def evaluate_model(task_name, resume_vectors, targets):
    logger.info("evaluate model for task %s", task_name)

    X, y = get_task_subjects(task_name, resume_vectors, targets)
    cv = KFold(
        n_splits=resume_moderation_config.cv_number_of_folds,
        random_state=resume_moderation_config.cv_seed,
        shuffle=True,
    ).split(np.arange(X.shape[0]))
    cross_validate_model(task_name, X, y, cv)


def run_evaluation():
    resume_vectors = get_resume_vectors()
    targets = get_targets()
    for task_name in TASKS:
        evaluate_model(task_name, resume_vectors, targets)


if __name__ == "__main__":
    run_evaluation()
