import argparse
import logging

from sklearn.metrics import precision_recall_curve, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from resume_moderation_ml.model import classifier
from resume_moderation_ml.model.train.config import ModerationConfig
from resume_moderation_ml.model.train.environment import init_train_env
from resume_moderation_ml.model.train.model import create_model, get_model_parameters, get_task_subjects
from resume_moderation_ml.model.train.source import get_targets
from resume_moderation_ml.model.train.vectorize import get_resume_vectors
from hhkardinal.utils.stats import select_threshold

logger = logging.getLogger(__name__)

config = ModerationConfig()


def fit_model(task_name, resume_vectors, targets):
    logger.info('fit model for task %s', task_name)

    X, y = get_task_subjects(task_name, resume_vectors, targets)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=config.cv_seed,
                                                        test_size=1.0 / config.cv_number_of_folds)
    model = create_model(task_name).fit(X_train, y_train)

    threshold_params = get_model_parameters(task_name).get('threshold', {})

    prediction = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, prediction)
    threshold = select_threshold(precision, recall, thresholds, threshold_params)

    model.threshold = threshold

    roc_auc = roc_auc_score(y_test, prediction)
    prediction = (prediction >= threshold).astype(int)
    logger.info('roc_auc is %.5f, chose threshold %.5f got precision %.5f and recall %.5f',
                roc_auc, threshold, precision_score(y_test, prediction), recall_score(y_test, prediction))
    return model


def run_fitting(tasks):
    resume_vectors = get_resume_vectors()
    targets = get_targets()

    for task_name in tasks:
        model = fit_model(task_name, resume_vectors, targets)
        classifier.save(model, task_name)


def read_tasks():
    parser = argparse.ArgumentParser(description='fit resume moderation models')
    all_tasks = ('approve_complete', 'approve_incomplete', 'block', 'careless_key_skill_information',
                 'careless_additional_information', 'bad_function', 'bad_education')
    parser.add_argument('--task', choices=all_tasks, help='select model to fit')
    args = parser.parse_args()
    if args.task:
        return [args.task]
    else:
        return all_tasks


if __name__ == '__main__':
    init_train_env()
    tasks = read_tasks()
    run_fitting(tasks)
