import logging
from collections import defaultdict
from copy import copy

import numpy as np
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from resume_moderation_ml.model.resume import SalaryDropFeaturesExtractor
from resume_moderation_ml.model.train import cache_obj
from resume_moderation_ml.model.train.config import resume_moderation_config
from resume_moderation_ml.model.train.logger import setup_logger
from resume_moderation_ml.model.train.source import get_source_csv_lines_from_hive, iterate_raw_source_csv
from resume_moderation_ml.model.train.utils import cache
from resume_moderation_ml.model.train.utils.stats import RunningStatistic, select_threshold
from resume_moderation_ml.model.train.utils.transformers import ValueExtractor
from resume_moderation_ml.model.train.xgb import XGBClassifier

logger = setup_logger(__name__)
logger.setLevel(level=logging.DEBUG)
_DATA_KEY = "dropsalary/data"
_RESUME_VECTORS_KEY = "dropsalary/resume_vectors"
_VECTORIZER_KEY = "dropsalary/vectorizer"
_CLASSIFIER_KEY = "dropsalary/classifier"


def load_currency_rates() -> dict:
    # data = requests.get('https://api.hh.ru/dictionaries').json()
    # return {currency['code']: currency['rate'] for currency in data['currency']}
    return {
        "AZN": 0.017696,
        "BYR": 0.034052,
        "EUR": 0.009496,
        "GEL": 0.028475,
        "KGS": 0.881694,
        "KZT": 5.015775,
        "RUR": 1.0,
        "UAH": 0.42873,
        "USD": 0.01041,
        "UZS": 132.983144,
    }


def _resume_salary_key(resume):
    salary = resume["salary"]
    return salary["amount"], salary["currency"]


@cache.cache(_DATA_KEY, cache_cls=cache_obj)
def get_raw_data():
    resumes = []
    drop_rates = defaultdict(RunningStatistic)

    for id_, status, source_version, moderated_version, complete_status, creation_time in iterate_raw_source_csv(
        get_source_csv_lines_from_hive["all"]()
    ):
        if status != "approved" or source_version["salary"] is None:
            continue

        resume_key = _resume_salary_key(source_version)
        if resume_key not in drop_rates:
            resumes.append({"salary": source_version["salary"]})
        drop_rates[resume_key].feed(int(moderated_version.get("salary") is None), 1)

    targets = np.array([int(drop_rates[_resume_salary_key(r)].value >= 0.5) for r in resumes])
    weights = np.array([drop_rates[_resume_salary_key(r)].seen for r in resumes])

    logger.info("created sample of %d object for drop salary task, rate is %.5f", len(resumes), targets.mean())
    return {"resumes": resumes, "targets": targets, "weights": weights}


def get_raw_resumes():
    return get_raw_data()["resumes"]


def get_targets_and_weights():
    raw_data = get_raw_data()
    return raw_data["targets"], raw_data["weights"]


@cache.cache(_VECTORIZER_KEY, cache_cls=cache_obj)
def get_vectorizer():
    return ValueExtractor(SalaryDropFeaturesExtractor(load_currency_rates()))


@cache.cache(_RESUME_VECTORS_KEY, cache_cls=cache_obj)
def get_resume_vectors():
    vectorizer = get_vectorizer()
    resumes = get_raw_resumes()
    logger.info("transform %d source resumes to a vector representation", len(resumes))
    resume_vectors = vectorizer.transform(resumes)
    logger.info("shape of transformed data is %s", repr(resume_vectors.shape))
    return resume_vectors


@cache.cache(_CLASSIFIER_KEY, cache_cls=cache_obj)
def fit_model():
    logger.info("fit drop salary model")
    X = get_resume_vectors()
    y, w = get_targets_and_weights()
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X,
        y,
        w,
        test_size=resume_moderation_config.drop_salary_params["validation_size"],
        random_state=resume_moderation_config.cv_seed,
    )

    xgb_params = copy(resume_moderation_config.common_xgb_params)
    xgb_params["nthread"] = resume_moderation_config.ncores
    xgb_params.update(resume_moderation_config.drop_salary_params["xgb_params"])

    model = XGBClassifier(**xgb_params)
    model.fit(X_train, y_train, sample_weight=w_train)
    prediction = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, prediction, sample_weight=w_test)

    precision, recall, thresholds = precision_recall_curve(y_test, prediction, sample_weight=w_test)
    threshold = select_threshold(
        precision, recall, thresholds, resume_moderation_config.drop_salary_params["threshold"]
    )
    model.threshold = threshold
    prediction = (prediction >= model.threshold).astype(int)

    logger.info(
        "roc_auc is %.5f, chose threshold %.5f got precision %.5f and recall %.5f",
        roc_auc,
        threshold,
        precision_score(y_test, prediction, sample_weight=w_test),
        recall_score(y_test, prediction, sample_weight=w_test),
    )
    return model


if __name__ == "__main__":
    fit_model()
