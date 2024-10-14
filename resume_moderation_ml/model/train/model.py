import copy

import numpy as np

from resume_moderation_ml.model.classifier import TARGET_FLAGS
from resume_moderation_ml.model.manual_rules import ManualRejectEstimator
from resume_moderation_ml.model.train.config import resume_moderation_config
from resume_moderation_ml.model.train.xgb import XGBClassifier


def get_model_parameters(task_name):
    parameters = copy.copy(resume_moderation_config.common_xgb_params)
    parameters["nthread"] = resume_moderation_config.ncores
    parameters.update(resume_moderation_config.model_params[task_name])
    return parameters


def get_xgb_parameters(task_name):
    parameters = get_model_parameters(task_name)
    parameters.pop("threshold", None)
    return parameters


def get_task_subjects(task_name, resume_vectors, targets):
    if task_name == "approve_complete":
        completed_idx = np.where(targets.incompleted == 0)[0]
        return resume_vectors[completed_idx], targets.approve_target[completed_idx]
    if task_name == "approve_incomplete":
        return resume_vectors, targets.approve_target
    if task_name == "block":
        return resume_vectors, 1 - targets.approve_target

    if task_name not in TARGET_FLAGS:
        raise ValueError("unknown task name {}".format(task_name))

    approved_idx = np.where(targets.approve_target == 1)[0]
    return resume_vectors[approved_idx], targets.get_flag_target(task_name)[approved_idx]


def create_model(task_name, manual_feature_number=0, xgb_parameters=None):
    if xgb_parameters is None:
        xgb_parameters = get_xgb_parameters(task_name)
    xgb_model = XGBClassifier(**xgb_parameters)
    if task_name not in ("approve_complete", "approve_incomplete"):
        return xgb_model

    return ManualRejectEstimator(manual_feature_number, xgb_model)
