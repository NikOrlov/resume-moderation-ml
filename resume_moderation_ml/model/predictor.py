from sklearn.pipeline import Pipeline
from typing import Any, Mapping
import xgboost as xgb


TARGET_FLAGS = ['careless_key_skill_information', 'careless_additional_information',
                'bad_function', 'bad_education']


def xgb_set_single_thread(xgb_booster: xgb.Booster) -> None:
    xgb_booster.set_param('nthread', 1)


def is_mapping(variable: Any) -> bool:
    return isinstance(variable, Mapping)


class ResumeFormatException(Exception):
    pass


class ResumeModerationPredictor:

    def __init__(self, vectorizer,
                 approve_incomplete,
                 approve_complete,
                 block,
                 dropsalary_classifier,
                 dropsalary_vectorizer):
        self.vectorizer = vectorizer
        self.approve_incomplete_model = approve_incomplete
        xgb_set_single_thread(self.approve_incomplete_model.delegate.booster_)

        self.approve_complete_model = approve_complete
        xgb_set_single_thread(self.approve_complete_model.delegate.booster_)

        self.block_model = storage.load('block') if 'block' in storage else None

        if self.block_model is not None:
            xgb_set_single_thread(self.block_model.booster_)

        self.flags_models = [(flag, storage.load(flag)) for flag in TARGET_FLAGS]
        for flag, model in self.flags_models:
            xgb_set_single_thread(model.booster_)

        xgb_set_single_thread(dropsalary_classifier.booster_)
        self.drop_salary = Pipeline([
            ('vectorizer', dropsalary_vectorizer),
            ('classifier', dropsalary_classifier)])

    def make_decision(self, resume):
        if not is_mapping(resume) or 'validationSchema' not in resume:
            raise ResumeFormatException()

        try:
            vector = self.vectorizer.transform([resume])
        except KeyError as e:
            raise ResumeFormatException(e.message)

        if resume['validationSchema'] == 'incomplete':
            approve_score = self.approve_incomplete_model.predict_proba(vector)[0, 1]
            approve = bool(approve_score > self.approve_incomplete_model.threshold)
        else:
            approve_score = self.approve_complete_model.predict_proba(vector)[0, 1]
            approve = bool(approve_score > self.approve_complete_model.threshold)

        flags = []
        salary = resume.get('salary')
        if approve:
            flags.extend(flag for flag, model in self.flags_models if model.predict(vector)[0] == 1.0)
            if salary is not None and self.drop_salary.predict([resume])[0] == 1.0:
                salary = None

        return {
            'approve': approve,
            'approveScore': approve_score,
            'flags': flags,
            'salary': salary,
        }

    def make_block_decision(self, resume):
        if not self.block_model:
            raise ValueError('no block model in git-LFS storage')
        if not is_mapping(resume):
            raise ResumeFormatException()

        try:
            vector = self.vectorizer.transform([resume])
        except KeyError as e:
            raise ResumeFormatException(e.message)
        model = self.block_model
        score = float(model.predict_proba(vector)[0, 1])
        block = bool(score > model.threshold)
        decision = {'block': block, 'blockScore': score}
        return decision

