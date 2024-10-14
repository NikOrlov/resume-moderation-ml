import pickle

import joblib
from mlback.server.app import MLBackApplication

from resume_moderation_ml.model.predictor import ResumeModerationPredictor
from resume_moderation_ml.model.train.utils import alpha_quotient, caps_quotient
from resume_moderation_ml.pages.resume_moderation_block_page import ResumeModerationBlockPage
from resume_moderation_ml.pages.resume_moderation_page import ResumeModerationPage

DIR_PATH = "resume_moderation_ml/model/classifier"


def load_model(name, dir_name=DIR_PATH):
    with open(f"{dir_name}/{name}.pickle", "rb") as file:
        return pickle.load(file)


class Application(MLBackApplication):
    def init_models(self) -> None:
        models = ["approve_incomplete", "approve_complete", "block"]
        flag_models = [
            "careless_key_skill_information",
            "careless_additional_information",
            "bad_function",
            "bad_education",
        ]
        self.models = {}
        self.flag_models = {}

        for model_name in models:
            self.models[model_name] = load_model(model_name)
        for model_name in flag_models:
            self.flag_models[model_name] = load_model(model_name)

        self.dropsalary = {
            "classifier": load_model("dropsalary/classifier"),
            "vectorizer": load_model("dropsalary/vectorizer"),
        }
        self.vectorizer = joblib.load("resume_moderation_ml/model/classifier/vectorizer.pickle", mmap_mode="r")

        predictor = ResumeModerationPredictor(
            vectorizer=self.vectorizer,
            models=self.models,
            flag_models=self.flag_models,
            dropsalary_classifier=self.dropsalary["classifier"],
            dropsalary_vectorizer=self.dropsalary["vectorizer"],
        )

        self.predictor = predictor

    def application_urls(self) -> list[tuple]:
        return [
            ("/moderation/resume/block", ResumeModerationBlockPage),
            ("/moderation/resume", ResumeModerationPage),
        ]
