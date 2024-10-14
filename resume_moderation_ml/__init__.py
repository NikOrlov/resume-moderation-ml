import pickle
from mlback.server.app import MLBackApplication

from resume_moderation_ml.model.predictor import ResumeModerationPredictor
from resume_moderation_ml.model.train.utils import alpha_quotient, caps_quotient
from resume_moderation_ml.pages.resume_moderation_block_page import ResumeModerationBlockPage
from resume_moderation_ml.pages.resume_moderation_page import ResumeModerationPage

DIR_PATH = "resume_moderation_ml/model/classifier"


def load_model(name, dir_name=DIR_PATH):
    with open(f"{dir_name}/{name}.pickle", "rb") as file:
        return pickle.load(file)


def load_artifact(artifact):
    with open(artifact.path, 'rb') as file:
        return pickle.load(file)


class Application(MLBackApplication):
    def init_models(self) -> None:
        # artifact = self.models_config["vod"][name]
        # with open(artifact.path, "rb") as file:
        #     self.models["vod"][name] = pickle.load(file)

        models = ["approve_incomplete", "approve_complete", "block"]
        flag_models = [
            "careless_key_skill_information",
            "careless_additional_information",
            "bad_function",
            "bad_education",
        ]
        self.models = {}
        self.flag_models = {}
        self.dropsalary = {}
        for model in models:
            artifact = self.models_config["models"][model]
            self.models[model] = load_artifact(artifact)
        for model in flag_models:
            artifact = self.models_config["flags"][model]
            self.flag_models[model] = load_artifact(artifact)
        for model in ['classifier', 'vectorizer']:
            artifact = self.models_config["dropsalary"][model]
            self.dropsalary[model] = load_artifact(artifact)

        self.dropsalary = {
            "classifier": load_model("dropsalary/classifier"),
            "vectorizer": load_model("dropsalary/vectorizer"),
        }
        vectorizer_artifact = self.models_config['vectorizer']
        self.vectorizer = load_artifact(vectorizer_artifact)

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
