import pickle

from mlback.server.app import MLBackApplication

from resume_moderation_ml.model.predictor import ResumeModerationPredictor
from resume_moderation_ml.pages.resume_moderation_block_page import ResumeModerationBlockPage
from resume_moderation_ml.pages.resume_moderation_page import ResumeModerationPage


class Application(MLBackApplication):
    def init_models(self) -> None:
        self.models = {}
        for artifact in self.models_config:
            with open(artifact.path, "rb") as file:
                self.models[artifact.name] = pickle.load(file)

        predictor = ResumeModerationPredictor(**self.models)
        self.predictor = {"predictor": predictor}

    def application_urls(self) -> list[tuple]:
        return [
            ("/moderation/resume", ResumeModerationPage),
            ("/moderation/resume/block", ResumeModerationBlockPage)
        ]
