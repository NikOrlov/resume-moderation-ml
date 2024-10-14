import os

import pytest

from resume_moderation_ml import Application


@pytest.fixture(scope="class")
async def frontik_app() -> Application:
    app = Application(
        app="resume-moderation-ml",
        app_module="resume_moderation_ml",
        app_root=os.path.dirname(os.path.dirname(__file__)),
    )
    await app.init()
    return app
