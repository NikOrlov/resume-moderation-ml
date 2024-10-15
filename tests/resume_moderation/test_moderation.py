from frontik.testing import FrontikTestBase

from resume_moderation_ml import Application
from resume_moderation_ml.pages import BlockResponse, ModerationResponse
from tests.resume_moderation.utils import build_moderation_request


class Test(FrontikTestBase):
    async def test_moderation(self, frontik_app: Application) -> None:
        response = await self.fetch_json(**build_moderation_request())
        response_model = ModerationResponse(**response)
        assert response_model.approve
        assert response_model.approveScore <= 1

    async def test_moderation_block(self, frontik_app: Application) -> None:
        response = await self.fetch_json(**build_moderation_request(path="/moderation/resume/block"))
        response_model = BlockResponse(**response)

        assert not response_model.block
        assert response_model.blockScore <= 1
