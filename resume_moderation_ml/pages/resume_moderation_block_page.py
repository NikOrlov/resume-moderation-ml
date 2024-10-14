import json

from frontik.handler import PageHandler, router
from tornado.web import HTTPError


class ResumeModerationBlockPage(PageHandler):
    @router.post()
    async def post_page(self) -> None:
        try:
            resume = self.json_body
        except Exception as e:
            err_msg = f"Error while parsing data: {e}"
            self.log.error(err_msg)
            raise HTTPError(400, err_msg)

        try:
            decision = self.application.predictor.make_block_decision(resume)
        except Exception as e:
            err_msg = f"Error while predicting: {e}"
            self.log.error(err_msg)
            raise HTTPError(400, err_msg)
        self.set_header("Content-Type", "application/json")
        self.text = json.dumps(decision)
