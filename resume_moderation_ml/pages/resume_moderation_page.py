from frontik.handler import PageHandler, router
from tornado.web import HTTPError


class ResumeModerationPage(PageHandler):
    @router.post()
    async def post_page(self) -> None:
        try:
            resume = self.json_body
        except Exception as e:
            err_msg = f"Error while parsing data: {e}"
            self.log(err_msg)
            raise HTTPError(400, err_msg)

        try:
            decision = self.application.predictor.make_decision(resume)
        except Exception as e:
            err_msg = f"Error while prediction: {e}"
            self.log(err_msg)
            raise HTTPError(400, err_msg)
        return decision
