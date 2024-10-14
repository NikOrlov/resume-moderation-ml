from pydantic import BaseModel


class ModerationResponse(BaseModel):
    approve: bool
    approveScore: float
    flags: list
    salary: dict


class BlockResponse(BaseModel):
    block: bool
    blockScore: float
