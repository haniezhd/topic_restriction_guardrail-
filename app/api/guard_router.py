from fastapi import APIRouter
from pydantic import BaseModel

from ml.guard import llama_guard_check

router = APIRouter(prefix="/guard", tags=["LlamaGuard"])


class GuardRequest(BaseModel):
    text: str


class GuardResponse(BaseModel):
    safe: bool
    reason: str


@router.post("/check", response_model=GuardResponse)
async def check_guard(req: GuardRequest):

    safe, reason = await llama_guard_check(req.text)

    return GuardResponse(
        safe=safe,
        reason=reason
    )