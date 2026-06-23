# api/main.py

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import httpx

from ml.policy import predict, explain, classify_with_score

app = FastAPI(title="Guardrail System")

# Request Schemas
class TextRequest(BaseModel):
    text: str


class BatchRequest(BaseModel):
    data: List[dict]

class GuardRequest(BaseModel):
    text: str


class GuardResponse(BaseModel):
    safe: bool
    reason: str

@app.get("/health")
def health():
    return {"status": "ok"}
    
# Centroid endpoints
@app.post("/predict")
def endpoint_predict(req: TextRequest):
    return predict(req.text)


@app.post("/score")
def endpoint_score(req: TextRequest):
    return explain(req.text)


@app.post("/batch")
def endpoint_batch(req: BatchRequest):

    results = []

    for item in req.data:
        text = item["text"]
        results.append(classify_with_score(text))

    return {
        "results": results
    }

# Llama Guard endpoint

import httpx
from fastapi import HTTPException

OPENROUTER_API_KEY = "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"
LLAMA_GUARD_URL = "https://openrouter.ai/api/v1/chat/completions"
LLAMA_GUARD_MODEL = "meta-llama/llama-guard-4-12b"
TIMEOUT = 30.0

@app.post("/guard", response_model=GuardResponse)
async def guard(req: GuardRequest):

    payload = {
        "model": LLAMA_GUARD_MODEL,
        "messages": [
            {"role": "user", "content": req.text}
        ],
        "max_tokens": 50
    }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.post(
                LLAMA_GUARD_URL,
                headers=headers,
                json=payload
            )

        # 1. HTTP-level failure
        if resp.status_code != 200:
            raise HTTPException(
                status_code=502,
                detail=f"OpenRouter HTTP error {resp.status_code}: {resp.text}"
            )

        data = resp.json()

        # 2. API-level error (OpenRouter format)
        if "error" in data:
            raise HTTPException(
                status_code=502,
                detail=f"OpenRouter error: {data['error']}"
            )

        # 3. Schema validation
        if "choices" not in data:
            raise HTTPException(
                status_code=502,
                detail=f"Invalid response format: {data}"
            )

        output = data["choices"][0]["message"]["content"].strip()

        # 4. Safer classification logic
        output_lower = output.lower()

        safe = output_lower.startswith("safe") and "unsafe" not in output_lower

        return GuardResponse(
            safe=safe,
            reason=output
        )

    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="OpenRouter request timed out"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
