from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str | list[str] = Field(..., description="분석할 텍스트 (단일 또는 배치)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"text": "This movie was absolutely fantastic!"},
                {"text": ["I love it", "I hate it"]},
            ]
        }
    }


class PredictionItem(BaseModel):
    label: str
    confidence: float


class PredictResponse(BaseModel):
    results: list[PredictionItem]
    latency_ms: float
    device: str


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
