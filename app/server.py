"""
FastAPI 서버
- /predict : 감성분석 추론
- /health  : 헬스체크
- lifespan으로 서버 시작 시 모델 로드
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.model import MODEL_NAME, model_manager
from app.pipeline import predict
from app.schema import HealthResponse, PredictRequest, PredictResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: 모델 로드
    model_manager.load()
    yield
    # Shutdown: 정리
    print("[Server] Shutting down.")


app = FastAPI(
    title="Transformer Serving Test",
    description="PyTorch + HuggingFace 모델 서빙 학습용 API (Apple M5 MPS)",
    lifespan=lifespan,
)


@app.post("/predict", response_model=PredictResponse)
async def predict_endpoint(request: PredictRequest):
    """
    텍스트 감성분석

    - 단일: {"text": "I love this"}
    - 배치: {"text": ["I love this", "I hate this"]}
    """
    result = predict(request.text)
    return result


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok" if model_manager.model is not None else "not_ready",
        model=MODEL_NAME,
        device=str(model_manager.device),
    )
