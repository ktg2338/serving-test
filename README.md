# Serving Test

PyTorch 트랜스포머 모델을 FastAPI로 서빙하는 프로젝트입니다. Apple M 시리즈 GPU(MPS)에 최적화된 감성 분석 추론 API를 제공합니다.

## 기술 스택

- **API**: FastAPI + Uvicorn
- **ML**: PyTorch + HuggingFace Transformers
- **모델**: `distilbert-base-uncased-finetuned-sst-2-english` (감성 분류)
- **하드웨어**: Apple MPS 가속 (CPU 폴백 지원)

## 프로젝트 구조

```
app/
├── server.py      # FastAPI 앱, 엔드포인트 정의
├── model.py       # ModelManager 싱글턴 (모델 로딩/관리)
├── pipeline.py    # 전처리 → 추론 → 후처리 파이프라인
└── schema.py      # Pydantic 요청/응답 스키마
test_client.py     # API 테스트 클라이언트
run.sh             # 서버 실행 스크립트
```

## 실행 방법

```bash
./run.sh
```

서버가 `http://localhost:8000`에서 실행되며, API 문서는 `http://localhost:8000/docs`에서 확인할 수 있습니다.

## API

### POST /predict

텍스트 감성 분석 (단건 및 배치 지원)

```bash
# 단건 예측
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was amazing!"}'

# 배치 예측
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": ["I love this!", "This is terrible."]}'
```

응답 예시:

```json
{
  "results": [
    {"label": "POSITIVE", "confidence": 0.9998}
  ],
  "latency_ms": 12.5,
  "device": "mps"
}
```

### GET /health

서버 및 모델 상태 확인

## 테스트

```bash
python test_client.py
```

## 주요 특징

- **싱글턴 모델 관리**: 모델을 한 번만 로드하여 재사용
- **MPS 가속**: Apple M 시리즈 GPU 활용, CPU 자동 폴백
- **워밍업**: 서버 시작 시 더미 추론으로 MPS 커널 컴파일 지연 방지
- **레이턴시 측정**: 모든 응답에 추론 소요 시간 포함
