# Serving Test

PyTorch 트랜스포머 모델을 FastAPI로 서빙하는 프로젝트입니다. Apple M 시리즈 GPU(MPS)에 최적화된 감성 분석 추론 API를 제공합니다.

## 기술 스택

- **API**: FastAPI + Uvicorn
- **ML**: PyTorch + HuggingFace Transformers
- **파인튜닝**: LoRA (PEFT)
- **모델**: `distilbert-base-uncased-finetuned-sst-2-english` (감성 분류)
- **하드웨어**: Apple MPS 가속 (CPU 폴백 지원)

## 프로젝트 구조

```
app/
├── server.py      # FastAPI 앱, 엔드포인트 정의
├── model.py       # ModelManager 싱글턴 (모델 로딩/관리, LoRA 자동 감지)
├── pipeline.py    # 전처리 → 추론 → 후처리 파이프라인
└── schema.py      # Pydantic 요청/응답 스키마
data/
├── raw_reviews.xlsx    # 원본 데이터 (전처리 연습용)
├── create_raw_data.py  # 원본 엑셀 생성 스크립트
├── preprocess.py       # 데이터 전처리 파이프라인
├── train.json          # 정제된 학습 데이터
└── eval.json           # 정제된 검증 데이터
train/
└── finetune_lora.py    # LoRA 파인튜닝 스크립트
models/
└── lora-finetuned/     # 파인튜닝된 LoRA 어댑터 저장 경로
test_client.py          # API 테스트 클라이언트
run.sh                  # 서버 실행 스크립트
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

## LoRA 파인튜닝

### 1. 데이터 전처리

원본 엑셀 데이터를 정제하여 학습용 JSON으로 변환합니다.

```bash
# 원본 데이터 생성 (샘플)
python data/create_raw_data.py

# 전처리 실행
python data/preprocess.py
```

전처리 항목: 결측값 제거, 라벨 통일, 텍스트 정제 (HTML/URL/이모지/반복문자), 스팸 필터링, 길이 필터링, 중복 제거, train/eval 분할

### 2. LoRA 학습

```bash
# 기본 실행
python train/finetune_lora.py

# 하이퍼파라미터 조정
python train/finetune_lora.py --epochs 20 --learning-rate 1e-4 --lora-r 16 --batch-size 16
```

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `--epochs` | 10 | 학습 에폭 수 |
| `--batch-size` | 8 | 배치 크기 |
| `--learning-rate` | 2e-4 | 학습률 |
| `--lora-r` | 8 | LoRA rank |
| `--lora-alpha` | 16 | LoRA alpha |
| `--lora-dropout` | 0.1 | LoRA dropout |

학습된 모델은 `models/lora-finetuned/final/`에 저장됩니다.

### 3. 서빙 연동

`models/lora-finetuned/final/` 경로에 LoRA 어댑터가 있으면 서버 시작 시 자동으로 파인튜닝 모델을 로드합니다. 어댑터가 없으면 원본 모델로 서빙됩니다.

## 주요 특징

- **싱글턴 모델 관리**: 모델을 한 번만 로드하여 재사용
- **MPS 가속**: Apple M 시리즈 GPU 활용, CPU 자동 폴백
- **LoRA 자동 감지**: 파인튜닝 어댑터 존재 시 자동 로드
- **워밍업**: 서버 시작 시 더미 추론으로 MPS 커널 컴파일 지연 방지
- **레이턴시 측정**: 모든 응답에 추론 소요 시간 포함
