"""
Model Loader - Apple M5 MPS 최적화
- 모델: distilbert-base-uncased-finetuned-sst-2-english (~260MB)
  16GB RAM에서 여유롭게 동작하는 경량 감성분석 모델
- Device 우선순위: MPS(Metal) > CPU
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"


def get_device() -> torch.device:
    """MPS 사용 가능하면 MPS, 아니면 CPU"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ModelManager:
    """싱글턴 패턴으로 모델/토크나이저를 한 번만 로드"""

    def __init__(self):
        self.device = get_device()
        self.model = None
        self.tokenizer = None
        self.id2label = None

    def load(self):
        print(f"[ModelManager] Loading model: {MODEL_NAME}")
        print(f"[ModelManager] Device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

        # 추론 모드 전환 + 디바이스 배치
        self.model.eval()
        self.model.to(self.device)

        # 라벨 매핑
        self.id2label = self.model.config.id2label

        # 워밍업 추론 (MPS 첫 호출 지연 방지)
        self._warmup()

        print(f"[ModelManager] Ready. Labels: {self.id2label}")

    def _warmup(self):
        """첫 추론은 MPS 커널 컴파일로 느림 → 미리 한 번 실행"""
        dummy = self.tokenizer("warmup", return_tensors="pt")
        dummy = {k: v.to(self.device) for k, v in dummy.items()}
        with torch.no_grad():
            self.model(**dummy)
        print("[ModelManager] Warmup done.")


# 전역 인스턴스
model_manager = ModelManager()
