"""
Model Loader - Apple M5 MPS 최적화
- 모델: distilbert-base-uncased-finetuned-sst-2-english (~260MB)
  16GB RAM에서 여유롭게 동작하는 경량 감성분석 모델
- LoRA 파인튜닝 모델이 있으면 자동으로 로드
- Device 우선순위: MPS(Metal) > CPU
"""

from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
LORA_PATH = Path(__file__).parent.parent / "models" / "lora-finetuned" / "final"


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
        base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

        # LoRA 어댑터가 있으면 합쳐서 로드
        if LORA_PATH.exists():
            print(f"[ModelManager] LoRA 어댑터 로드: {LORA_PATH}")
            self.model = PeftModel.from_pretrained(base_model, str(LORA_PATH))
        else:
            print("[ModelManager] LoRA 없음 → 원본 모델 사용")
            self.model = base_model

        # 추론 모드 전환 + 디바이스 배치
        self.model.eval()
        self.model.to(self.device)

        # 라벨 매핑 (PeftModel은 base_model 안에 config가 있음)
        config = getattr(self.model, "config", None) or self.model.base_model.config
        self.id2label = config.id2label

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
