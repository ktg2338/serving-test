"""
전처리 → 추론 → 후처리 파이프라인
"""

import time

import torch

from app.model import model_manager


def preprocess(text: str | list[str]) -> dict:
    """
    텍스트 → 토큰화 → 텐서 변환
    - padding/truncation 처리
    - max_length=512 (DistilBERT 기본)
    - 디바이스로 이동
    """
    tokenizer = model_manager.tokenizer
    device = model_manager.device

    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    # GPU(MPS)로 텐서 이동
    inputs = {k: v.to(device) for k, v in inputs.items()}
    return inputs


def inference(inputs: dict) -> torch.Tensor:
    """
    모델 추론 (no_grad로 그래디언트 비활성화)
    반환: logits tensor
    """
    model = model_manager.model

    with torch.no_grad():
        outputs = model(**inputs)

    return outputs.logits


def postprocess(logits: torch.Tensor) -> list[dict]:
    """
    logits → softmax → 라벨 + 확률값 변환
    배치 처리 지원
    """
    id2label = model_manager.id2label

    probabilities = torch.softmax(logits, dim=-1)
    predicted_classes = torch.argmax(probabilities, dim=-1)
    confidences = probabilities.max(dim=-1).values

    results = []
    for i in range(logits.size(0)):
        label = id2label[predicted_classes[i].item()]
        confidence = round(confidences[i].item(), 4)
        results.append({"label": label, "confidence": confidence})

    return results


def predict(text: str | list[str]) -> dict:
    """
    전체 파이프라인 실행: 전처리 → 추론 → 후처리
    latency 측정 포함
    """
    start = time.perf_counter()

    # [1] 전처리
    inputs = preprocess(text)

    # [2] 추론
    logits = inference(inputs)

    # [3] 후처리
    results = postprocess(logits)

    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

    return {
        "results": results,
        "latency_ms": elapsed_ms,
        "device": str(model_manager.device),
    }
