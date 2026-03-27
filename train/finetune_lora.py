"""
LoRA 파인튜닝 스크립트
- 모델: distilbert-base-uncased-finetuned-sst-2-english
- 방식: LoRA (Low-Rank Adaptation) — 전체 파라미터의 ~0.5%만 학습
- 디바이스: MPS (Apple Silicon) > CPU 자동 선택
- 출력: models/lora-finetuned/ 디렉토리에 저장
"""

import json
import argparse
from pathlib import Path

import torch
import evaluate
import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, TaskType, get_peft_model

# ──────────────────────────────────────────
# 설정
# ──────────────────────────────────────────
BASE_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "models" / "lora-finetuned"

DEFAULT_ARGS = {
    "epochs": 10,
    "batch_size": 8,
    "learning_rate": 2e-4,
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
}


# ──────────────────────────────────────────
# 디바이스 선택
# ──────────────────────────────────────────
def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ──────────────────────────────────────────
# 데이터 로드 & 토크나이즈
# ──────────────────────────────────────────
def load_data(path: Path) -> Dataset:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Dataset.from_list(data)


def tokenize_dataset(dataset: Dataset, tokenizer) -> Dataset:
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=128,
        )

    dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format("torch")
    return dataset


# ──────────────────────────────────────────
# 메트릭
# ──────────────────────────────────────────
accuracy_metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)


# ──────────────────────────────────────────
# LoRA 설정 & 모델 준비
# ──────────────────────────────────────────
def prepare_model(args):
    print(f"[모델] {BASE_MODEL} 로드 중...")
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=2
    )

    # LoRA 설정
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_lin", "v_lin"],  # DistilBERT attention layers
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ──────────────────────────────────────────
# 학습 실행
# ──────────────────────────────────────────
def train(args):
    device = get_device()
    print(f"[디바이스] {device}")

    # 토크나이저
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # 데이터 로드 & 토크나이즈
    print("[데이터] 로드 중...")
    train_dataset = tokenize_dataset(load_data(DATA_DIR / "train.json"), tokenizer)
    eval_dataset = tokenize_dataset(load_data(DATA_DIR / "eval.json"), tokenizer)
    print(f"  train: {len(train_dataset)}건, eval: {len(eval_dataset)}건")

    # 모델
    model = prepare_model(args)

    # MPS에서 float32 사용 (half precision 미지원 이슈 방지)
    use_fp16 = device.type == "cuda"

    # 학습 설정
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=5,
        fp16=use_fp16,
        report_to="none",  # wandb 등 외부 로깅 비활성화
        seed=42,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # 학습
    print("\n" + "=" * 50)
    print("학습 시작")
    print("=" * 50)
    trainer.train()

    # 최종 평가
    print("\n" + "=" * 50)
    print("최종 평가")
    print("=" * 50)
    results = trainer.evaluate()
    print(f"  Loss:     {results['eval_loss']:.4f}")
    print(f"  Accuracy: {results['eval_accuracy']:.4f}")

    # 모델 저장
    save_path = OUTPUT_DIR / "final"
    trainer.save_model(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    print(f"\n모델 저장 완료: {save_path}")

    return results


# ──────────────────────────────────────────
# CLI
# ──────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning for DistilBERT")
    parser.add_argument("--epochs", type=int, default=DEFAULT_ARGS["epochs"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_ARGS["batch_size"])
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_ARGS["learning_rate"])
    parser.add_argument("--lora-r", type=int, default=DEFAULT_ARGS["lora_r"])
    parser.add_argument("--lora-alpha", type=int, default=DEFAULT_ARGS["lora_alpha"])
    parser.add_argument("--lora-dropout", type=float, default=DEFAULT_ARGS["lora_dropout"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
