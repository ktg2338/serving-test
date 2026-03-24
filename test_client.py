"""
서버 테스트 클라이언트
서버 실행 후: python test_client.py
"""

import json
import urllib.request

BASE_URL = "http://localhost:8000"


def post_json(url: str, data: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url) as resp:
        return json.loads(resp.read())


def main():
    # 1. Health check
    print("=" * 50)
    print("[1] Health Check")
    print("=" * 50)
    health = get_json(f"{BASE_URL}/health")
    print(json.dumps(health, indent=2))

    # 2. 단일 텍스트 추론
    print("\n" + "=" * 50)
    print("[2] Single Text Prediction")
    print("=" * 50)
    result = post_json(
        f"{BASE_URL}/predict",
        {"text": "This movie was absolutely fantastic! I loved every minute of it."},
    )
    print(json.dumps(result, indent=2))

    # 3. 배치 추론
    print("\n" + "=" * 50)
    print("[3] Batch Prediction")
    print("=" * 50)
    result = post_json(
        f"{BASE_URL}/predict",
        {
            "text": [
                "I love this product, it works perfectly!",
                "Terrible experience, worst purchase ever.",
                "It's okay, nothing special.",
                "이 영화 정말 재밌다!",
                "The service was disappointing and slow.",
            ]
        },
    )
    for i, r in enumerate(result["results"]):
        print(f"  [{i}] {r['label']:>8s}  (confidence: {r['confidence']:.4f})")
    print(f"\n  Device: {result['device']}")
    print(f"  Latency: {result['latency_ms']}ms (batch of {len(result['results'])})")


if __name__ == "__main__":
    main()
