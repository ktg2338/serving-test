#!/bin/bash
# 서버 실행 스크립트
# Usage: ./run.sh

set -e

echo "=== Transformer Serving Test ==="
echo "Installing dependencies..."
pip install -q -r requirements.txt

echo ""
echo "Starting server on http://localhost:8000"
echo "API docs: http://localhost:8000/docs"
echo ""

uvicorn app.server:app --host 0.0.0.0 --port 8000 --reload
