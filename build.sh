#!/usr/bin/env bash
# Exit immediately if a command exits with a non-zero status
set -o errexit

echo "=== 1. Building React Frontend ==="
cd frontend
npm install
npm run build
cd ..

echo "=== 2. Installing Backend Python Dependencies ==="
pip install --upgrade pip
pip install -r backend/requirements.txt

echo "=== Build Completed Successfully ==="
