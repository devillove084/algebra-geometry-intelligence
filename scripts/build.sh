#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TARGET="${1:-html}"

# 自动检测 venv
if [ -f "${PROJECT_ROOT}/.venv/bin/python3" ]; then
  export QUARTO_PYTHON="${PROJECT_ROOT}/.venv/bin/python3"
fi

if ! command -v quarto >/dev/null 2>&1; then
  echo "错误：未找到 Quarto，请先运行 bash scripts/setup.sh。" >&2
  exit 1
fi

if [ "${TARGET}" != "clean" ]; then
  PYTHON_BIN="${QUARTO_PYTHON:-python3}"
  if ! "${PYTHON_BIN}" -c "import nbformat, numpy" >/dev/null 2>&1; then
    echo "错误：${PYTHON_BIN} 缺少 nbformat 或 NumPy。请先运行 bash scripts/setup.sh。" >&2
    exit 1
  fi
fi

cd "${PROJECT_ROOT}"

if [ "${TARGET}" != "clean" ]; then
  "${PYTHON_BIN}" scripts/check_unicode_math.py
  export PYTHONWARNINGS="error:Glyph${PYTHONWARNINGS:+,${PYTHONWARNINGS}}"
fi

case "${TARGET}" in
  html)
    quarto render --to html
    ;;
  pdf|typst)
    quarto render --to typst
    ;;
  all)
    quarto render --to html
    quarto render --to typst
    ;;
  clean)
    rm -rf _site .quarto _freeze
    echo "构建产物已清理。"
    ;;
  *)
    echo "用法：bash scripts/build.sh [html|pdf|typst|all|clean]" >&2
    exit 2
    ;;
esac
