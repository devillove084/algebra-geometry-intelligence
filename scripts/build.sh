#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TARGET="${1:-all}"

if ! command -v quarto >/dev/null 2>&1; then
  echo "错误：未找到 Quarto，请先运行 bash scripts/setup.sh。" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"

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
