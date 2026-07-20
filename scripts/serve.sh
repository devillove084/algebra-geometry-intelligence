#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
HOST="${HOST:-0.0.0.0}"
PORT="${1:-${PORT:-4200}}"

# 自动检测 venv
if [ -f "${PROJECT_ROOT}/.venv/bin/python3" ]; then
  export QUARTO_PYTHON="${PROJECT_ROOT}/.venv/bin/python3"
fi

if ! command -v quarto >/dev/null 2>&1; then
  echo "错误：未找到 Quarto，请先运行 bash scripts/setup.sh。" >&2
  exit 1
fi

PYTHON_BIN="${QUARTO_PYTHON:-python3}"
if ! "${PYTHON_BIN}" -c "import nbformat, numpy" >/dev/null 2>&1; then
  echo "错误：${PYTHON_BIN} 缺少 nbformat 或 NumPy。请先运行 bash scripts/setup.sh。" >&2
  exit 1
fi

if ! [[ "${PORT}" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
  echo "错误：端口必须是 1 到 65535 之间的整数。" >&2
  exit 2
fi

echo "Algebra, Geometry, and Intelligence：开发预览服务器"
echo "本机访问：http://127.0.0.1:${PORT}/"

if command -v hostname >/dev/null 2>&1; then
  for address in $(hostname -I 2>/dev/null); do
    case "${address}" in
      127.*|169.254.*|*:*) ;;
      *) echo "网络访问：http://${address}:${PORT}/" ;;
    esac
  done
fi

cat <<EOF

监听地址：${HOST}:${PORT}
按 Ctrl+C 停止。
警告：这是无认证、无 TLS 的开发服务器，只应暴露在可信网络中。
EOF

cd "${PROJECT_ROOT}"
exec quarto preview --host "${HOST}" --port "${PORT}" --no-browser --render html
