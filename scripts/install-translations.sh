#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SOURCE_FILE="${PROJECT_ROOT}/config/pandoc/translations/zh.yaml"

if ! command -v quarto >/dev/null 2>&1; then
  echo "错误：未找到 Quarto。" >&2
  exit 1
fi

QUARTO_BIN="$(readlink -f "$(command -v quarto)")"
QUARTO_ROOT="$(cd "$(dirname "${QUARTO_BIN}")/.." && pwd)"
TARGET_DIR="${QUARTO_ROOT}/share/pandoc/datadir/translations"
TARGET_FILE="${TARGET_DIR}/zh.yaml"

install_translation() {
  install -d "${TARGET_DIR}"
  install -m 0644 "${SOURCE_FILE}" "${TARGET_FILE}"
}

if [ -w "$(dirname "${TARGET_DIR}")" ] || { [ -d "${TARGET_DIR}" ] && [ -w "${TARGET_DIR}" ]; }; then
  install_translation
elif command -v sudo >/dev/null 2>&1; then
  sudo install -d "${TARGET_DIR}"
  sudo install -m 0644 "${SOURCE_FILE}" "${TARGET_FILE}"
else
  echo "错误：写入 ${TARGET_DIR} 需要管理员权限，但未找到 sudo。" >&2
  exit 1
fi

echo "已安装 Pandoc 中文翻译：${TARGET_FILE}"
