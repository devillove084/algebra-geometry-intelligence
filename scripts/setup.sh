#!/usr/bin/env bash

set -euo pipefail

QUARTO_VERSION="${QUARTO_VERSION:-1.9.38}"
ENV_NAME="${ENV_NAME:-algebra-geometry-intelligence}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export PIP_INDEX_URL
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

log() {
  printf '\n\033[1;34m==> %s\033[0m\n' "$1"
}

fail() {
  printf '\n\033[1;31m错误：%s\033[0m\n' "$1" >&2
  exit 1
}

if [[ "$(uname -s)" != "Linux" ]]; then
  fail "当前脚本面向 Ubuntu/Debian Linux；其他系统请参考 README.md。"
fi

if ! command -v apt-get >/dev/null 2>&1; then
  fail "未找到 apt-get。当前脚本仅支持 Ubuntu/Debian。"
fi

if [[ "$(id -u)" -eq 0 ]]; then
  SUDO=()
elif command -v sudo >/dev/null 2>&1; then
  SUDO=(sudo)
else
  fail "安装系统软件需要 root 权限或 sudo。"
fi

log "安装基础工具和中文字体"
"${SUDO[@]}" apt-get update
"${SUDO[@]}" apt-get install -y curl ca-certificates fonts-noto-cjk fontconfig

if command -v quarto >/dev/null 2>&1; then
  log "检测到 Quarto $(quarto --version)，跳过安装"
else
  case "$(dpkg --print-architecture)" in
    arm64)
      QUARTO_ARCH="arm64"
      ;;
    amd64)
      QUARTO_ARCH="amd64"
      ;;
    *)
      fail "Quarto 自动安装暂不支持架构：$(dpkg --print-architecture)"
      ;;
  esac

  log "安装 Quarto ${QUARTO_VERSION} (${QUARTO_ARCH})"
  TEMP_DIR="$(mktemp -d)"
  trap 'rm -rf "${TEMP_DIR}"' EXIT
  QUARTO_PACKAGE="quarto-${QUARTO_VERSION}-linux-${QUARTO_ARCH}.deb"
  curl -fL \
    "https://github.com/quarto-dev/quarto-cli/releases/download/v${QUARTO_VERSION}/${QUARTO_PACKAGE}" \
    -o "${TEMP_DIR}/${QUARTO_PACKAGE}"
  "${SUDO[@]}" apt-get install -y "${TEMP_DIR}/${QUARTO_PACKAGE}"
fi

if ! command -v conda >/dev/null 2>&1; then
  fail "未找到 Conda。请先安装 Miniconda，然后重新运行本脚本。"
fi

log "创建或更新 Conda 环境 ${ENV_NAME}（PyPI 镜像：${PIP_INDEX_URL}）"
if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  conda env update --name "${ENV_NAME}" --file "${PROJECT_ROOT}/environment.yml" --prune
else
  conda env create --file "${PROJECT_ROOT}/environment.yml"
fi

log "验证工具链"
quarto --version
quarto typst --version
conda run --name "${ENV_NAME}" python -c \
  "import jupyter, jupyter_cache, numpy, sympy, matplotlib; print('Python 数学与缓存环境正常')"

cat <<EOF

安装完成。

开始写作：
  conda activate ${ENV_NAME}
  cd ${PROJECT_ROOT}
  quarto preview

生成输出：
  quarto render --to html
  quarto render --to typst

Typst 已内置在 Quarto 中，不需要安装 TinyTeX。
EOF
