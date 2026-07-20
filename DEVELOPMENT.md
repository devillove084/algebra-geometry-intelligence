# Development and publishing

项目源文件使用 Quarto Markdown，数学公式使用 LaTeX 语法。网页由 Quarto 生成，PDF 使用 Quarto 内置 Typst，因此不依赖 TinyTeX。

## 环境

Ubuntu/Debian：

```bash
bash scripts/setup.sh
```

安装脚本会准备 Quarto、中文字体、用于 Mermaid PDF 渲染的 Chrome Headless Shell、Jupyter、`jupyter-cache`、NumPy、SymPy 和 Matplotlib。若系统中存在 `uv`，脚本会优先创建项目内的 `.venv`；否则回退到 Conda。默认 PyPI 镜像为清华镜像，可通过 `PIP_INDEX_URL` 覆盖。

若环境已经安装完成、只需消除 Pandoc 的中文翻译警告，可以单独运行：

```bash
make translations
```

## 构建

```bash
make build  # 输出网站到 _site/
make pdf    # 使用 Typst 生成 PDF
make all      # 两种格式
make figures  # 重新生成线性代数 SVG 插图
make clean    # 清理构建产物
```

## 开发预览

```bash
make serve
```

默认监听 `0.0.0.0:4200` 并打印本机与网络 URL：

```text
http://127.0.0.1:4200/
http://<本机地址>:4200/
```

指定其他端口：

```bash
PORT=8080 make serve
```

这是无认证、无 TLS 的开发服务器，仅应暴露在可信网络中。跨主机访问还需要防火墙或云安全组允许对应 TCP 端口。

## GitHub Pages

工作流位于 `.github/workflows/publish.yml`：

1. push 到 `main`；
2. 安装 Conda 环境与 Quarto；
3. 执行 `quarto render --to html`；
4. 上传 `_site/`；
5. 使用官方 Pages action 部署。

仓库 Pages 的 Source 必须设为 **GitHub Actions**。部署地址：

<https://devillove084.github.io/algebra-geometry-intelligence/>

工作流也可以在 GitHub Actions 页面手动触发。
