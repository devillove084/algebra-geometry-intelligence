# Development and publishing

项目源文件使用 Quarto Markdown，数学公式使用 LaTeX 语法。网页由 Quarto 生成，PDF 使用 Quarto 内置 Typst，因此不依赖 TinyTeX。

## 环境

Ubuntu/Debian：

```bash
bash scripts/setup.sh
conda activate algebra-geometry-intelligence
```

安装脚本会准备 Quarto、中文字体、Jupyter、`jupyter-cache`、NumPy、SymPy 和 Matplotlib。默认 PyPI 镜像为清华镜像，可通过 `PIP_INDEX_URL` 覆盖。

## 构建

```bash
bash scripts/build.sh html   # 输出网站到 _site/
bash scripts/build.sh pdf    # 使用 Typst 生成 PDF
bash scripts/build.sh all    # 两种格式
bash scripts/build.sh clean  # 清理构建产物
```

## 开发预览

```bash
bash scripts/serve.sh
```

默认监听 `0.0.0.0:4200` 并打印本机与网络 URL：

```text
http://127.0.0.1:4200/
http://<本机地址>:4200/
```

指定其他端口：

```bash
bash scripts/serve.sh 8080
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
