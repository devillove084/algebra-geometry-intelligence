# Development and publishing

项目源文件使用 Quarto Markdown，数学公式使用 LaTeX 语法。网页由 Quarto 生成，PDF 使用 Quarto 内置 Typst，因此不依赖 TinyTeX。

## 基础主线与第一处交汇

第二、第三部分分别保留微积分和概率论的基础章节，第四部分“三种方法的交汇”位于 `chapters/04-connections/`，围绕同一个问题比较不同方法，而不挪动基础章节。数学语言作为全书总纲，不占部分编号。

本轮对照的 MIT 官方材料：

- 18.01SC [近似与曲线描绘](https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/pages/unit-2-applications-of-differentiation/part-a-approximation-and-curve-sketching/)由线性、二次近似进入函数形状；[Session 29](https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/pages/unit-2-applications-of-differentiation/part-b-optimization-related-rates-and-newtons-method/session-29-optimization-problems/)讨论优化的边界与内部极值。本书在现有近似章之后补中值定理和导数判别，不声称已逐讲覆盖课程中的所有求导技巧。
- 6.041SC [Lecture 3](https://ocw.mit.edu/courses/6-041sc-probabilistic-systems-analysis-and-applied-probability-fall-2013/pages/unit-i/lecture-3/)讨论事件独立性；[Lecture 5](https://ocw.mit.edu/courses/6-041sc-probabilistic-systems-analysis-and-applied-probability-fall-2013/pages/unit-i/lecture-5/)讨论离散随机变量、期望与方差；[Lecture 7](https://ocw.mit.edu/courses/6-041sc-probabilistic-systems-analysis-and-applied-probability-fall-2013/pages/unit-i/lecture-7/)讨论联合分布、条件化与独立。二项分布所需的位置计数在重复试验章内证明；当前聚焦有限支持，并非已覆盖全部可数离散分布。

第一处交汇选择“重复测量一个常数”，因为三条前置都已具备：

| 主线 | 需要的结论 | 对同一测量问题的贡献 |
|---|---|---|
| 线性代数 | 直线投影、正交残差与最小二乘 | 把读数向量投影到常数向量空间 |
| 微积分 | 导数符号、一元二次目标的全局最优性 | 求平方误差最小的常数 |
| 概率论 | 期望线性、方差和、协方差 | 判断估计规则的偏差与重复实验波动 |

交汇章进一步证明偏差—方差分解，以及固定线性无偏类别中的等权与逆方差最优性。不使用大数定律、中心极限定理或高斯似然，因此不需要假装这些尚未铺开的知识已经掌握。

后续主线继续补齐求导与积分计算、连续概率、独立随机变量及多元导数；有了连续密度、偏导和二次型后，再把一般线性拟合、正态噪声与似然放到下一层交汇中。基础路线不会因已有一个交汇例子而结束。

新章节的图形内嵌在 QMD 中，由 `make build` 执行；没有新增外部 SVG 生成脚本，不需要扩展 `make figures`。配图分镜分别展示候选变化、曲线斜率、概率质量聚合、独立与相关模型，以及同一测量问题的几何与统计区别。

## 环境

Ubuntu/Debian：

```bash
bash scripts/setup.sh
```

安装脚本会准备 Quarto、中文字体、用于 Mermaid PDF 渲染的 Chrome Headless Shell、Jupyter、`jupyter-cache`、NumPy、SymPy 和 Matplotlib。若系统中存在 `uv`，脚本会优先创建项目内的 `.venv`；否则回退到 Conda。默认 PyPI 镜像为清华镜像，可通过 `PIP_INDEX_URL` 覆盖。

HTML 统一使用 Noto Sans SC：页面先加载国内镜像、再加载 Google Fonts，并依次回退到本机 Noto/思源及系统中文字体。Typst 和 Matplotlib 使用同字体的系统名称 `Noto Sans CJK SC`。

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

`make figures` 重建通用线代插图以及正交补、QR、独立列分解和最小二乘插图。只重建正交补插图可运行：

```bash
.venv/bin/python scripts/generate_orthogonal_complement_figures.py --format svg
```

正交基与 QR 专题插图也可单独重建：

```bash
.venv/bin/python scripts/generate_orthonormal_qr_figures.py --format svg
```

独立列分解与最小二乘插图可分别重建：

```bash
.venv/bin/python scripts/generate_column_factorization_figure.py --format svg
.venv/bin/python scripts/generate_least_squares_figures.py
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
2. 使用 `uv` 恢复 Python 依赖缓存并创建 `.venv`；
3. 恢复 Quarto freeze 与 Jupyter 执行缓存；
4. 缓存未精确命中时安装 Noto CJK，保证重新执行的中文图表字体正确；
5. 重新生成 SVG，并确认生成结果已提交；
6. 执行 `quarto render --to html`；
7. 上传 `_site/`，再使用官方 Pages action 部署。

执行缓存按 Python 依赖、Quarto 配置和源码分层：依赖或配置不变时，新构建可复用未变化页面的 freeze 和代码单元；完全命中时还会跳过系统字体安装。

仓库 Pages 的 Source 必须设为 **GitHub Actions**。部署地址：

<https://devillove084.github.io/algebra-geometry-intelligence/>

工作流也可以在 GitHub Actions 页面手动触发。
