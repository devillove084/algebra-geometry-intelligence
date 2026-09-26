#!/usr/bin/env python3
"""Generate the two least-squares diagrams without importing other scripts."""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "images" / "linear-algebra"
BLUE = "#1769aa"
GREEN = "#218739"
PURPLE = "#8a3fa0"
ORANGE = "#a96408"
INK = "#263238"
GREY = "#607080"


def configure() -> None:
    matplotlib.rcParams.update({
        "font.family": "Noto Sans CJK SC",
        "font.size": 13,
        "text.color": INK,
        "mathtext.fontset": "dejavusans",
        "svg.fonttype": "path",
        "svg.hashsalt": "least-squares-figures",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def arrow(ax, start, end, color, width=2.8) -> None:
    ax.annotate("", xy=end, xytext=start, arrowprops={
        "arrowstyle": "-|>", "mutation_scale": 17,
        "color": color, "lw": width, "shrinkA": 0, "shrinkB": 0,
    })


def projection():
    fig = plt.figure(figsize=(12, 6.4))
    fig.suptitle("最小二乘：在线上找到离目标最近的输出", y=0.96,
                 fontsize=20, weight="bold")
    ax = fig.add_axes([0.035, 0.24, 0.58, 0.56])
    ax.set_aspect("equal")
    ax.set_xlim(-0.45, 3.85)
    ax.set_ylim(-0.72, 1.85)
    ax.axis("off")

    # An isometric view of span(a, b): q=a/||a||, r=e/||e||.
    # Both coordinates use the same scale; no perspective distorts the angle.
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 2.0])
    x = (a @ b) / (a @ a)
    p = x * a
    e = b - p
    q, r = a / np.linalg.norm(a), e / np.linalg.norm(e)
    assert np.isclose(q @ r, 0.0)
    p2 = np.array([q @ p, 0.0])
    b2 = np.array([q @ b, r @ b])
    assert np.isclose(np.linalg.norm(b2), np.linalg.norm(b))

    ax.plot([-0.35, 3.65], [0, 0], color=GREEN, lw=2, alpha=0.5)
    ax.text(0.0, -0.56, r"$\mathcal{C}(A)=\operatorname{span}\{(1,2,3)^{\mathsf{T}}\}$",
            color=GREEN, fontsize=14)
    arrow(ax, (0, 0), b2, BLUE)
    arrow(ax, (0, 0), p2, GREEN, 3.6)
    arrow(ax, p2, b2, PURPLE)
    ax.scatter([0, p2[0], b2[0]], [0, 0, b2[1]],
               c=[INK, GREEN, BLUE], s=24, zorder=5)
    d = 0.13
    ax.plot([p2[0] - d, p2[0] - d, p2[0]], [0, d, d], color=PURPLE, lw=1.7)
    ax.text(-0.07, -0.24, r"$0$", fontsize=15)
    ax.text(1.15, -0.26, r"$\mathbf{p}=A\widehat{x}$", color=GREEN, fontsize=17)
    ax.text(1.3, 0.48, r"$\mathbf{b}$", color=BLUE, fontsize=18)
    ax.text(p2[0] + 0.13, b2[1] / 2, r"$\mathbf{e}$", color=PURPLE, fontsize=18)
    ax.text(b2[0] - 0.12, b2[1] + 0.2, r"$\mathbf{b}=(1,2,2)^{\mathsf{T}}$",
            color=BLUE, ha="center", fontsize=14)
    ax.text(0.1, 1.5, "改变 x：输出沿绿线移动\n垂足处：残差不再含沿线的分量",
            fontsize=14, linespacing=1.7)

    info = fig.add_axes([0.66, 0.20, 0.32, 0.64])
    info.axis("off")
    info.add_patch(FancyBboxPatch((0, 0), 0.99, 0.99, boxstyle="round,pad=0.01",
                                facecolor="#f5f8fb", edgecolor="#d8e0e8"))
    rows = [
        (0.89, "从系数到输出，再看剩余", INK, 15),
        (0.74, r"$\widehat{x}=\dfrac{11}{14}$", INK, 20),
        (0.57, r"$\mathbf{p}=\dfrac{1}{14}(11,22,33)^{\mathsf{T}}$", GREEN, 16),
        (0.40, r"$\mathbf{e}=\dfrac{1}{14}(3,6,-5)^{\mathsf{T}}$", PURPLE, 16),
        (0.25, r"$A^{\mathsf{T}}\mathbf{e}=0,\quad \|\mathbf{e}\|^2=5/14$", PURPLE, 15),
        (0.10, r"$\mathbf{b}=\mathbf{p}+\mathbf{e}$", BLUE, 17),
    ]
    for y, text, color, size in rows:
        info.text(0.07, y, text, color=color, fontsize=size, va="center")
    fig.text(0.5, 0.075, "三维中的平面 span{a, b} 被等距展开；水平沿 a，竖直沿 e。\n"
             "这不是原三维坐标的前两维，两个方向使用相同长度尺度。",
             ha="center", fontsize=12, color=GREY, linespacing=1.6)
    return fig


def equations():
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.text(6, 9.65, "从正交残差到方程：同一个最近输出", ha="center",
            fontsize=20, weight="bold")

    def box(x, y, w, h, title, formula, note, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                   facecolor="#f8fafc", edgecolor=color, linewidth=1.7))
        ax.text(x + w / 2, y + h - 0.29, title, ha="center", fontsize=14,
                color=color, weight="bold")
        ax.text(x + w / 2, y + h / 2 - 0.02, formula, ha="center", va="center",
                fontsize=19, color=color)
        ax.text(x + w / 2, y + 0.24, note, ha="center", fontsize=12)

    box(2.4, 7.75, 7.2, 1.45, "无解的过定方程：改找最近可达输出",
        r"$\min_x\|Ax-\mathbf{b}\|^2$", "最小化误差平方 ⇔ 投影到列空间", BLUE)
    arrow(ax, (6, 7.70), (6, 7.28), GREY, 2)
    box(2.4, 5.82, 7.2, 1.4, "正交残差条件（不要求满列秩）",
        r"$\mathbf{e}=\mathbf{b}-A\widehat{x}\ \perp\ \mathcal{C}(A)$",
        r"残差与每列正交 $\Longleftrightarrow A^{\mathsf{T}}\mathbf{e}=0$", PURPLE)

    arrow(ax, (6, 5.76), (6, 5.28), GREY, 2)
    box(2.4, 3.82, 7.2, 1.4, "正规方程（总相容）",
        r"$A^{\mathsf{T}}A\widehat{x}=A^{\mathsf{T}}\mathbf{b}$",
        r"本例：$14\widehat{x}=11$", BLUE)

    arrow(ax, (4.0, 3.76), (3.05, 2.94), ORANGE, 2)
    arrow(ax, (8.0, 3.76), (8.95, 2.94), GREEN, 2)
    ax.text(2.68, 3.35, "满列秩", color=ORANGE, fontsize=12, ha="center")
    ax.text(9.33, 3.35, "秩亏", color=GREEN, fontsize=12, ha="center")
    box(0.55, 1.10, 5.05, 1.75, "系数唯一",
        r"$A^{\mathsf{T}}A$ 正定可逆",
        r"$\widehat{x}=(A^{\mathsf{T}}A)^{-1}A^{\mathsf{T}}\mathbf{b}$", ORANGE)
    box(6.4, 1.10, 5.05, 1.75, "系数不唯一",
        r"$\widehat{x}+\mathbf{z},\quad A\mathbf{z}=0$",
        r"同一个唯一投影输出 $A\widehat{x}$", GREEN)
    ax.text(6, 0.50, r"两种情形都有同一个几何结论： $\mathbf{p}$ 唯一，$\mathbf{b}-\mathbf{p}\perp\mathcal{C}(A)$",
            ha="center", fontsize=14)
    ax.text(6, 0.15, "投影唯一来自正交分解；系数唯一还需要列线性无关。",
            ha="center", color=GREY, fontsize=12)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-dir", type=Path,
                        help="Also write PNG previews into an existing directory")
    args = parser.parse_args()
    if args.preview_dir is not None and not args.preview_dir.is_dir():
        parser.error("--preview-dir must already exist")
    configure()
    figures = [
        ("least-squares-projection-residual", projection,
         "最小二乘投影与正交残差",
         "在 span(a,b) 的等距平面截面中，绿色列空间沿水平方向，蓝色 b 的投影为 p，"
         "紫色残差 e 从 p 指向 b，与列空间垂直。a=(1,2,3)，b=(1,2,2)，投影系数为 11/14。"),
        ("least-squares-normal-equations", equations,
         "正交残差、正规方程与解的唯一性",
         "最小二乘等价于残差与列空间正交，从而得到正规方程。满列秩时系数唯一；秩亏时"
         "投影输出仍唯一而参数可沿零空间变化。图中同时强调正规方程中的投影结构。"),
    ]
    for stem, build, title, description in figures:
        fig = build()
        path = OUTPUT / f"{stem}.svg"
        fig.savefig(path, metadata={
            "Date": None, "Description": description,
        })
        svg = path.read_text(encoding="utf-8")
        start = svg.index("<svg")
        end = svg.index(">", start)
        svg = (svg[:end] + ' role="img" aria-labelledby="figure-title figure-desc"'
               + svg[end:end + 1]
               + f'\n<title id="figure-title">{escape(title)}</title>'
               + f'\n<desc id="figure-desc">{escape(description)}</desc>'
               + svg[end + 1:])
        svg = "\n".join(line.rstrip() for line in svg.splitlines()) + "\n"
        path.write_text(svg, encoding="utf-8")
        if args.preview_dir is not None:
            fig.savefig(args.preview_dir / f"{stem}.png", dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    main()
