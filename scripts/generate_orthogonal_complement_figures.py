#!/usr/bin/env python3
"""Generate the orthogonal-complement geometry storyboards."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from generate_linear_algebra_figures import (
    BLUE,
    CHINESE_FIGURE_STYLE,
    GREEN,
    GRID,
    ORANGE,
    PURPLE,
    SLATE,
    configure_matplotlib,
    draw_vector_2d,
    geometry_label,
    geometry_note,
    geometry_panels,
    normalize_svg,
    right_angle,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "images" / "linear-algebra"

A = np.array([[1.0, 2.0, 0.0], [2.0, 4.0, 0.0]])
r = np.array([1.0, 2.0, 0.0])
n = np.array([-2.0, 1.0, 0.0])
e3 = np.array([0.0, 0.0, 1.0])
c = np.array([1.0, 2.0])
h = np.array([-2.0, 1.0])
p = np.array([1.0, 2.0, 0.0])
b = np.array([5.0, 10.0])


def assert_storyboard_numbers() -> None:
    """Keep every visual tied to the same rank-one example."""
    assert np.allclose(A[0], r)
    assert np.allclose(A[1], 2.0 * r)
    assert np.allclose(A @ n, [0.0, 0.0])
    assert np.allclose(A @ e3, [0.0, 0.0])
    assert np.allclose(A @ p, b)
    assert np.allclose(A @ np.array([-1.0, 3.0, 0.0]), b)
    assert np.allclose(r[:2] @ n[:2], 0.0)
    assert np.linalg.matrix_rank(A) == 1
    assert np.linalg.matrix_rank(np.column_stack((n, e3))) == 2
    assert np.allclose(A @ r, 5 * c)
    assert np.allclose(h @ A, np.zeros(3))
    assert np.allclose(p @ n, 0.0) and np.allclose(p @ e3, 0.0)
    assert np.allclose(h @ c, 0.0)
    assert np.allclose(h @ b, 0.0)
    assert np.allclose(h @ np.array([5.0, 9.0]), -1.0)
    assert np.allclose(p[:2] + n[:2], [-1.0, 3.0])
    for t in (-1.0, 0.0, 1.0):
        for s in (-2.0, 0.0, 3.0):
            x = p + t * n + s * e3
            assert np.isclose(x @ x, 5.0 + 5.0 * t * t + s * s)


def draw_x3_free_marker(ax: Axes, *, text_y: float = 0.97) -> None:
    """Mark that a 2-D x3=constant slice is not the whole 3-D set."""

    ax.text(
        0.96,
        text_y,
        r"$x_3=s$ 自由（出画面）",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.2,
        color=PURPLE,
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": PURPLE, "alpha": 0.92},
        zorder=10,
    )


def figure_orthogonal_complement_constraints() -> Figure:
    """Show a homogeneous line and a parallel affine line in the x3=0 slice."""
    assert_storyboard_numbers()
    fig, axes = geometry_panels(
        ("① 齐次约束：过原点", "② 非齐次约束：平移后"),
        xlim=(-3.2, 4.0),
        ylim=(-1.2, 4.1),
    )
    zero = np.zeros(2)
    line_t = np.linspace(-1.45, 1.9, 100)
    first_line = line_t[:, None] * n[:2]
    second_line = p[:2][None, :] + line_t[:, None] * n[:2]

    axes[0].plot(first_line[:, 0], first_line[:, 1], color=ORANGE, lw=2.6, zorder=3)
    draw_vector_2d(axes[0], zero, n[:2], ORANGE, None, linewidth=2.8)
    draw_vector_2d(axes[0], zero, r[:2], BLUE, None, linewidth=2.4)
    right_angle(axes[0], zero, n[:2], r[:2])
    geometry_label(axes[0], n[:2], r"$n=(-2,1)$", ORANGE, (-24, 9))
    geometry_label(axes[0], 0.64 * r[:2], r"$r=(1,2)$", BLUE, (5, 4))
    geometry_note(axes[0], r"$x_1+2x_2=0$：垂直于 $r$" + "\n" + r"完整零空间 $\operatorname{span}(n,e_3)$ 是平面")
    draw_x3_free_marker(axes[0])

    axes[1].plot(second_line[:, 0], second_line[:, 1], color=ORANGE, lw=2.6, zorder=3)
    draw_vector_2d(axes[1], zero, p[:2], GREEN, None, linewidth=2.7)
    draw_vector_2d(axes[1], p[:2], np.array([-1.0, 3.0]), ORANGE, None, linewidth=2.4)
    right_angle(axes[1], p[:2], -p[:2], n[:2])
    axes[1].scatter(*p[:2], color=GREEN, s=42, zorder=8)
    axes[1].scatter(-1.0, 3.0, color=ORANGE, s=42, zorder=8)
    geometry_label(axes[1], p[:2], r"$p=(1,2)$", GREEN, (-5, -45))
    geometry_label(axes[1], np.array([-1.0, 3.0]), r"$q=(-1,3)=p+n$", ORANGE, (-50, -28))
    geometry_label(axes[1], np.array([0.0, 2.5]), r"$n=(-2,1)$", ORANGE, (7, 0))
    geometry_note(axes[1], r"$x_1+2x_2=5$：与上一帧平行" + "\n" + r"完整解平面 $p+\operatorname{span}(n,e_3)$")
    draw_x3_free_marker(axes[1])
    for ax in axes:
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
    fig.suptitle(r"固定 $x_3=0$ 看切片：约束改变，正交方向不变", fontsize=13, y=0.97)
    fig.text(0.5, 0.02, r"两帧同尺度，每格 1；图中向量省略第三坐标 0", ha="center", fontsize=10, color=SLATE)
    return fig



def _diagram_arrow(ax: Axes, start: tuple[float, float], end: tuple[float, float], color: str, label: str, label_y_offset: float = 0.16) -> None:
    ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "-|>", "color": color, "lw": 2.1, "mutation_scale": 14})
    mid_x = (start[0] + end[0]) / 2
    mid_y = (start[1] + end[1]) / 2 + label_y_offset
    ax.text(mid_x, mid_y, label, ha="center", va="center", fontsize=10.2, color=color, weight="bold", bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": "none", "alpha": 0.9})


def figure_four_subspaces_input_output() -> Figure:
    """Diagram the four subspaces without implying geometric scale."""
    assert_storyboard_numbers()
    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.07, top=0.94)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.5)
    ax.axis("off")
    ax.text(2.3, 6.1, r"输入 $\mathbb{R}^3$：$1+2=3$", ha="center", fontsize=13, color=SLATE)
    ax.text(9.7, 6.1, r"输出 $\mathbb{R}^2$：$1+1=2$", ha="center", fontsize=13, color=SLATE)
    ax.text(6, 6.1, "A = [[1, 2, 0], [2, 4, 0]]", ha="center", fontsize=10, color=SLATE)

    ax.text(2.3, 5.15, "行空间 · 维 1", ha="center", fontsize=12, color=BLUE)
    ax.text(2.3, 4.7, r"$\operatorname{Row}(A)=\operatorname{span}(r)$", ha="center", fontsize=11, color=BLUE)
    ax.text(2.3, 4.25, r"$r=(1,2,0)$", ha="center", fontsize=11, color=BLUE)
    ax.text(9.7, 5.15, "列空间 · 维 1（所有可达输出）", ha="center", fontsize=12, color=GREEN)
    ax.text(9.7, 4.7, r"$\operatorname{Col}(A)=\operatorname{span}(c)$", ha="center", fontsize=11, color=GREEN)
    ax.text(9.7, 4.25, r"$c=(1,2)$", ha="center", fontsize=11, color=GREEN)
    _diagram_arrow(ax, (4.35, 4.85), (7.65, 4.85), BLUE, r"$A|_{\operatorname{Row}(A)}$ 双射", 0.42)
    ax.text(6, 4.4, r"$\alpha r\mapsto 5\alpha c$", ha="center", fontsize=11, color=SLATE)

    ax.text(2.3, 3.6, r"$\perp$", ha="center", fontsize=20, color=SLATE)
    ax.text(9.7, 3.6, r"$\perp$", ha="center", fontsize=20, color=SLATE)
    ax.text(2.3, 3.05, "零空间 · 维 2", ha="center", fontsize=12, color=ORANGE)
    ax.text(2.3, 2.6, r"$\operatorname{Null}(A)=\operatorname{span}(n,e_3)$", ha="center", fontsize=11, color=ORANGE)
    ax.text(2.3, 2.15, r"$n=(-2,1,0),\ e_3=(0,0,1)$", ha="center", fontsize=10.5, color=ORANGE)
    _diagram_arrow(ax, (4.35, 2.8), (6.7, 2.8), ORANGE, r"$A(tn+se_3)$", 0.35)
    ax.scatter(7.0, 2.8, s=26, color=SLATE)
    ax.text(7.0, 2.3, r"$0\in\operatorname{Col}(A)$", ha="center", fontsize=10, color=SLATE)
    ax.text(9.7, 3.05, "左零空间 · 维 1", ha="center", fontsize=12, color=PURPLE)
    ax.text(9.7, 2.6, r"$\operatorname{Null}(A^{\mathsf{T}})=\operatorname{span}(h)$", ha="center", fontsize=10.5, color=PURPLE)
    ax.text(9.7, 2.15, r"$h=(-2,1)$", ha="center", fontsize=11, color=PURPLE)
    ax.text(6, 1.25, r"左零：$h^{\mathsf{T}}(Ax)=0$，与所有可达输出正交", ha="center", fontsize=11, color=PURPLE)
    ax.text(6, 0.78, r"它在输出侧，不是被 $A$ 消掉的输入方向", ha="center", fontsize=11, color=SLATE)
    ax.text(6, 0.1, "结构示意，非比例几何图；箭头只表示 A 的作用", ha="center", fontsize=10, color=SLATE)
    return fig


def _draw_col_geometry(ax: Axes, rhs: np.ndarray, *, label: str, color: str) -> None:
    t = np.linspace(-2, 8, 120)
    line = t[:, None] * c[None, :]
    ax.plot(line[:, 0], line[:, 1], color=GREEN, lw=2.4, zorder=2)
    draw_vector_2d(ax, np.zeros(2), c, GREEN, None, linewidth=2.0)
    draw_vector_2d(ax, np.zeros(2), h, PURPLE, None, linewidth=2.1)
    right_angle(ax, np.zeros(2), c, h)
    ax.scatter(*rhs, color=color, s=52, zorder=8)
    geometry_label(ax, rhs, label, color, (-82, 18))
    ax.text(0.06, 0.94, r"$\operatorname{Col}(A)$", transform=ax.transAxes, color=GREEN, fontsize=10)
    ax.text(0.06, 0.88, r"$y_2=2y_1$", transform=ax.transAxes, color=GREEN, fontsize=10)
    geometry_label(ax, np.array([-1.15, 0.55]), r"$h=(-2,1)$", PURPLE, (-4, -17))


def figure_left_null_inconsistency() -> Figure:
    """Contrast a reachable and an unreachable output, with a separate zoom."""
    assert_storyboard_numbers()
    fig, axes = geometry_panels(
        ("① 可达：内积为 0", "② 不可达：内积为 −1", "局部放大：差一小步"),
        xlim=(-3.2, 7.0),
        ylim=(-2.0, 14.0),
    )
    fig.set_size_inches(10.8, 6.4)
    fig.subplots_adjust(top=0.82, bottom=0.25)
    for ax in axes[:2]:
        ax.set_xlabel(r"$y_1$")
        ax.set_ylabel(r"$y_2$")
        ax.set_yticks(np.arange(-2, 15, 1))
    good = np.array([5.0, 10.0])
    bad = np.array([5.0, 9.0])
    _draw_col_geometry(axes[0], good, label=r"$b_{good}=(5,10)$", color=BLUE)
    _draw_col_geometry(axes[1], bad, label=r"$b_{bad}=(5,9)$", color=ORANGE)
    geometry_note(axes[0], r"$h^{\mathsf{T}}b_{good}=0$" + "\n" + "在列空间上，可解")
    geometry_note(axes[1], r"$h^{\mathsf{T}}b_{bad}=-1$" + "\n" + "偏离列空间，不可解")

    axes[2].set_xlim(4.35, 5.65)
    axes[2].set_ylim(8.35, 10.65)
    axes[2].set_xticks(np.arange(4.5, 5.6, 0.5))
    axes[2].set_yticks(np.arange(8.5, 10.6, 0.5))
    axes[2].plot([4.35, 5.65], [8.7, 11.3], color=GREEN, lw=2.2, zorder=2)
    axes[2].scatter(*good, color=BLUE, s=52, zorder=8)
    axes[2].scatter(*bad, color=ORANGE, s=52, zorder=8)
    axes[2].plot([5.0, 5.0], [9.0, 10.0], color=SLATE, lw=1.3, linestyle="--", zorder=3)
    geometry_label(axes[2], good, r"$b_{good}$", BLUE, (7, 5))
    geometry_label(axes[2], bad, r"$b_{bad}$", ORANGE, (7, -17))
    axes[2].text(0.5, 0.08, "局部坐标刻度：每格 0.5", transform=axes[2].transAxes, ha="center", va="bottom", fontsize=9.5, color=SLATE, bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": GRID})
    geometry_note(axes[2], r"$b_{bad}-b_{good}=(0,-1)$" + "\n" + "纵坐标只差 1，却已不可达")
    fig.text(0.5, 0.03, "左侧两帧同尺度，每格 1；右侧单独放大，每格 0.5", ha="center", fontsize=10, color=SLATE)
    fig.suptitle("左零向量用一个内积检测输出是否可达", fontsize=13, y=0.95)
    return fig


def figure_orthogonal_affine_solutions() -> Figure:
    """Show three equal-scale slices of the affine solution plane."""
    assert_storyboard_numbers()
    fig, axes = geometry_panels(
        (r"① $t=-1$：$p-n$", r"② $t=0$：最近点 $p$", r"③ $t=1$：$p+n$"),
        xlim=(-3.2, 4.0),
        ylim=(-1.2, 4.1),
    )
    fig.set_size_inches(11.4, 5.5)
    fig.subplots_adjust(top=0.84, bottom=0.32)
    zero = np.zeros(2)
    line_t = np.linspace(-1.45, 1.9, 100)
    affine_line = p[:2][None, :] + line_t[:, None] * n[:2]
    for ax, t in zip(axes, (-1.0, 0.0, 1.0)):
        x_t = p[:2] + t * n[:2]
        ax.plot(affine_line[:, 0], affine_line[:, 1], color=ORANGE, lw=2.5, zorder=3)
        if t != 0:
            draw_vector_2d(ax, zero, p[:2], GREEN, None, linewidth=1.4, alpha=0.5)
            ax.scatter(*p[:2], color=GREEN, s=20, zorder=8)
        draw_vector_2d(ax, zero, x_t, GREEN if t == 0 else BLUE, None, linewidth=2.7)
        if t != 0:
            draw_vector_2d(ax, p[:2], x_t, ORANGE, None, linewidth=2.2)
        else:
            right_angle(ax, p[:2], -p[:2], n[:2])
        draw_x3_free_marker(ax, text_y=0.97)
        ax.scatter(*x_t, color=GREEN if t == 0 else BLUE, s=48, zorder=8)
        geometry_label(
            ax, x_t,
            r"$p=(1,2)$" if t == 0 else (r"$q=(3,1)$" if t < 0 else r"$q=(-1,3)$"),
            GREEN if t == 0 else BLUE,
            (5, -40) if t == 0 else ((-50, -28) if t < 0 else (-22, 16)),
        )
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        geometry_label(ax, np.array([-0.05, 2.52]), r"$p+t n$", ORANGE, (6, 0))
        geometry_note(
            ax,
            rf"$\|q\|^2={5 + 5 * t * t:g}$" + "\n" +
            (r"$p\perp n$，$p\perp e_3$：最近" if t == 0 else "离开垂足，长度增加"),
        )
    fig.text(0.5, 0.11, r"完整解平面：$x=p+tn+se_3$，$\|x\|^2=5+5t^2+s^2$；仅 $t=s=0$ 最小", ha="center", fontsize=11, color=SLATE)
    fig.text(0.5, 0.04, r"三帧均为 $s=x_3=0$ 切片，$q=p+tn$，每格 1；$s$ 自由出画面，完整解集不只是一条线", ha="center", fontsize=10, color=SLATE)
    fig.suptitle("沿同一解集移动：垂足 p 是距离原点最近的解", fontsize=13, y=0.95)
    return fig


FIGURES = {
    "orthogonal-complement-constraints": figure_orthogonal_complement_constraints,
    "four-subspaces-input-output": figure_four_subspaces_input_output,
    "left-null-inconsistency": figure_left_null_inconsistency,
    "orthogonal-affine-solutions": figure_orthogonal_affine_solutions,
}



def save_figure(figure: Figure, stem: str, output_dir: Path, formats: tuple[str, ...]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for file_format in formats:
        target = output_dir / f"{stem}.{file_format}"
        figure.savefig(
            target,
            dpi=180,
            bbox_inches="tight",
            metadata={"Creator": "Matplotlib", "Date": "2026-09-25"},
        )
        if file_format == "svg":
            normalize_svg(target)
        print(f"generated {target.relative_to(ROOT) if target.is_relative_to(ROOT) else target}")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("svg", "png", "both"), default="svg", help="output format (default: svg)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="directory for generated files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    font_manager.findfont("Noto Sans CJK SC", fallback_to_default=False)
    assert_storyboard_numbers()
    formats = ("svg", "png") if args.format == "both" else (args.format,)
    with mpl.rc_context(CHINESE_FIGURE_STYLE):
        for stem, factory in FIGURES.items():
            save_figure(factory(), stem, args.output_dir, formats)


if __name__ == "__main__":
    main()