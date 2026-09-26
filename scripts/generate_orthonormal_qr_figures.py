#!/usr/bin/env python3
"""Generate the Gram--Schmidt and QR factorization teaching storyboards."""

from __future__ import annotations

import argparse
import html
from collections.abc import Callable
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
    normalize_svg,
    right_angle,
    style_plane,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "images" / "linear-algebra"

FONT_SIZE = 15
TITLE_SIZE = 17


# ---------------------------------------------------------------------------
# Shared drawing helpers


def vector_label(ax: Axes, point: np.ndarray, text: str, color: str, offset: tuple[float, float] = (7, 7)) -> None:
    """Place a readable label without changing the geometry of the arrow."""
    ax.annotate(
        text,
        point,
        xytext=offset,
        textcoords="offset points",
        fontsize=FONT_SIZE,
        color=color,
        weight="bold",
        zorder=8,
    )


def plane(ax: Axes, title: str, *, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    """Use the existing project plane styling, with the storyboard font size."""
    style_plane(ax, xlim=xlim, ylim=ylim, title=title)
    ax.set_facecolor("white")
    ax.set_title(title, fontsize=TITLE_SIZE, pad=12, weight="bold")
    ax.set_xticks(np.arange(np.ceil(xlim[0]), xlim[1], 1))
    ax.set_yticks(np.arange(np.ceil(ylim[0]), ylim[1], 1))
    ax.tick_params(labelsize=FONT_SIZE)
    ax.set_xlabel("$x_1$", fontsize=FONT_SIZE, labelpad=5)
    ax.set_ylabel("$x_2$", fontsize=FONT_SIZE, labelpad=5)


def note(ax: Axes, text: str, *, y: float = -0.20, color: str = SLATE) -> None:
    ax.text(
        0.5,
        y,
        text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=FONT_SIZE,
        color=color,
        linespacing=1.45,
    )


def arrow_2d(
    ax: Axes,
    start: np.ndarray,
    end: np.ndarray,
    color: str,
    label: str | None = None,
    *,
    linewidth: float = 3.0,
    linestyle: str = "-",
    label_offset: tuple[float, float] = (7, 7),
) -> None:
    draw_vector_2d(
        ax,
        start,
        end,
        color,
        None,
        linewidth=linewidth,
        linestyle=linestyle,
    )
    # The shared helper uses point padding; remove it for exact vector lengths.
    ax.texts[-1].arrow_patch.shrinkA = 0
    ax.texts[-1].arrow_patch.shrinkB = 0
    if label is not None:
        vector_label(ax, end, label, color, label_offset)


def add_svg_metadata(path: Path, title: str, description: str) -> None:
    """Normalize an SVG and add accessible, searchable document metadata."""
    normalize_svg(path)
    content = path.read_text(encoding="utf-8")
    svg_start = content.index("<svg")
    opening_end = content.index(">", svg_start) + 1
    metadata = (
        f"\n<title>{html.escape(title)}</title>"
        f"\n<desc>{html.escape(description)}</desc>"
    )
    content = content[:opening_end] + metadata + content[opening_end:]
    path.write_text(content, encoding="utf-8")


def save_figure(
    figure: Figure,
    stem: str,
    output_dir: Path,
    formats: tuple[str, ...],
    *,
    title: str,
    description: str,
) -> None:
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
            add_svg_metadata(target, title, description)
        print(f"generated {target.relative_to(ROOT) if target.is_relative_to(ROOT) else target}")
    plt.close(figure)


# ---------------------------------------------------------------------------
# Numerical storyboards


def assert_storyboard_numbers() -> None:
    """Keep the drawings tied to the exact examples shown in their labels."""
    a1 = np.array([2.0, 1.0])
    q1 = a1 / np.sqrt(5.0)
    a2 = np.array([1.0, 3.0])
    r12 = q1 @ a2
    u2 = a2 - r12 * q1
    q2 = u2 / np.linalg.norm(u2)
    assert np.isclose(np.linalg.norm(a1), np.sqrt(5.0))
    assert np.isclose(np.linalg.norm(q1), 1.0)
    assert np.allclose(r12 * q1, a1)
    assert np.allclose(u2, [-1.0, 2.0])
    assert np.isclose(q1 @ u2, 0.0)
    assert np.allclose(a2, r12 * q1 + np.linalg.norm(u2) * q2)
    q = np.column_stack((q1, q2))
    r = np.sqrt(5.0) * np.array([[1.0, 1.0], [0.0, 1.0]])
    assert np.allclose(q.T @ q, np.eye(2))
    assert np.allclose(q @ r, np.column_stack((a1, a2)))
    assert np.allclose(r, np.triu(r))

    vectors = np.array([[2.0, 1.0, 0.0], [1.0, 3.0, 0.0], [1.0, 1.0, 2.0]])
    q1_3 = vectors[0] / np.linalg.norm(vectors[0])
    u2_3 = vectors[1] - (q1_3 @ vectors[1]) * q1_3
    q2_3 = u2_3 / np.linalg.norm(u2_3)
    u3_3 = vectors[2] - (q1_3 @ vectors[2]) * q1_3 - (q2_3 @ vectors[2]) * q2_3
    assert np.allclose(vectors[2] - (q1_3 @ vectors[2]) * q1_3, [-0.2, 0.4, 2.0])
    assert np.allclose([q1_3 @ vectors[2], q2_3 @ vectors[2]], [3 / np.sqrt(5), 1 / np.sqrt(5)])
    assert np.allclose(u3_3, [0.0, 0.0, 2.0])
    assert np.allclose(u3_3 / np.linalg.norm(u3_3), [0, 0, 1])
    assert np.isclose(q1_3 @ q2_3, 0.0)
    assert np.isclose(q1_3 @ u3_3, 0.0) and np.isclose(q2_3 @ u3_3, 0.0)

    near_a1 = np.array([1.0, 1.0])
    near_a2 = np.array([1.0, 1.001])
    assert np.allclose(near_a2 - near_a1, [0.0, 0.001])
    # If c1*a1 + c2*a2 = 0, subtracting the first component from the second
    # gives 0.001*c2 = 0; then the first component gives c1 = 0.
    assert near_a1[1] - near_a1[0] == 0.0
    assert np.isclose(near_a2[1] - near_a2[0], 0.001)
    assert near_a2[1] - near_a2[0] != 0.0
    assert near_a1[0] != 0.0
    near_q1 = near_a1 / np.linalg.norm(near_a1)
    near_u2 = near_a2 - (near_q1 @ near_a2) * near_q1
    assert np.isclose(near_q1 @ near_u2, 0.0, atol=1e-14)
    assert np.allclose(near_u2 + (near_q1 @ near_a2) * near_q1, near_a2)
    assert np.allclose(near_u2, [-0.0005, 0.0005])
    assert np.isclose(np.linalg.norm(near_u2), 0.001 / np.sqrt(2.0))
    assert np.linalg.norm(near_u2) > 0.0


def figure_orthogonalize_first_vector() -> Figure:
    a1 = np.array([2.0, 1.0])
    q1 = a1 / np.sqrt(5.0)
    zero = np.zeros(2)
    perpendicular = np.array([-1.0, 2.0]) / np.sqrt(5.0)

    # Fixed margins keep equal-aspect axes and out-of-axes notes independent
    # of prior draws and export order (constrained_layout iterates on each draw).
    fig, ax = plt.subplots(figsize=(8.2, 7.0))
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.22, top=0.86)
    plane(ax, "① 先把第一列单位化", xlim=(-0.45, 2.75), ylim=(-0.45, 2.05))
    arrow_2d(ax, zero, a1, BLUE, r"$a_1=(2,1)$", label_offset=(8, 8), linewidth=3.5)
    arrow_2d(ax, zero, q1, GREEN, r"$q_1=a_1/\sqrt{5}$", label_offset=(8, -24), linewidth=3.0)

    # A parallel dimension line makes the length claim visible without changing scale.
    offset = 0.22 * perpendicular
    ax.annotate(
        "",
        xy=a1 + offset,
        xytext=offset,
        arrowprops={"arrowstyle": "<->", "color": SLATE, "lw": 1.8},
        zorder=4,
    )
    ax.text(
        *(0.52 * a1 + 0.38 * perpendicular),
        r"$\|a_1\|=\sqrt{5}$",
        fontsize=FONT_SIZE,
        color=SLATE,
        ha="center",
        va="center",
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "none", "alpha": 0.88},
    )
    ax.scatter(*zero, color=SLATE, s=35, zorder=9)
    note(ax, r"同一坐标尺度：$\|q_1\|=1$，方向不变，只除以 $\sqrt{5}$。", y=-0.18)
    fig.suptitle("Gram–Schmidt：第一列先成为单位向量", fontsize=TITLE_SIZE + 1, weight="bold")
    return fig


def figure_gram_schmidt_second_vector() -> Figure:
    a1 = np.array([2.0, 1.0])
    q1 = a1 / np.sqrt(5.0)
    a2 = np.array([1.0, 3.0])
    r12 = q1 @ a2
    projection = r12 * q1
    u2 = a2 - projection
    q2 = u2 / np.linalg.norm(u2)
    zero = np.zeros(2)

    fig, ax = plt.subplots(figsize=(9.4, 7.0))
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.22, top=0.86)
    plane(ax, r"② 从第二列减去 $q_1$ 方向的投影", xlim=(-1.25, 2.8), ylim=(-0.55, 3.55))
    arrow_2d(ax, zero, a2, ORANGE, r"$a_2=(1,3)$", linewidth=3.5, label_offset=(8, 8))
    arrow_2d(ax, zero, q1, BLUE, r"$q_1$", linewidth=2.8, label_offset=(-34, -5))
    arrow_2d(
        ax,
        zero,
        projection,
        GREEN,
        r"投影 $r_{12}q_1$",
        linewidth=2.8,
        linestyle="--",
        label_offset=(-35, -30),
    )
    arrow_2d(
        ax,
        projection,
        a2,
        PURPLE,
        None,
        linewidth=3.3,
        label_offset=(8, 2),
    )
    arrow_2d(ax, zero, q2, PURPLE, r"$q_2=u_2/\sqrt{5}$", linewidth=2.5, label_offset=(-66, 10))
    vector_label(ax, (projection + a2) / 2, r"$u_2=(-1,2)$", PURPLE, (12, 5))
    right_angle(ax, projection, -q1, u2)
    ax.scatter(*projection, color=GREEN, s=42, zorder=9)
    ax.scatter(*zero, color=SLATE, s=35, zorder=9)
    ax.text(
        0.98,
        0.04,
        r"$a_2=r_{12}q_1+r_{22}q_2$"
        "\n"
        r"$\qquad=\sqrt{5}\,q_1+\sqrt{5}\,q_2$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=FONT_SIZE,
        color=SLATE,
        bbox={"boxstyle": "round,pad=0.38", "facecolor": "white", "edgecolor": GRID, "alpha": 0.95},
    )
    note(ax, r"$u_2=a_2-\operatorname{proj}_{q_1}a_2$，$u_2\perp q_1$", y=-0.18)
    fig.suptitle("第二列：投影留下正交残余", fontsize=TITLE_SIZE + 1, weight="bold")
    return fig


# ---------------------------------------------------------------------------
# Three-dimensional storyboard


def style_3d(ax: Axes, title: str) -> None:
    ax.set_title(title, fontsize=TITLE_SIZE, pad=14, weight="bold")
    ax.set_facecolor("white")
    ax.view_init(elev=25, azim=-110)
    ax.set_proj_type("ortho")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1, 1, 1, 0))
        axis.set_ticks([0, 1, 2])
    ax.set_xlim(-1.25, 2.35)
    ax.set_ylim(-0.85, 2.35)
    ax.set_zlim(-0.25, 2.55)
    ax.set_box_aspect((3.6, 3.2, 2.8))
    ax.set_xlabel("$x_1$", fontsize=FONT_SIZE, labelpad=8)
    ax.set_ylabel("$x_2$", fontsize=FONT_SIZE, labelpad=8)
    ax.set_zlabel("$x_3$", fontsize=FONT_SIZE, labelpad=8)
    ax.tick_params(labelsize=FONT_SIZE)
    ax.grid(False)


def arrow_3d(
    ax: Axes,
    start: np.ndarray,
    end: np.ndarray,
    color: str,
    label: str,
    *,
    linestyle: str = "-",
    alpha: float = 1.0,
    label_offset: tuple[float, float, float] = (0.06, 0.06, 0.06),
) -> None:
    delta = end - start
    ax.quiver(
        *start,
        *delta,
        color=color,
        linewidth=2.8,
        alpha=alpha,
        linestyle=linestyle,
        arrow_length_ratio=0.10,
        zorder=5,
    )
    label_point = end + np.asarray(label_offset)
    if label:
        ax.text(*label_point, label, color=color, fontsize=FONT_SIZE, weight="bold")


def draw_x3_zero_plane(ax: Axes) -> None:
    xx, yy = np.meshgrid(np.linspace(-1.0, 2.2, 5), np.linspace(-0.55, 2.0, 5))
    ax.plot_surface(xx, yy, np.zeros_like(xx), color=GRID, alpha=0.15, linewidth=0, shade=False)
    ax.text2D(0.78, 0.53, "$x_3=0$", transform=ax.transAxes, color=SLATE, fontsize=FONT_SIZE)


def figure_gram_schmidt_three_vectors() -> Figure:
    a1 = np.array([2.0, 1.0, 0.0])
    a2 = np.array([1.0, 3.0, 0.0])
    a3 = np.array([1.0, 1.0, 2.0])
    q1 = a1 / np.sqrt(5.0)
    u2 = a2 - (q1 @ a2) * q1
    q2 = u2 / np.linalg.norm(u2)
    r13 = q1 @ a3
    first_residual = a3 - r13 * q1
    r23 = q2 @ first_residual
    u3 = first_residual - r23 * q2
    zero = np.zeros(3)

    fig = plt.figure(figsize=(15.6, 6.8))
    fig.subplots_adjust(left=0.055, right=0.96, top=0.80, bottom=0.29, wspace=0.28)
    ax1 = fig.add_subplot(1, 3, 1)
    plane(ax1, "① 先看 $x_3=0$ 投影", xlim=(-1.15, 2.6), ylim=(-0.55, 3.55))
    arrow_2d(ax1, zero[:2], a1[:2], BLUE, r"$a_1$", linewidth=3.0, label_offset=(7, 7))
    arrow_2d(ax1, zero[:2], a2[:2], ORANGE, r"$a_2$", linewidth=3.0, label_offset=(7, 7))
    arrow_2d(ax1, zero[:2], q1[:2], GREEN, r"$q_1$", linewidth=2.5, label_offset=(-30, -4))
    arrow_2d(ax1, zero[:2], q2[:2], PURPLE, r"$q_2$", linewidth=2.5, label_offset=(-37, 8))
    right_angle(ax1, zero[:2], q1[:2], q2[:2])
    ax1.scatter(1, 1, color=SLATE, s=40, zorder=8)
    vector_label(ax1, np.array([1, 1]), r"$a_3$ 的投影", SLATE, (-72, 10))

    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    style_3d(ax2, "② 先减去 $r_{13}q_1$")
    draw_x3_zero_plane(ax2)
    r13q1 = r13 * q1
    arrow_3d(ax2, zero, a3, ORANGE, r"$a_3$", label_offset=(-0.5, 0.0, 0.22))
    arrow_3d(ax2, zero, r13q1, BLUE, r"$r_{13}q_1$", linestyle="--", label_offset=(0.02, 0.02, 0.04))
    arrow_3d(ax2, r13q1, a3, PURPLE, "")
    ax2.text(*(r13q1 + 0.55 * first_residual + [0.25, 0, 0]), r"$v_1$", color=PURPLE, fontsize=FONT_SIZE)

    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    style_3d(ax3, "③ 再减去 $r_{23}q_2$")
    draw_x3_zero_plane(ax3)
    r23q2 = r23 * q2
    arrow_3d(ax3, zero, first_residual, PURPLE, r"$v_1$", alpha=0.62, label_offset=(-0.45, 0.0, 0.22))
    arrow_3d(ax3, zero, r23q2, BLUE, r"$r_{23}q_2$", linestyle="--", label_offset=(0.02, 0.02, 0.03))
    arrow_3d(ax3, r23q2, first_residual, GREEN, "")
    ax3.text(*(r23q2 + 0.72 * u3 + [0.25, 0, 0]), r"$u_3$", color=GREEN, fontsize=FONT_SIZE)
    arrow_3d(ax3, zero, u3 / np.linalg.norm(u3), GREEN, r"$q_3$", label_offset=(0.1, -0.4, 0.0))
    for x, text in (
        (0.19, r"$q_1=(2,1,0)/\sqrt{5}$" + "\n" + r"$q_2=(-1,2,0)/\sqrt{5}$"),
        (0.51, r"$v_1=a_3-\frac{3}{\sqrt{5}}q_1$" + "\n" + r"$v_1=(-1/5,2/5,2)$"),
        (0.83, r"$u_3=v_1-\frac{1}{\sqrt{5}}q_2=(0,0,2)$" + "\n" + r"$q_3=u_3/2=e_3$"),
    ):
        fig.text(x, 0.19, text, ha="center", va="top", fontsize=FONT_SIZE, linespacing=1.65, color=SLATE)
    fig.text(0.5, 0.86, r"$a_1=(2,1,0)$，$a_2=(1,3,0)$，$a_3=(1,1,2)$", ha="center", fontsize=FONT_SIZE, color=SLATE)
    fig.text(0.5, 0.025, r"首帧为平面投影；后两帧同视角、同尺度。残余箭头平移到投影端点，首尾相接。", ha="center", fontsize=FONT_SIZE, color=SLATE)
    fig.suptitle(r"三列的正交化：先减去 $q_1$ 分量，再减去 $q_2$ 分量", fontsize=TITLE_SIZE + 1, weight="bold")
    return fig


# ---------------------------------------------------------------------------
# QR column view


def draw_upper_triangular_r(ax: Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.50, 0.41, "$R$ 上三角", ha="center", va="center", fontsize=TITLE_SIZE, color=SLATE, weight="bold")
    left, bottom, width, height = 0.19, 0.10, 0.62, 0.27
    ax.add_patch(plt.Rectangle((left, bottom), width, height, facecolor="white", edgecolor=GRID, linewidth=2.0))
    ax.plot([left + width / 2, left + width / 2], [bottom, bottom + height], color=GRID, linewidth=1.2)
    ax.plot([left, left + width], [bottom + height / 2, bottom + height / 2], color=GRID, linewidth=1.2)
    ax.text(left + width * 0.25, bottom + height * 0.75, r"$\sqrt{5}$", ha="center", va="center", fontsize=FONT_SIZE, color=BLUE)
    ax.text(left + width * 0.75, bottom + height * 0.75, r"$\sqrt{5}$", ha="center", va="center", fontsize=FONT_SIZE, color=ORANGE)
    ax.text(left + width * 0.25, bottom + height * 0.25, "$0$", ha="center", va="center", fontsize=FONT_SIZE, color=SLATE)
    ax.text(left + width * 0.75, bottom + height * 0.25, r"$\sqrt{5}$", ha="center", va="center", fontsize=FONT_SIZE, color=GREEN)



def figure_qr_factorization_columns() -> Figure:
    a1 = np.array([2.0, 1.0])
    a2 = np.array([1.0, 3.0])
    q1 = a1 / np.sqrt(5.0)
    u2 = a2 - (q1 @ a2) * q1
    q2 = u2 / np.linalg.norm(u2)
    zero = np.zeros(2)

    fig = plt.figure(figsize=(15.2, 6.2))
    grid = fig.add_gridspec(1, 3, left=0.055, right=0.98, top=0.82, bottom=0.19,
                          width_ratios=(1.15, 0.95, 1.15), wspace=0.22)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_mid = fig.add_subplot(grid[0, 1])
    ax_q = fig.add_subplot(grid[0, 2])
    plane(ax_a, "A 的列向量", xlim=(-1.3, 2.9), ylim=(-0.65, 3.55))
    plane(ax_q, r"$Q=[q_1\ q_2]$", xlim=(-1.3, 2.9), ylim=(-0.65, 3.55))

    arrow_2d(ax_a, zero, a1, BLUE, r"$a_1$", linewidth=3.4, label_offset=(8, 8))
    arrow_2d(ax_a, zero, a2, ORANGE, r"$a_2$", linewidth=3.4, label_offset=(8, 8))
    ax_a.text(0.05, 0.94, r"$A=[a_1\ a_2]$", transform=ax_a.transAxes, fontsize=FONT_SIZE, color=SLATE, va="top")

    arrow_2d(ax_q, zero, q1, BLUE, r"$q_1$", linewidth=3.0, label_offset=(-34, -4))
    arrow_2d(ax_q, zero, q2, GREEN, r"$q_2$", linewidth=3.0, label_offset=(-36, 7))
    right_angle(ax_q, zero, q1, q2)
    ax_q.text(0.05, 0.94, r"$Q^{\mathsf{T}}Q=I$", transform=ax_q.transAxes, fontsize=FONT_SIZE, color=SLATE, va="top")

    ax_mid.set_xlim(0, 1)
    ax_mid.set_ylim(0, 1)
    ax_mid.axis("off")
    ax_mid.text(0.50, 0.96, r"$A=QR$", ha="center", va="top", fontsize=TITLE_SIZE + 1, color=SLATE, weight="bold")
    ax_mid.annotate("", xy=(0.96, 0.70), xytext=(0.04, 0.70), arrowprops={"arrowstyle": "-|>", "color": BLUE, "lw": 2.4, "mutation_scale": 16})
    ax_mid.text(0.50, 0.78, r"$a_1=r_{11}q_1$", ha="center", va="center", fontsize=FONT_SIZE, color=BLUE, weight="bold")
    ax_mid.annotate("", xy=(0.96, 0.46), xytext=(0.04, 0.46), arrowprops={"arrowstyle": "-|>", "color": ORANGE, "lw": 2.4, "mutation_scale": 16})
    ax_mid.text(0.50, 0.54, r"$a_2=r_{12}q_1+r_{22}q_2$", ha="center", va="center", fontsize=FONT_SIZE, color=ORANGE, weight="bold")
    draw_upper_triangular_r(ax_mid)

    fig.suptitle("QR 的列视角：用正交列重写 A", fontsize=TITLE_SIZE + 1, weight="bold")
    fig.text(0.5, 0.02, r"两侧严格同尺度；$r_{11}=\|a_1\|$，$r_{22}=\|u_2\|$；$r_{12}=q_1^{\mathsf{T}}a_2$。", ha="center", fontsize=FONT_SIZE, color=SLATE)
    return fig


# ---------------------------------------------------------------------------
# Nearly dependent example


def figure_orthogonal_vs_nearly_dependent() -> Figure:
    a1 = np.array([1.0, 1.0])
    a2 = np.array([1.0, 1.001])
    q1 = a1 / np.linalg.norm(a1)
    projection = (q1 @ a2) * q1
    u2 = a2 - projection
    zero = np.zeros(2)

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 7.3))
    fig.subplots_adjust(left=0.07, right=0.96, top=0.84, bottom=0.27, wspace=0.27)
    for ax, title in zip(axes, ("输入：两列几乎重合", "正交化：残余很小")):
        plane(ax, title, xlim=(-0.12, 1.30), ylim=(-0.12, 1.30))

    arrow_2d(axes[0], zero, a1, BLUE, r"$a_1=(1,1)$", linewidth=5.0, label_offset=(-115, 14))
    arrow_2d(axes[0], zero, a2, ORANGE, r"$a_2=(1,1.001)$", linewidth=2.0, label_offset=(-30, -28))
    axes[0].text(0.06, 0.94, "肉眼几乎看成同一条线", transform=axes[0].transAxes, fontsize=FONT_SIZE, color=SLATE, va="top")
    note(axes[0], "同一尺度；差异集中在末端的千分之一。" + "\n"
         + r"$a_2-a_1=(0,0.001)$" + "\n"
         + r"若 $c_1a_1+c_2a_2=0$，两分量相减得 $0.001c_2=0$" + "\n"
         + "残余非零，两列独立", y=-0.18)

    arrow_2d(axes[1], zero, q1, BLUE, r"$q_1$", linewidth=4.5, label_offset=(-35, 0))
    arrow_2d(axes[1], zero, a2, ORANGE, r"$a_2$", linewidth=3.0, label_offset=(7, 7))
    arrow_2d(axes[1], zero, projection, GREEN, r"$\operatorname{proj}_{q_1}a_2$", linewidth=2.5, linestyle="--", label_offset=(-96, -22))
    axes[1].plot(*np.array([projection, a2]).T, color=PURPLE, linewidth=1.0)
    note(axes[1], r"$u_2=a_2-(q_1^{\mathsf{T}}a_2)q_1$" + "\n"
         + r"$u_2=(-0.0005,0.0005)$，$u_2\perp q_1$" + "\n"
         + r"长度 $\|u_2\|=0.001/\sqrt{2}\approx7.07\times10^{-4}$", y=-0.18)

    # The main panels remain honest about scale; this inset only makes the tiny endpoint gap visible.
    inset = axes[1].inset_axes([0.02, 0.52, 0.43, 0.32])
    inset.set_facecolor("white")
    inset.set_aspect("equal")
    inset.set_title("末端放大示意", fontsize=FONT_SIZE, pad=5)
    scale = 1000.0
    projection_delta = scale * (projection - np.array([1.0, 1.0]))
    a2_delta = scale * (a2 - np.array([1.0, 1.0]))
    inset.annotate("", xy=a2_delta, xytext=projection_delta, arrowprops={"arrowstyle": "-|>", "color": PURPLE, "lw": 2.4, "mutation_scale": 13})
    inset.scatter(*projection_delta, color=GREEN, s=28, zorder=6)
    inset.scatter(*a2_delta, color=ORANGE, s=28, zorder=6)
    inset.text(projection_delta[0] - 0.25, projection_delta[1] - 0.35, "投影", fontsize=FONT_SIZE, color=GREEN)
    inset.text(a2_delta[0] + 0.06, a2_delta[1] + 0.08, "$a_2$", fontsize=FONT_SIZE, color=ORANGE)
    inset.plot([-0.15, 0.5], [-0.15, 0.5], color=GREEN, linestyle="--", lw=1.4)
    right_angle(inset, projection_delta, -q1, u2)
    inset.set_xlim(-0.35, 1.15)
    inset.set_ylim(-0.25, 1.25)
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_xlabel(r"$1000\,(x-(1,1))$", fontsize=FONT_SIZE, labelpad=3)
    inset.tick_params(labelsize=FONT_SIZE)
    inset.grid(color=GRID, linewidth=0.8)

    fig.suptitle("两列几乎重合：减去投影后，只剩很小的正交残余", fontsize=TITLE_SIZE + 1, weight="bold")
    return fig


FIGURES: dict[str, tuple[Callable[[], Figure], str, str]] = {
    "orthogonalize-first-vector": (
        figure_orthogonalize_first_vector,
        "第一列单位化",
        "在同一坐标尺度中比较 a1=(2,1) 与 q1=a1/sqrt(5)，并标出 |a1|=sqrt(5)。",
    ),
    "gram-schmidt-second-vector": (
        figure_gram_schmidt_second_vector,
        "第二列的 Gram–Schmidt 投影",
        "展示 a2=(1,3) 分解为 q1 方向投影 r12 q1 与正交残余 u2=(-1,2)，并标出直角。",
    ),
    "gram-schmidt-three-vectors": (
        figure_gram_schmidt_three_vectors,
        "三向量的逐步正交化",
        "用 x3=0 投影和两帧简洁三维图展示先减去 r13 q1，再减去 r23 q2，得到 u3=(0,0,2)。",
    ),
    "qr-factorization-columns": (
        figure_qr_factorization_columns,
        "QR 分解的列视角",
        "同尺度比较 A 的列与 Q 的正交列，并展示上三角 R 及 a1=r11 q1、a2=r12 q1+r22 q2。",
    ),
    "orthogonal-vs-nearly-dependent": (
        figure_orthogonal_vs_nearly_dependent,
        "近相关列的正交残余",
        "比较 a1=(1,1)、a2=(1,1.001) 的几乎重合输入与很小的正交残余；局部放大只用于看清几何差异。",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--format",
        choices=("svg", "png", "both"),
        default="svg",
        help="output format (default: svg)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory for generated files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    font_manager.findfont("Noto Sans CJK SC", fallback_to_default=False)
    assert_storyboard_numbers()
    formats = ("svg", "png") if args.format == "both" else (args.format,)
    with mpl.rc_context(CHINESE_FIGURE_STYLE):
        for stem, (factory, title, description) in FIGURES.items():
            save_figure(
                factory(),
                stem,
                args.output_dir,
                formats,
                title=title,
                description=description,
            )


if __name__ == "__main__":
    main()
