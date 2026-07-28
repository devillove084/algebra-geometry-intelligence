#!/usr/bin/env python3
"""Generate the geometric figures used by the linear algebra chapters."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "images" / "linear-algebra"

BLUE = "#2563eb"
ORANGE = "#ea580c"
GREEN = "#059669"
SLATE = "#334155"
GRID = "#cbd5e1"
BACKGROUND = "#f8fafc"


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.sans-serif": ["DejaVu Sans"],
            "axes.unicode_minus": False,
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
            "axes.facecolor": BACKGROUND,
            "svg.fonttype": "path",
            "svg.hashsalt": "algebra-geometry-intelligence",
        }
    )


def style_plane(
    ax: Axes,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    title: str,
    grid: bool = True,
) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, pad=12, fontsize=13)
    if grid:
        ax.grid(color=GRID, linewidth=0.8, alpha=0.8)
    ax.axhline(0, color="#64748b", linewidth=1.3)
    ax.axvline(0, color="#64748b", linewidth=1.3)
    ax.tick_params(colors="#64748b", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID)


def arrow(ax: Axes, start: np.ndarray, end: np.ndarray, color: str, label: str) -> None:
    delta = end - start
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={"arrowstyle": "-|>", "color": color, "lw": 2.8, "mutation_scale": 15},
        zorder=5,
    )
    midpoint = start + 0.55 * delta
    ax.annotate(
        label,
        midpoint,
        xytext=(5, 6),
        textcoords="offset points",
        color=color,
        fontsize=11,
        weight="bold",
        zorder=6,
    )


def transformed_grid(ax: Axes, matrix: np.ndarray, shift: np.ndarray | None = None) -> None:
    shift = np.zeros(2) if shift is None else shift
    t = np.linspace(-2.6, 2.6, 160)
    for constant in np.arange(-2.0, 2.01, 0.5):
        vertical = matrix @ np.vstack((np.full_like(t, constant), t)) + shift[:, None]
        horizontal = matrix @ np.vstack((t, np.full_like(t, constant))) + shift[:, None]
        ax.plot(vertical[0], vertical[1], color=GRID, linewidth=0.75, zorder=1)
        ax.plot(horizontal[0], horizontal[1], color=GRID, linewidth=0.75, zorder=1)


def figure_basis_transformation() -> Figure:
    matrix = np.array([[2.0, 1.0], [-1.0, 1.0]])
    e1, e2 = np.eye(2)
    a1, a2 = matrix @ e1, matrix @ e2

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), constrained_layout=True)
    style_plane(axes[0], xlim=(-2.6, 2.6), ylim=(-2.6, 2.6), title="Before: standard coordinates")
    arrow(axes[0], np.zeros(2), e1, BLUE, r"$\mathbf{e}_1$")
    arrow(axes[0], np.zeros(2), e2, ORANGE, r"$\mathbf{e}_2$")
    axes[0].scatter(0, 0, color=SLATE, s=24, zorder=7)

    style_plane(
        axes[1], xlim=(-4.2, 4.2), ylim=(-3.2, 3.2), title="After: columns of the matrix", grid=False
    )
    transformed_grid(axes[1], matrix)
    arrow(axes[1], np.zeros(2), a1, BLUE, r"$\mathbf{a}_1=A\mathbf{e}_1$")
    arrow(axes[1], np.zeros(2), a2, ORANGE, r"$\mathbf{a}_2=A\mathbf{e}_2$")
    axes[1].scatter(0, 0, color=SLATE, s=24, zorder=7)

    fig.suptitle("A matrix is determined by where the basis vectors go", fontsize=16, weight="bold")
    return fig


def figure_column_combination() -> Figure:
    matrix = np.array([[2.0, 1.0], [-1.0, 1.0]])
    coefficients = np.array([3.0, 2.0])
    a1, a2 = matrix[:, 0], matrix[:, 1]
    first = coefficients[0] * a1
    second = coefficients[1] * a2
    result = first + second

    fig, ax = plt.subplots(figsize=(9.5, 5.1), constrained_layout=True)
    style_plane(ax, xlim=(-1.0, 9.2), ylim=(-4.2, 3.2), title="Combine the columns head to tail")

    arrow(ax, np.zeros(2), a1, BLUE, r"$\mathbf{a}_1$")
    arrow(ax, np.zeros(2), a2, ORANGE, r"$\mathbf{a}_2$")
    arrow(ax, np.zeros(2), first, BLUE, r"$3\mathbf{a}_1$")
    arrow(ax, first, result, ORANGE, r"$2\mathbf{a}_2$")
    arrow(ax, np.zeros(2), result, GREEN, r"$\mathbf{y}$")

    ax.plot(
        [0, second[0], result[0], first[0], 0],
        [0, second[1], result[1], first[1], 0],
        color="#94a3b8",
        linestyle="--",
        linewidth=1.3,
        zorder=2,
    )
    ax.scatter(0, 0, color=SLATE, s=24, zorder=7)
    ax.text(
        0.98,
        0.06,
        r"$A\mathbf{x}=x_1\mathbf{a}_1+x_2\mathbf{a}_2$"
        "\n"
        r"$=3\mathbf{a}_1+2\mathbf{a}_2=[8,-1]^T$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=12,
        color=SLATE,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "edgecolor": GRID},
    )
    fig.suptitle("Matrix-vector multiplication is a column combination", fontsize=16, weight="bold")
    return fig


def figure_linear_vs_affine() -> Figure:
    matrix = np.array([[1.25, 0.45], [-0.25, 1.0]])
    bias = np.array([1.2, 0.8])
    zero = np.zeros(2)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), constrained_layout=True)
    for ax, shift, title in (
        (axes[0], zero, r"Linear map: $T(\mathbf{x})=A\mathbf{x}$"),
        (axes[1], bias, r"Affine map: $F(\mathbf{x})=A\mathbf{x}+\mathbf{b}$"),
    ):
        style_plane(ax, xlim=(-3.6, 4.0), ylim=(-3.2, 3.8), title=title, grid=False)
        transformed_grid(ax, matrix, shift)
        ax.scatter(0, 0, color=SLATE, s=34, zorder=7, label="origin")
        image_of_zero = shift
        ax.scatter(*image_of_zero, color=GREEN, s=52, zorder=8)
        if np.linalg.norm(shift) > 0:
            arrow(ax, zero, image_of_zero, GREEN, r"$\mathbf{b}$")
            ax.annotate(
                r"$F(\mathbf{0})=\mathbf{b}$",
                image_of_zero,
                xytext=(8, 9),
                textcoords="offset points",
                color=GREEN,
                fontsize=11,
            )
        else:
            ax.annotate(
                r"$T(\mathbf{0})=\mathbf{0}$",
                zero,
                xytext=(8, 9),
                textcoords="offset points",
                color=GREEN,
                fontsize=11,
            )

    fig.suptitle("A bias translates the transformed space", fontsize=16, weight="bold")
    return fig


def figure_matrix_anatomy() -> Figure:
    matrix = np.array(
        [
            [1.0, 2.0, -1.0, 0.0],
            [3.0, -2.0, 4.0, 1.0],
            [0.0, 5.0, 2.0, -3.0],
        ]
    )

    fig, ax = plt.subplots(figsize=(9.8, 5.2), constrained_layout=True)
    ax.imshow(np.ones_like(matrix), cmap="Greys", vmin=0, vmax=4, alpha=0.12)

    ax.add_patch(Rectangle((-0.48, 0.52), 3.96, 0.96, facecolor=ORANGE, alpha=0.16, edgecolor="none"))
    ax.add_patch(Rectangle((1.52, -0.48), 0.96, 2.96, facecolor=BLUE, alpha=0.16, edgecolor="none"))
    ax.add_patch(Rectangle((1.52, 0.52), 0.96, 0.96, fill=False, edgecolor=GREEN, linewidth=3.0))

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:g}", ha="center", va="center", fontsize=16, color=SLATE)

    ax.set_xticks(range(matrix.shape[1]), [f"column {j}" for j in range(1, matrix.shape[1] + 1)])
    ax.set_yticks(range(matrix.shape[0]), [f"row {i}" for i in range(1, matrix.shape[0] + 1)])
    ax.tick_params(length=0, colors="#475569", labelsize=10)
    ax.set_xlim(-0.7, matrix.shape[1] - 0.3)
    ax.set_ylim(matrix.shape[0] - 0.3, -0.7)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.text(
        1.0,
        -0.16,
        r"$A_{2,3}=4$ lies at row $2$, column $3$",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=12,
        color=GREEN,
        weight="bold",
    )
    fig.suptitle(r"A matrix has shape $3\times4$: rows first, columns second", fontsize=16, weight="bold")
    return fig


def figure_row_column_views() -> Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8), constrained_layout=True)

    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    box = {"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": GRID, "linewidth": 1.5}

    axes[0].set_title("Row view: each row produces one output coordinate", fontsize=13, weight="bold", pad=12)
    axes[0].text(0.08, 0.5, r"$\mathbf{x}$", ha="center", va="center", fontsize=18, color=SLATE, bbox=box)
    for y_pos, row_index, color in ((0.76, 1, BLUE), (0.5, 2, ORANGE), (0.24, 3, GREEN)):
        axes[0].annotate("", xy=(0.37, y_pos), xytext=(0.15, 0.5), arrowprops={"arrowstyle": "->", "color": GRID, "lw": 1.8})
        axes[0].text(
            0.48,
            y_pos,
            rf"$\mathbf{{r}}_{row_index}^{{T}}\mathbf{{x}}$",
            ha="center",
            va="center",
            fontsize=15,
            color=color,
            bbox=box,
        )
        axes[0].annotate("", xy=(0.78, y_pos), xytext=(0.62, y_pos), arrowprops={"arrowstyle": "->", "color": color, "lw": 2.0})
        axes[0].text(0.87, y_pos, rf"$y_{row_index}$", ha="center", va="center", fontsize=16, color=color)

    axes[1].set_title("Column view: input coordinates weight the columns", fontsize=13, weight="bold", pad=12)
    axes[1].text(0.18, 0.67, r"$x_1\mathbf{a}_1$", ha="center", va="center", fontsize=16, color=BLUE, bbox=box)
    axes[1].text(0.18, 0.33, r"$x_2\mathbf{a}_2$", ha="center", va="center", fontsize=16, color=ORANGE, bbox=box)
    axes[1].annotate("", xy=(0.53, 0.55), xytext=(0.31, 0.67), arrowprops={"arrowstyle": "->", "color": BLUE, "lw": 2.0})
    axes[1].annotate("", xy=(0.53, 0.45), xytext=(0.31, 0.33), arrowprops={"arrowstyle": "->", "color": ORANGE, "lw": 2.0})
    axes[1].text(0.58, 0.5, "+", ha="center", va="center", fontsize=24, color=SLATE)
    axes[1].annotate("", xy=(0.78, 0.5), xytext=(0.64, 0.5), arrowprops={"arrowstyle": "->", "color": GREEN, "lw": 2.2})
    axes[1].text(0.88, 0.5, r"$\mathbf{y}$", ha="center", va="center", fontsize=18, color=GREEN, bbox=box)
    axes[1].text(0.5, 0.12, r"$\mathbf{y}=x_1\mathbf{a}_1+x_2\mathbf{a}_2$", ha="center", va="center", fontsize=14, color=SLATE)

    fig.suptitle("The same matrix-vector product answers two different questions", fontsize=16, weight="bold")
    return fig


def figure_rectangular_linear_map() -> Figure:
    matrix = np.array([[1.0, -0.5], [0.4, 1.2], [1.2, 0.8]])
    e1, e2 = np.eye(2)
    x = np.array([1.2, 0.8])
    a1, a2 = matrix @ e1, matrix @ e2
    y = matrix @ x

    fig = plt.figure(figsize=(11.4, 5.0), constrained_layout=True)
    ax_input = fig.add_subplot(1, 2, 1)
    ax_output = fig.add_subplot(1, 2, 2, projection="3d")

    style_plane(ax_input, xlim=(-0.4, 2.2), ylim=(-0.4, 2.2), title=r"Input space $\mathbb{R}^2$")
    arrow(ax_input, np.zeros(2), e1, BLUE, r"$\mathbf{e}_1$")
    arrow(ax_input, np.zeros(2), e2, ORANGE, r"$\mathbf{e}_2$")
    arrow(ax_input, np.zeros(2), x, GREEN, r"$\mathbf{x}$")
    ax_input.scatter(0, 0, color=SLATE, s=24, zorder=7)

    ax_output.set_title(r"Output space $\mathbb{R}^3$", pad=12, fontsize=13, weight="bold")
    u, v = np.meshgrid(np.linspace(-0.2, 1.4, 9), np.linspace(-0.2, 1.4, 9))
    column_plane = a1[:, None, None] * u[None, :, :] + a2[:, None, None] * v[None, :, :]
    ax_output.plot_surface(
        column_plane[0],
        column_plane[1],
        column_plane[2],
        color=GRID,
        alpha=0.24,
        linewidth=0,
        shade=False,
    )
    ax_output.text(-0.72, 1.9, 0.85, "column plane", color="#64748b", fontsize=10)
    for vector, color, label in ((a1, BLUE, r"$\mathbf{a}_1$"), (a2, ORANGE, r"$\mathbf{a}_2$"), (y, GREEN, r"$\mathbf{y}=A\mathbf{x}$")):
        ax_output.quiver(0, 0, 0, *vector, color=color, linewidth=2.5, arrow_length_ratio=0.12)
        ax_output.text(*(1.06 * vector), label, color=color, fontsize=11, weight="bold")
    ax_output.scatter(0, 0, 0, color=SLATE, s=24)
    ax_output.set_xlim(-1.0, 1.8)
    ax_output.set_ylim(-0.5, 2.5)
    ax_output.set_zlim(-0.4, 3.0)
    ax_output.set_xlabel("output 1")
    ax_output.set_ylabel("output 2")
    ax_output.set_zlabel("output 3")
    ax_output.grid(True, color=GRID, alpha=0.7)

    fig.suptitle(r"A $3\times2$ matrix maps two input coordinates to three outputs", fontsize=16, weight="bold")
    return fig


def figure_batch_layout() -> Figure:
    fig, ax = plt.subplots(figsize=(11.0, 5.4), constrained_layout=True)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    def shape_box(x: float, y: float, width: float, height: float, color: str, label: str, shape: str) -> None:
        ax.add_patch(Rectangle((x, y), width, height, facecolor=color, edgecolor=color, alpha=0.16, linewidth=2.0))
        ax.add_patch(Rectangle((x, y), width, height, fill=False, edgecolor=color, linewidth=2.0))
        ax.text(x + width / 2, y + height / 2 + 0.18, label, ha="center", va="center", fontsize=15, color=color, weight="bold")
        ax.text(x + width / 2, y + height / 2 - 0.28, shape, ha="center", va="center", fontsize=11, color=SLATE)

    ax.text(0.25, 6.65, "One column vector (algebraic order)", fontsize=13, color=SLATE, weight="bold")
    shape_box(1.4, 4.35, 2.2, 2.0, ORANGE, r"$\mathbf{W}$", r"$m\times n$")
    shape_box(4.8, 4.7, 0.75, 1.6, BLUE, r"$\mathbf{x}$", r"$n\times1$")
    shape_box(8.0, 4.7, 0.75, 1.6, GREEN, r"$\mathbf{y}$", r"$m\times1$")
    ax.text(4.2, 5.35, r"$\times$", ha="center", va="center", fontsize=18, color=SLATE)
    ax.text(6.75, 5.35, r"$=$", ha="center", va="center", fontsize=18, color=SLATE)
    ax.text(10.0, 5.35, r"$\mathbf{y}=\mathbf{W}\mathbf{x}$", ha="center", va="center", fontsize=13, color=SLATE)

    ax.text(0.25, 2.35, "Samples stored as rows", fontsize=13, color=SLATE, weight="bold")
    shape_box(1.0, 0.65, 2.8, 1.6, BLUE, r"$\mathbf{X}$", r"$B\times n$")
    shape_box(4.9, 0.65, 2.1, 1.6, ORANGE, r"$\mathbf{W}^{T}$", r"$n\times m$")
    shape_box(8.2, 0.65, 2.8, 1.6, GREEN, r"$\mathbf{Y}$", r"$B\times m$")
    ax.text(4.35, 1.45, r"$\times$", ha="center", va="center", fontsize=18, color=SLATE)
    ax.text(7.6, 1.45, r"$=$", ha="center", va="center", fontsize=18, color=SLATE)
    ax.text(6.0, 2.6, r"$\mathbf{Y}=\mathbf{X}\mathbf{W}^{T}$", ha="center", fontsize=13, color=SLATE)

    fig.suptitle("Column-vector formulas and samples stored as rows use the same weights", fontsize=16, weight="bold")
    return fig


def figure_composition_spaces() -> Figure:
    fig, ax = plt.subplots(figsize=(11.2, 5.0), constrained_layout=True)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    space_box = {"boxstyle": "round,pad=0.55", "facecolor": "white", "edgecolor": GRID, "linewidth": 1.6}
    matrix_box = {"boxstyle": "round,pad=0.55", "facecolor": "#fff7ed", "edgecolor": ORANGE, "linewidth": 1.8}

    ax.text(1.2, 3.2, r"$\mathbb{R}^{n}$", ha="center", va="center", fontsize=19, color=BLUE, bbox=space_box)
    ax.text(4.5, 3.2, r"$\mathbb{R}^{p}$", ha="center", va="center", fontsize=19, color=SLATE, bbox=space_box)
    ax.text(7.8, 3.2, r"$\mathbb{R}^{m}$", ha="center", va="center", fontsize=19, color=GREEN, bbox=space_box)

    ax.annotate("", xy=(3.7, 3.2), xytext=(2.0, 3.2), arrowprops={"arrowstyle": "->", "color": BLUE, "lw": 2.2})
    ax.text(2.85, 3.75, r"$\mathbf{B}$", ha="center", va="center", fontsize=17, color=BLUE, bbox=matrix_box)
    ax.text(2.85, 2.55, r"$p\times n$", ha="center", va="center", fontsize=11, color=SLATE)

    ax.annotate("", xy=(7.0, 3.2), xytext=(5.3, 3.2), arrowprops={"arrowstyle": "->", "color": ORANGE, "lw": 2.2})
    ax.text(6.15, 3.75, r"$\mathbf{A}$", ha="center", va="center", fontsize=17, color=ORANGE, bbox=matrix_box)
    ax.text(6.15, 2.55, r"$m\times p$", ha="center", va="center", fontsize=11, color=SLATE)

    ax.annotate(
        "",
        xy=(7.3, 4.05),
        xytext=(1.7, 4.05),
        arrowprops={"arrowstyle": "->", "color": GREEN, "lw": 2.4, "connectionstyle": "arc3,rad=-0.24"},
    )
    ax.text(4.5, 5.05, r"$\mathbf{A}\mathbf{B}$", ha="center", va="center", fontsize=18, color=GREEN, bbox=space_box)
    ax.text(4.5, 4.45, r"$m\times n$", ha="center", va="center", fontsize=11, color=SLATE)

    ax.text(9.65, 3.75, "The right matrix acts first", ha="center", va="center", fontsize=12, color=SLATE, weight="bold")
    ax.text(9.65, 3.1, r"$\mathbf{x}\mapsto\mathbf{B}\mathbf{x}\mapsto\mathbf{A}(\mathbf{B}\mathbf{x})$", ha="center", va="center", fontsize=12, color=SLATE)
    ax.text(9.65, 2.45, r"$=\,(\mathbf{A}\mathbf{B})\mathbf{x}$", ha="center", va="center", fontsize=13, color=GREEN)

    fig.suptitle("Matrix multiplication records the composition of two linear maps", fontsize=16, weight="bold")
    return fig


def figure_matrix_product_entry() -> Figure:
    matrix_a = np.array([[1, 2, 0], [-1, 0, 1]])
    matrix_b = np.array([[1, 0, 2, -1], [0, 1, -1, 2], [2, -1, 0, 1]])
    product = matrix_a @ matrix_b

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.8), constrained_layout=True)

    def draw_matrix(ax: Axes, matrix: np.ndarray, title: str) -> None:
        ax.imshow(np.ones_like(matrix), cmap="Greys", vmin=0, vmax=4, alpha=0.12)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:g}", ha="center", va="center", fontsize=15, color=SLATE)
        ax.set_xticks(range(matrix.shape[1]), [str(j) for j in range(1, matrix.shape[1] + 1)])
        ax.set_yticks(range(matrix.shape[0]), [str(i) for i in range(1, matrix.shape[0] + 1)])
        ax.set_xlabel("column")
        ax.set_ylabel("row")
        ax.set_title(title, fontsize=14, weight="bold", pad=12)
        ax.tick_params(length=0, colors="#64748b", labelsize=9)
        for spine in ax.spines.values():
            spine.set_visible(False)

    draw_matrix(axes[0], matrix_a, r"$\mathbf{A}\;(2\times3)$")
    draw_matrix(axes[1], matrix_b, r"$\mathbf{B}\;(3\times4)$")
    draw_matrix(axes[2], product, r"$\mathbf{C}=\mathbf{A}\mathbf{B}\;(2\times4)$")

    axes[0].add_patch(Rectangle((-0.48, 0.52), 2.96, 0.96, facecolor=ORANGE, alpha=0.22, edgecolor=ORANGE, linewidth=2.2))
    axes[1].add_patch(Rectangle((1.52, -0.48), 0.96, 2.96, facecolor=BLUE, alpha=0.20, edgecolor=BLUE, linewidth=2.2))
    axes[2].add_patch(Rectangle((1.52, 0.52), 0.96, 0.96, fill=False, edgecolor=GREEN, linewidth=3.0))

    fig.text(
        0.5,
        0.02,
        r"$C_{2,3}=(-1)\cdot2+0\cdot(-1)+1\cdot0=-2$",
        ha="center",
        va="bottom",
        fontsize=13,
        color=GREEN,
        weight="bold",
    )
    fig.suptitle("One product entry pairs one row of A with one column of B", fontsize=16, weight="bold")
    return fig


def figure_product_columns() -> Figure:
    fig, ax = plt.subplots(figsize=(11.2, 5.2), constrained_layout=True)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6)
    ax.axis("off")

    colors = (BLUE, ORANGE, GREEN, "#7c3aed")

    def draw_column_matrix(x: float, y: float, rows: int, columns: int, label: str, row_label: str) -> None:
        cell_width = 0.72
        cell_height = 0.72
        for j in range(columns):
            for i in range(rows):
                ax.add_patch(
                    Rectangle(
                        (x + j * cell_width, y + (rows - 1 - i) * cell_height),
                        cell_width,
                        cell_height,
                        facecolor=colors[j],
                        edgecolor="white",
                        alpha=0.24,
                        linewidth=1.2,
                    )
                )
            ax.add_patch(
                Rectangle(
                    (x + j * cell_width, y),
                    cell_width,
                    rows * cell_height,
                    fill=False,
                    edgecolor=colors[j],
                    linewidth=2.0,
                )
            )
            ax.text(x + (j + 0.5) * cell_width, y - 0.38, rf"${label}_{j + 1}$", ha="center", va="center", fontsize=11, color=colors[j])
        ax.text(x + columns * cell_width / 2, y + rows * cell_height + 0.48, row_label, ha="center", va="center", fontsize=14, color=SLATE, weight="bold")

    draw_column_matrix(0.9, 1.35, 3, 4, r"\mathbf{b}", r"columns of $\mathbf{B}$")
    draw_column_matrix(8.8, 1.7, 2, 4, r"\mathbf{c}", r"columns of $\mathbf{C}=\mathbf{A}\mathbf{B}$")

    ax.text(5.55, 3.0, r"apply $\mathbf{A}$", ha="center", va="center", fontsize=15, color=ORANGE, weight="bold")
    ax.annotate("", xy=(8.35, 2.85), xytext=(3.95, 2.85), arrowprops={"arrowstyle": "->", "color": ORANGE, "lw": 2.5})
    ax.text(6.15, 2.25, r"$\mathbf{c}_j=\mathbf{A}\mathbf{b}_j$ for every $j$", ha="center", va="center", fontsize=13, color=SLATE)
    ax.text(6.15, 0.55, r"$(\mathbf{A}\mathbf{B})_{:,j}=\mathbf{A}\mathbf{B}_{:,j}$", ha="center", va="center", fontsize=14, color=GREEN, weight="bold")

    fig.suptitle("Left multiplication transforms every column without changing its position", fontsize=16, weight="bold")
    return fig


def figure_noncommutativity() -> Figure:
    shear = np.array([[1.0, 1.0], [0.0, 1.0]])
    scale = np.array([[2.0, 0.0], [0.0, 1.0]])
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.4), constrained_layout=True)

    def draw_shape(ax: Axes, vertices: np.ndarray, color: str, label: str, *, alpha: float, linestyle: str = "-") -> None:
        ax.fill(vertices[:, 0], vertices[:, 1], color=color, alpha=alpha, zorder=2)
        ax.plot(vertices[:, 0], vertices[:, 1], color=color, linewidth=2.3, linestyle=linestyle, label=label, zorder=3)

    for ax in axes:
        style_plane(ax, xlim=(-0.5, 4.6), ylim=(-0.5, 2.2), title="")
        draw_shape(ax, square, SLATE, "original", alpha=0.05, linestyle="--")

    axes[0].set_title("Original unit square", fontsize=13, weight="bold", pad=10)
    draw_shape(axes[0], square, BLUE, "original", alpha=0.16)

    intermediate_ab = (scale @ square.T).T
    final_ab = (shear @ intermediate_ab.T).T
    axes[1].set_title(r"$\mathbf{A}\mathbf{B}$: scale, then shear", fontsize=13, weight="bold", pad=10)
    draw_shape(axes[1], intermediate_ab, GRID, "after B", alpha=0.10, linestyle=":")
    draw_shape(axes[1], final_ab, GREEN, "after A", alpha=0.18)

    intermediate_ba = (shear @ square.T).T
    final_ba = (scale @ intermediate_ba.T).T
    axes[2].set_title(r"$\mathbf{B}\mathbf{A}$: shear, then scale", fontsize=13, weight="bold", pad=10)
    draw_shape(axes[2], intermediate_ba, GRID, "after A", alpha=0.10, linestyle=":")
    draw_shape(axes[2], final_ba, ORANGE, "after B", alpha=0.18)

    axes[1].text(0.96, 0.08, r"$\mathbf{A}\mathbf{B}\ne\mathbf{B}\mathbf{A}$", transform=axes[1].transAxes, ha="right", va="bottom", fontsize=13, color=SLATE, weight="bold")
    fig.suptitle("Changing the order changes the transformation", fontsize=16, weight="bold")
    return fig


def figure_system_matrix_correspondence() -> Figure:
    coefficient = np.array([[1, 1, 1], [2, 3, 4], [1, 2, 4]])
    rhs = np.array([[2], [5], [5]])
    augmented = np.hstack((coefficient, rhs))
    column_colors = (BLUE, ORANGE, GREEN, "#7c3aed")

    fig, ax = plt.subplots(figsize=(12.0, 5.8), constrained_layout=True)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis("off")

    equations = (
        r"$x_1+x_2+x_3=2$",
        r"$2x_1+3x_2+4x_3=5$",
        r"$x_1+2x_2+4x_3=5$",
    )
    ax.text(2.0, 6.55, "Linear system", ha="center", fontsize=14, color=SLATE, weight="bold")
    for row, equation in enumerate(equations):
        ax.text(2.0, 5.45 - row * 1.2, equation, ha="center", va="center", fontsize=14, color=SLATE)

    def draw_matrix(matrix: np.ndarray, x: float, y: float, colors: tuple[str, ...], label: str) -> None:
        cell_width = 0.72
        cell_height = 0.72
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.add_patch(
                    Rectangle(
                        (x + j * cell_width, y + (matrix.shape[0] - 1 - i) * cell_height),
                        cell_width,
                        cell_height,
                        facecolor=colors[j],
                        edgecolor="white",
                        alpha=0.20,
                        linewidth=1.0,
                    )
                )
                ax.text(
                    x + (j + 0.5) * cell_width,
                    y + (matrix.shape[0] - i - 0.5) * cell_height,
                    f"{matrix[i, j]:g}",
                    ha="center",
                    va="center",
                    fontsize=13,
                    color=SLATE,
                )
        ax.add_patch(
            Rectangle(
                (x, y),
                matrix.shape[1] * cell_width,
                matrix.shape[0] * cell_height,
                fill=False,
                edgecolor=GRID,
                linewidth=1.8,
            )
        )
        ax.text(x + matrix.shape[1] * cell_width / 2, y + matrix.shape[0] * cell_height + 0.5, label, ha="center", fontsize=14, color=SLATE, weight="bold")

    draw_matrix(coefficient, 5.2, 3.2, column_colors[:3], r"$\mathbf{A}$")
    ax.text(7.8, 4.25, r"$\times$", ha="center", va="center", fontsize=18, color=SLATE)
    draw_matrix(np.array([[3], [-3], [2]]), 8.25, 3.2, ("#94a3b8",), r"$\mathbf{x}$")
    ax.text(9.35, 4.25, r"$=$", ha="center", va="center", fontsize=18, color=SLATE)
    draw_matrix(rhs, 9.75, 3.2, (column_colors[3],), r"$\mathbf{b}$")
    ax.text(7.75, 1.95, r"$\mathbf{A}\mathbf{x}=\mathbf{b}$", ha="center", fontsize=14, color=GREEN, weight="bold")

    draw_matrix(augmented, 12.25, 3.2, column_colors, r"$[\mathbf{A}\mid\mathbf{b}]$")
    ax.plot([14.41, 14.41], [3.2, 5.36], color=SLATE, linewidth=2.0)
    ax.text(14.77, 2.55, "right-hand side", ha="center", fontsize=10, color=column_colors[3])

    ax.annotate("", xy=(4.65, 4.25), xytext=(3.35, 4.25), arrowprops={"arrowstyle": "->", "color": GRID, "lw": 2.0})
    ax.annotate("", xy=(11.75, 4.25), xytext=(10.85, 4.25), arrowprops={"arrowstyle": "->", "color": GRID, "lw": 2.0})

    ax.text(5.56, 2.62, r"$x_1$", ha="center", fontsize=10, color=column_colors[0])
    ax.text(6.28, 2.62, r"$x_2$", ha="center", fontsize=10, color=column_colors[1])
    ax.text(7.0, 2.62, r"$x_3$", ha="center", fontsize=10, color=column_colors[2])

    fig.suptitle("The equations, matrix equation, and augmented matrix store the same system", fontsize=16, weight="bold")
    return fig


def figure_row_operation_same_solution() -> Figure:
    x_values = np.linspace(-1.0, 4.0, 240)
    first_line = 3.0 - x_values
    second_line = 2.0 * x_values
    replacement_line = np.full_like(x_values, 2.0)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), constrained_layout=True)
    for ax in axes:
        style_plane(ax, xlim=(-1.0, 4.0), ylim=(-1.0, 4.0), title="")
        ax.plot(x_values, first_line, color=BLUE, linewidth=2.4, label=r"$x_1+x_2=3$")
        ax.scatter(1.0, 2.0, color=GREEN, s=58, zorder=7)
        ax.annotate(r"$(1,2)$", (1.0, 2.0), xytext=(8, 8), textcoords="offset points", color=GREEN, fontsize=11, weight="bold")
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")

    axes[0].plot(x_values, second_line, color=ORANGE, linewidth=2.4, label=r"$2x_1-x_2=0$")
    axes[0].set_title("Before the row operation", fontsize=13, weight="bold", pad=10)
    axes[0].legend(loc="upper right", fontsize=9)

    axes[1].plot(x_values, replacement_line, color=ORANGE, linewidth=2.4, label=r"$-3x_2=-6$")
    axes[1].set_title(r"After $R_2\leftarrow R_2-2R_1$", fontsize=13, weight="bold", pad=10)
    axes[1].legend(loc="upper right", fontsize=9)

    fig.suptitle("A row operation changes an equation but preserves the common solution", fontsize=16, weight="bold")
    return fig


def figure_gaussian_elimination_steps() -> Figure:
    matrices = (
        np.array([[1, 1, 1, 2], [2, 3, 4, 5], [1, 2, 4, 5]]),
        np.array([[1, 1, 1, 2], [0, 1, 2, 1], [0, 1, 3, 3]]),
        np.array([[1, 1, 1, 2], [0, 1, 2, 1], [0, 0, 1, 2]]),
    )

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.9), constrained_layout=True)

    def draw_augmented(ax: Axes, matrix: np.ndarray, title: str, pivots: tuple[tuple[int, int], ...] = ()) -> None:
        ax.imshow(np.ones_like(matrix), cmap="Greys", vmin=0, vmax=4, alpha=0.10)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:g}", ha="center", va="center", fontsize=15, color=SLATE)
        ax.axvline(2.5, color=SLATE, linewidth=2.0)
        for i, j in pivots:
            ax.add_patch(Rectangle((j - 0.46, i - 0.46), 0.92, 0.92, fill=False, edgecolor=GREEN, linewidth=3.0))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=13, weight="bold", pad=10)
        for spine in ax.spines.values():
            spine.set_visible(False)

    draw_augmented(axes[0], matrices[0], "Start")
    draw_augmented(axes[1], matrices[1], "Eliminate below the first pivot")
    draw_augmented(axes[2], matrices[2], "Row echelon form", ((0, 0), (1, 1), (2, 2)))

    axes[0].text(1.5, 3.05, r"$R_2\leftarrow R_2-2R_1$" "\n" r"$R_3\leftarrow R_3-R_1$", ha="center", va="top", fontsize=10, color=SLATE)
    axes[1].text(1.5, 3.05, r"$R_3\leftarrow R_3-R_2$", ha="center", va="top", fontsize=10, color=SLATE)
    axes[2].text(1.5, 3.05, r"$x_3=2\;\longrightarrow\;x_2=-3\;\longrightarrow\;x_1=3$", ha="center", va="top", fontsize=10.5, color=GREEN, weight="bold")

    fig.suptitle("Gaussian elimination moves downward; back substitution moves upward", fontsize=16, weight="bold")
    return fig


def figure_parameter_system_outcomes() -> Figure:
    point = np.array([1.0, 1.0, 0.0])
    direction = np.array([1.0, -2.0, 1.0])
    parameter = np.linspace(-1.4, 2.6, 160)
    line = point[:, None] + direction[:, None] * parameter[None, :]
    x_grid, y_grid = np.meshgrid(np.linspace(-0.8, 3.8, 18), np.linspace(-3.8, 3.4, 18))

    cases = (
        (1.0, 2.0, "one point", GREEN),
        (0.0, 0.0, "the whole line", BLUE),
        (0.0, 1.0, "no intersection", ORANGE),
    )

    fig = plt.figure(figsize=(12.2, 4.8), constrained_layout=True)
    for index, (lambda_value, mu_value, title, color) in enumerate(cases, start=1):
        ax = fig.add_subplot(1, 3, index, projection="3d")
        z_grid = (3.0 + mu_value - x_grid - 2.0 * y_grid) / (3.0 + lambda_value)
        ax.plot_surface(x_grid, y_grid, z_grid, color=color, alpha=0.16, linewidth=0, shade=False)
        ax.plot(line[0], line[1], line[2], color=SLATE, linewidth=2.6)
        if lambda_value != 0:
            intersection_parameter = mu_value / lambda_value
            intersection = point + intersection_parameter * direction
            ax.scatter(*intersection, color=GREEN, s=55)
            ax.text(*(intersection + np.array([0.08, 0.08, 0.08])), r"$(3,-3,2)$", color=GREEN, fontsize=9, weight="bold")
        elif mu_value == 0:
            selected = np.array([-1.0, 0.0, 1.0])
            selected_points = point[:, None] + direction[:, None] * selected[None, :]
            ax.scatter(selected_points[0], selected_points[1], selected_points[2], color=BLUE, s=28)
        ax.set_xlim(-1.0, 4.0)
        ax.set_ylim(-4.0, 3.5)
        ax.set_zlim(-1.8, 2.8)
        ax.set_xlabel(r"$x_1$", labelpad=2)
        ax.set_ylabel(r"$x_2$", labelpad=2)
        ax.set_zlabel(r"$x_3$", labelpad=2)
        ax.set_title(title + "\n" + rf"$\lambda={lambda_value:g},\;\mu={mu_value:g}$", fontsize=12, weight="bold", pad=8)
        ax.view_init(elev=23, azim=-57)
        ax.grid(True, color=GRID, alpha=0.6)

    fig.suptitle("The full system has one point, a whole line, or no solution", fontsize=16, weight="bold")
    return fig


def figure_affine_solution_translation() -> Figure:
    particular = np.array([1.0, 1.0, 0.0])
    direction = np.array([1.0, -2.0, 1.0])
    parameter = np.linspace(-1.5, 1.5, 160)
    homogeneous = direction[:, None] * parameter[None, :]
    nonhomogeneous = particular[:, None] + homogeneous

    fig = plt.figure(figsize=(10.4, 5.6), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(homogeneous[0], homogeneous[1], homogeneous[2], color=BLUE, linewidth=3.0, label="homogeneous solutions")
    ax.plot(nonhomogeneous[0], nonhomogeneous[1], nonhomogeneous[2], color=GREEN, linewidth=3.0, label="nonhomogeneous solutions")

    for t_value in (-1.0, 0.0, 1.0):
        start = t_value * direction
        end = particular + start
        ax.scatter(*start, color=BLUE, s=32)
        ax.scatter(*end, color=GREEN, s=32)
        ax.quiver(*start, *particular, color=ORANGE, linewidth=1.8, arrow_length_ratio=0.14)

    ax.scatter(0, 0, 0, color=SLATE, s=30)
    ax.text(0.08, 0.08, 0.08, r"$\mathbf{0}$", color=SLATE, fontsize=10)
    ax.text(*(particular + np.array([0.08, 0.08, 0.08])), r"$\mathbf{x}_p$", color=ORANGE, fontsize=11, weight="bold")
    ax.set_xlim(-1.8, 2.8)
    ax.set_ylim(-3.5, 3.5)
    ax.set_zlim(-1.8, 1.8)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_zlabel(r"$x_3$")
    ax.view_init(elev=23, azim=-57)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, color=GRID, alpha=0.6)

    fig.suptitle("A consistent nonhomogeneous solution set is a translated homogeneous solution set", fontsize=16, weight="bold")
    return fig


def figure_two_by_two_determinant() -> Figure:
    fig, ax = plt.subplots(figsize=(11.0, 5.2), constrained_layout=True)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def draw_two_by_two(x: float, y: float, entries: tuple[tuple[str, str], tuple[str, str]], label: str) -> None:
        offsets = ((-0.35, 0.35), (0.35, 0.35), (-0.35, -0.35), (0.35, -0.35))
        values = (entries[0][0], entries[0][1], entries[1][0], entries[1][1])
        for (x_offset, y_offset), value in zip(offsets, values):
            ax.text(x + x_offset, y + y_offset, value, ha="center", va="center", fontsize=18, color=SLATE)
        ax.plot([x - 0.85, x - 0.85, x - 0.68], [y + 0.85, y - 0.85, y - 0.85], color=SLATE, linewidth=2.0)
        ax.plot([x - 0.85, x - 0.68], [y + 0.85, y + 0.85], color=SLATE, linewidth=2.0)
        ax.plot([x + 0.85, x + 0.85, x + 0.68], [y + 0.85, y - 0.85, y - 0.85], color=SLATE, linewidth=2.0)
        ax.plot([x + 0.85, x + 0.68], [y + 0.85, y + 0.85], color=SLATE, linewidth=2.0)
        ax.text(x, y + 1.2, label, ha="center", va="center", fontsize=15, color=SLATE, weight="bold")

    draw_two_by_two(1.8, 3.4, ((r"$a$", r"$b$"), (r"$c$", r"$d$")), r"$A$")
    ax.text(1.8, 2.15, r"$a\ne0$", ha="center", va="center", fontsize=12, color=SLATE)

    ax.annotate("", xy=(5.05, 3.4), xytext=(3.0, 3.4), arrowprops={"arrowstyle": "->", "color": ORANGE, "lw": 2.4})
    ax.text(4.0, 4.05, r"$R_2\leftarrow R_2-\frac{c}{a}R_1$", ha="center", va="center", fontsize=13, color=ORANGE)

    draw_two_by_two(
        6.4,
        3.4,
        ((r"$a$", r"$b$"), (r"$0$", r"$d-\frac{cb}{a}$")),
        r"$U$",
    )

    ax.annotate("", xy=(9.35, 3.4), xytext=(7.75, 3.4), arrowprops={"arrowstyle": "->", "color": GREEN, "lw": 2.4})
    ax.text(10.45, 3.65, r"$a\left(d-\frac{cb}{a}\right)$", ha="center", va="center", fontsize=17, color=GREEN, weight="bold")
    ax.text(10.45, 2.85, r"$=\;ad-bc$", ha="center", va="center", fontsize=18, color=GREEN, weight="bold")
    ax.text(6.0, 1.15, "row addition preserves the quantity; the pivot product reveals it", ha="center", fontsize=11.5, color=SLATE)

    fig.suptitle(r"For a $2\times2$ system, elimination naturally produces $ad-bc$", fontsize=16, weight="bold")
    return fig


def figure_permutation_selections() -> Figure:
    permutations = (
        ((1, 2, 3), 1, 0),
        ((2, 3, 1), 1, 2),
        ((3, 1, 2), 1, 2),
        ((1, 3, 2), -1, 1),
        ((2, 1, 3), -1, 1),
        ((3, 2, 1), -1, 3),
    )

    fig, axes = plt.subplots(2, 3, figsize=(10.8, 7.0), constrained_layout=True)
    for ax, (permutation, sign, inversion_count) in zip(axes.flat, permutations):
        ax.imshow(np.ones((3, 3)), cmap="Greys", vmin=0, vmax=4, alpha=0.10)
        color = GREEN if sign > 0 else ORANGE
        for i in range(3):
            for j in range(3):
                ax.text(j, i, rf"$a_{{{i + 1}{j + 1}}}$", ha="center", va="center", fontsize=11, color=SLATE)
            selected_column = permutation[i] - 1
            ax.add_patch(Rectangle((selected_column - 0.46, i - 0.46), 0.92, 0.92, fill=False, edgecolor=color, linewidth=3.0))
        sign_label = "+" if sign > 0 else "-"
        ax.set_title(
            rf"$\sigma={permutation}$   {sign_label}" + "\n" + rf"$\tau(\sigma)={inversion_count}$",
            fontsize=11,
            weight="bold",
            color=color,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle("A determinant term chooses exactly one entry from every row and every column", fontsize=16, weight="bold")
    return fig


def figure_inversion_crossings() -> Figure:
    permutation = (3, 1, 4, 2)
    colors = (BLUE, ORANGE, GREEN, "#7c3aed")

    fig, ax = plt.subplots(figsize=(9.6, 5.4), constrained_layout=True)
    ax.set_xlim(0.3, 4.7)
    ax.set_ylim(0.2, 4.8)
    ax.axis("off")

    for position in range(1, 5):
        ax.scatter(position, 4.0, s=280, facecolor="white", edgecolor=GRID, linewidth=2.0, zorder=4)
        ax.scatter(position, 1.0, s=280, facecolor="white", edgecolor=GRID, linewidth=2.0, zorder=4)
        ax.text(position, 4.0, str(position), ha="center", va="center", fontsize=13, color=SLATE, weight="bold", zorder=5)
        ax.text(position, 1.0, str(position), ha="center", va="center", fontsize=13, color=SLATE, weight="bold", zorder=5)

    for index, image in enumerate(permutation, start=1):
        ax.plot([index, image], [3.78, 1.22], color=colors[index - 1], linewidth=2.8, zorder=2)

    ax.text(0.55, 4.45, "positions", fontsize=12, color=SLATE, weight="bold")
    ax.text(0.55, 0.45, "values", fontsize=12, color=SLATE, weight="bold")
    ax.text(4.55, 3.0, r"$\sigma=(3,1,4,2)$", ha="right", fontsize=14, color=SLATE)
    ax.text(4.55, 2.5, r"$\tau(\sigma)=3$", ha="right", fontsize=14, color=ORANGE, weight="bold")
    ax.text(4.55, 2.0, r"$\operatorname{sgn}(\sigma)=-1$", ha="right", fontsize=14, color=ORANGE, weight="bold")

    fig.suptitle("Each crossing is an inversion pair", fontsize=16, weight="bold")
    return fig


def figure_determinant_orientation() -> Figure:
    matrix = np.array([[2.0, 0.5], [0.3, 1.2]])
    swapped = matrix[:, [1, 0]]
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.5), constrained_layout=True)

    def transformed_polygon(transform: np.ndarray) -> np.ndarray:
        return (transform @ square.T).T

    panels = (
        (np.eye(2), BLUE, r"$\det(I)=1$", "standard orientation"),
        (matrix, GREEN, r"$\det(A)=2.25$", "area scaled, orientation kept"),
        (swapped, ORANGE, r"$\det(A_{\mathrm{swap}})=-2.25$", "same area, orientation reversed"),
    )

    for ax, (transform, color, determinant_label, title) in zip(axes, panels):
        style_plane(ax, xlim=(-0.6, 3.0), ylim=(-0.6, 2.4), title=title)
        polygon = transformed_polygon(transform)
        ax.fill(polygon[:, 0], polygon[:, 1], color=color, alpha=0.20)
        ax.plot(polygon[:, 0], polygon[:, 1], color=color, linewidth=2.5)
        first_column = transform[:, 0]
        second_column = transform[:, 1]
        arrow(ax, np.zeros(2), first_column, BLUE, r"$\mathbf{a}_1$")
        arrow(ax, np.zeros(2), second_column, ORANGE, r"$\mathbf{a}_2$")
        ax.text(0.96, 0.06, determinant_label, transform=ax.transAxes, ha="right", va="bottom", fontsize=12, color=color, weight="bold")

    fig.suptitle("The determinant is signed area in two dimensions", fontsize=16, weight="bold")
    return fig


def figure_invertibility_equivalences() -> Figure:
    fig, ax = plt.subplots(figsize=(8.6, 8.6), constrained_layout=True)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    statements = (
        (8.8, r"$\det(A)\ne0$", "determinant", BLUE),
        (7.0, r"$n$ pivots", "elimination", GREEN),
        (5.2, r"$Ah=0\Rightarrow h=0$", "homogeneous directions", ORANGE),
        (3.4, r"$Ax=b$ has one solution for every $b$", "linear systems", "#7c3aed"),
        (1.6, r"$A^{-1}$ exists", "inverse map", BLUE),
    )
    box = {"boxstyle": "round,pad=0.55", "facecolor": "white", "edgecolor": GRID, "linewidth": 1.8}

    for index, (y_pos, statement, viewpoint, color) in enumerate(statements):
        ax.text(5.2, y_pos, statement, ha="center", va="center", fontsize=15, color=color, weight="bold", bbox=box)
        ax.text(8.15, y_pos, viewpoint, ha="left", va="center", fontsize=10.5, color=SLATE)
        if index < len(statements) - 1:
            next_y = statements[index + 1][0]
            ax.annotate(
                "",
                xy=(5.2, next_y + 0.55),
                xytext=(5.2, y_pos - 0.55),
                arrowprops={"arrowstyle": "<->", "color": "#94a3b8", "lw": 2.0},
            )

    ax.text(1.0, 5.2, "same\nnondegeneracy", ha="center", va="center", fontsize=12, color=SLATE, weight="bold")
    ax.annotate("", xy=(2.25, 8.8), xytext=(2.25, 1.6), arrowprops={"arrowstyle": "<->", "color": GRID, "lw": 2.0})
    fig.suptitle("Five equivalent views of an invertible square matrix", fontsize=16, weight="bold")
    return fig


def figure_inverse_solves_basis() -> Figure:
    fig, ax = plt.subplots(figsize=(12.0, 5.4), constrained_layout=True)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6)
    ax.axis("off")

    box = {"boxstyle": "round,pad=0.5", "facecolor": "white", "edgecolor": GRID, "linewidth": 1.6}
    inverse_box = {"boxstyle": "round,pad=0.55", "facecolor": "#eff6ff", "edgecolor": BLUE, "linewidth": 2.0}

    for y_pos, basis_label, solution_label in (
        (4.6, r"$\mathbf{e}_1$", r"$\mathbf{x}_1$"),
        (3.0, r"$\vdots$", r"$\vdots$"),
        (1.4, r"$\mathbf{e}_n$", r"$\mathbf{x}_n$"),
    ):
        ax.text(0.9, y_pos, basis_label, ha="center", va="center", fontsize=16, color=ORANGE, bbox=box)
        ax.annotate("", xy=(3.6, y_pos), xytext=(1.55, y_pos), arrowprops={"arrowstyle": "->", "color": GRID, "lw": 2.0})
        if basis_label != r"$\vdots$":
            ax.text(2.55, y_pos + 0.35, r"solve $A\mathbf{x}_j=\mathbf{e}_j$", ha="center", fontsize=9.5, color=SLATE)
        ax.text(4.2, y_pos, solution_label, ha="center", va="center", fontsize=16, color=GREEN, bbox=box)

    ax.annotate("", xy=(6.5, 3.0), xytext=(4.9, 3.0), arrowprops={"arrowstyle": "->", "color": GREEN, "lw": 2.3})
    ax.text(5.7, 3.45, "collect columns", ha="center", fontsize=10.5, color=SLATE)
    ax.text(7.25, 3.0, r"$A^{-1}=[\,\mathbf{x}_1\;\cdots\;\mathbf{x}_n\,]$", ha="center", va="center", fontsize=14, color=BLUE, bbox=inverse_box)

    ax.text(9.45, 4.55, r"$\mathbf{b}$", ha="center", va="center", fontsize=16, color=ORANGE, bbox=box)
    ax.annotate("", xy=(11.25, 4.55), xytext=(10.1, 4.55), arrowprops={"arrowstyle": "->", "color": BLUE, "lw": 2.2})
    ax.text(10.68, 4.9, r"$A^{-1}$", ha="center", fontsize=11, color=BLUE, weight="bold")
    ax.text(11.85, 4.55, r"$\mathbf{x}$", ha="center", va="center", fontsize=16, color=GREEN, bbox=box)
    ax.annotate("", xy=(9.95, 1.45), xytext=(11.35, 1.45), arrowprops={"arrowstyle": "->", "color": ORANGE, "lw": 2.2})
    ax.text(10.68, 1.8, r"$A$", ha="center", fontsize=11, color=ORANGE, weight="bold")
    ax.text(9.35, 1.45, r"$\mathbf{b}$", ha="center", va="center", fontsize=16, color=ORANGE, bbox=box)
    ax.text(11.85, 1.45, r"$\mathbf{x}$", ha="center", va="center", fontsize=16, color=GREEN, bbox=box)
    ax.text(10.65, 3.0, r"$A^{-1}$ reverses $A$", ha="center", va="center", fontsize=12, color=SLATE, weight="bold")

    fig.suptitle("Solving for the basis vectors builds the inverse matrix column by column", fontsize=16, weight="bold")
    return fig


def figure_cramer_column_replacement() -> Figure:
    fig, ax = plt.subplots(figsize=(12.0, 5.2), constrained_layout=True)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6)
    ax.axis("off")

    box = {"boxstyle": "round,pad=0.55", "facecolor": "white", "edgecolor": GRID, "linewidth": 1.6}
    ax.text(1.9, 3.4, r"$\det(\mathbf{a}_1,\ldots,\mathbf{b},\ldots,\mathbf{a}_n)$", ha="center", va="center", fontsize=14, color=BLUE, bbox=box)
    ax.text(1.9, 2.4, r"$\mathbf{b}=\sum_k x_k\mathbf{a}_k$", ha="center", va="center", fontsize=13, color=SLATE)

    ax.annotate("", xy=(5.1, 3.35), xytext=(3.65, 3.35), arrowprops={"arrowstyle": "->", "color": ORANGE, "lw": 2.3})
    ax.text(4.38, 3.75, "column linearity", ha="center", fontsize=10.5, color=ORANGE, weight="bold")

    ax.text(6.45, 4.35, r"$k\ne j:$ repeated column", ha="center", va="center", fontsize=13, color=ORANGE, bbox=box)
    ax.text(6.45, 3.35, r"$x_k\det(\ldots,\mathbf{a}_k,\ldots,\mathbf{a}_k,\ldots)=0$", ha="center", va="center", fontsize=12, color=ORANGE)
    ax.text(6.45, 2.05, r"$k=j:$ original columns", ha="center", va="center", fontsize=13, color=GREEN, bbox=box)
    ax.text(6.45, 1.05, r"$x_j\det(A)$", ha="center", va="center", fontsize=15, color=GREEN, weight="bold")

    ax.annotate("", xy=(10.0, 3.0), xytext=(8.55, 3.0), arrowprops={"arrowstyle": "->", "color": GREEN, "lw": 2.4})
    ax.text(11.35, 3.65, r"$\det(A_j(\mathbf{b}))$", ha="center", va="center", fontsize=15, color=BLUE, bbox=box)
    ax.text(11.35, 2.55, r"$=x_j\det(A)$", ha="center", va="center", fontsize=16, color=GREEN, weight="bold")
    ax.text(11.35, 1.55, r"$x_j=\frac{\det(A_j(\mathbf{b}))}{\det(A)}$", ha="center", va="center", fontsize=13.5, color=SLATE)

    fig.suptitle("Cramer's rule isolates one coordinate because every duplicate-column term vanishes", fontsize=16, weight="bold")
    return fig


def figure_singular_system_two_rhs() -> Figure:
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.8), constrained_layout=True)

    output_ax, input_ax, text_ax = axes
    style_plane(output_ax, xlim=(-0.5, 4.0), ylim=(-1.0, 8.2), title="Output: only one reachable line")
    output_x = np.linspace(-0.5, 4.0, 120)
    output_ax.plot(output_x, 2.0 * output_x, color=BLUE, linewidth=2.8, label=r"$y_2=2y_1$")
    output_ax.scatter(3.0, 6.0, s=70, color=GREEN, zorder=6)
    output_ax.scatter(3.0, 7.0, s=70, color=ORANGE, zorder=6)
    output_ax.annotate(r"$\mathbf{b}_{\mathrm{on}}=(3,6)$", (3.0, 6.0), xytext=(-88, -18), textcoords="offset points", color=GREEN, fontsize=10, weight="bold")
    output_ax.annotate(r"$\mathbf{b}_{\mathrm{off}}=(3,7)$", (3.0, 7.0), xytext=(-92, 10), textcoords="offset points", color=ORANGE, fontsize=10, weight="bold")
    output_ax.legend(loc="upper left", fontsize=9)

    style_plane(input_ax, xlim=(-3.0, 5.0), ylim=(-2.0, 3.5), title="Input: infinitely many preimages")
    parameter = np.linspace(-1.8, 3.0, 120)
    solution_x1 = 3.0 - 2.0 * parameter
    input_ax.plot(solution_x1, parameter, color=GREEN, linewidth=2.8)
    input_ax.scatter(3.0, 0.0, s=60, color=ORANGE, zorder=6)
    arrow(input_ax, np.array([3.0, 0.0]), np.array([1.0, 1.0]), BLUE, r"$\mathbf{h}=(-2,1)$")
    input_ax.text(0.04, 0.94, r"$x_1+2x_2=3$", transform=input_ax.transAxes, ha="left", va="top", fontsize=11, color=GREEN, weight="bold")

    text_ax.set_xlim(0, 1)
    text_ax.set_ylim(0, 1)
    text_ax.axis("off")
    result_box = {"boxstyle": "round,pad=0.55", "facecolor": "white", "edgecolor": GRID, "linewidth": 1.6}
    text_ax.text(0.5, 0.78, r"$\mathbf{b}_{\mathrm{on}}$", ha="center", va="center", fontsize=14, color=GREEN, weight="bold")
    text_ax.text(0.5, 0.60, "[1  2 | 3]\n[0  0 | 0]", ha="center", va="center", family="monospace", fontsize=12, color=SLATE, bbox=result_box)
    text_ax.text(0.5, 0.43, "consistent + free variable", ha="center", fontsize=10.5, color=GREEN, weight="bold")
    text_ax.text(0.5, 0.28, r"$\mathbf{b}_{\mathrm{off}}$", ha="center", va="center", fontsize=14, color=ORANGE, weight="bold")
    text_ax.text(0.5, 0.11, "[1  2 | 3]\n[0  0 | 1]", ha="center", va="center", family="monospace", fontsize=12, color=SLATE, bbox=result_box)
    text_ax.text(0.5, -0.02, "contradiction", ha="center", fontsize=10.5, color=ORANGE, weight="bold")

    fig.suptitle(r"The same singular matrix can give infinitely many solutions or no solution", fontsize=16, weight="bold")
    return fig


def figure_rank_minor_pivots() -> Figure:
    original = np.array([[1, 2, 0, 1], [0, 1, 1, 1], [1, 3, 1, 2]])
    echelon = np.array([[1, 2, 0, 1], [0, 1, 1, 1], [0, 0, 0, 0]])
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.4), constrained_layout=True)

    def draw_rank_matrix(ax: Axes, matrix: np.ndarray, title: str, *, highlight_minor: bool) -> None:
        ax.imshow(np.ones_like(matrix), cmap="Greys", vmin=0, vmax=4, alpha=0.10)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                ax.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=15, color=SLATE)
        if highlight_minor:
            ax.add_patch(Rectangle((-0.47, -0.47), 1.94, 1.94, fill=False, edgecolor=BLUE, linewidth=3.0))
        else:
            for row, column in ((0, 0), (1, 1)):
                ax.add_patch(Rectangle((column - 0.46, row - 0.46), 0.92, 0.92, fill=False, edgecolor=GREEN, linewidth=3.0))
            ax.add_patch(Rectangle((-0.47, 1.53), 3.94, 0.94, facecolor=ORANGE, edgecolor=ORANGE, alpha=0.12, linewidth=2.0))
        ax.set_xticks(range(matrix.shape[1]), [f"c{index}" for index in range(1, 5)])
        ax.set_yticks(range(matrix.shape[0]), [f"r{index}" for index in range(1, 4)])
        ax.tick_params(length=0, colors="#64748b", labelsize=9)
        ax.set_title(title, fontsize=13, weight="bold", pad=12)
        for spine in ax.spines.values():
            spine.set_visible(False)

    draw_rank_matrix(axes[0], original, "A square window survives", highlight_minor=True)
    draw_rank_matrix(axes[1], echelon, "Elimination reveals the same count", highlight_minor=False)
    fig.text(0.5, 0.025, r"nonzero $2\times2$ minor $\Longleftrightarrow$ 2 pivots $\Longleftrightarrow\operatorname{rank}(A)=2$", ha="center", fontsize=12.5, color=SLATE, weight="bold")
    fig.suptitle("Rank connects determinant witnesses to elimination pivots", fontsize=16, weight="bold")
    return fig


def figure_rank_consistency_criterion() -> Figure:
    fig, ax = plt.subplots(figsize=(9.2, 7.0), constrained_layout=True)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    box = {"boxstyle": "round,pad=0.55", "facecolor": "white", "edgecolor": GRID, "linewidth": 1.7}
    ax.text(5.0, 8.8, r"compare $\operatorname{rank}(A)$ and $\operatorname{rank}([A\mid b])$", ha="center", va="center", fontsize=14, color=BLUE, weight="bold", bbox=box)

    ax.annotate("", xy=(2.5, 6.8), xytext=(4.45, 8.15), arrowprops={"arrowstyle": "->", "color": ORANGE, "lw": 2.2})
    ax.text(2.25, 7.75, "different", ha="center", fontsize=10.5, color=ORANGE, weight="bold")
    ax.text(2.5, 6.1, "no solution", ha="center", va="center", fontsize=15, color=ORANGE, weight="bold", bbox=box)
    ax.text(2.5, 5.2, "new pivot in the\naugmented column", ha="center", va="center", fontsize=10.5, color=SLATE)

    ax.annotate("", xy=(7.5, 6.8), xytext=(5.55, 8.15), arrowprops={"arrowstyle": "->", "color": GREEN, "lw": 2.2})
    ax.text(7.75, 7.75, "equal", ha="center", fontsize=10.5, color=GREEN, weight="bold")
    ax.text(7.5, 6.1, r"compare $\operatorname{rank}(A)$ with $n$", ha="center", va="center", fontsize=13, color=GREEN, weight="bold", bbox=box)

    ax.annotate("", xy=(6.1, 3.35), xytext=(7.1, 5.45), arrowprops={"arrowstyle": "->", "color": BLUE, "lw": 2.2})
    ax.text(5.7, 4.65, r"$=n$", ha="center", fontsize=11, color=BLUE, weight="bold")
    ax.text(5.6, 2.7, "unique solution", ha="center", va="center", fontsize=15, color=BLUE, weight="bold", bbox=box)
    ax.text(5.6, 1.75, "no free variables", ha="center", fontsize=10.5, color=SLATE)

    ax.annotate("", xy=(8.7, 3.35), xytext=(7.9, 5.45), arrowprops={"arrowstyle": "->", "color": "#7c3aed", "lw": 2.2})
    ax.text(9.0, 4.65, r"$<n$", ha="center", fontsize=11, color="#7c3aed", weight="bold")
    ax.text(8.55, 2.7, "infinitely many", ha="center", va="center", fontsize=15, color="#7c3aed", weight="bold", bbox=box)
    ax.text(8.55, 1.75, r"$n-r$ free variables", ha="center", fontsize=10.5, color=SLATE)

    fig.suptitle("Rank first decides consistency, then counts the remaining freedom", fontsize=16, weight="bold")
    return fig


def figure_subspace_vs_affine() -> Figure:
    """A through-origin plane is a subspace; a shifted plane is only affine."""
    fig = plt.figure(figsize=(11.4, 4.8), constrained_layout=True)

    # Left panel: subspace plane  x + 2y - z = 0  (through origin)
    ax_sub = fig.add_subplot(1, 2, 1, projection="3d")
    xx, yy = np.meshgrid(np.linspace(-2.0, 2.0, 16), np.linspace(-2.0, 2.0, 16))
    zz_sub = xx + 2.0 * yy
    ax_sub.plot_surface(xx, yy, zz_sub, color=BLUE, alpha=0.16, linewidth=0, shade=False)
    # two in-plane basis vectors
    b1 = np.array([1.0, 0.0, 1.0])
    b2 = np.array([0.0, 1.0, 2.0])
    for vec, color, label in ((b1, BLUE, r"$\mathbf{b}_1$"), (b2, ORANGE, r"$\mathbf{b}_2$")):
        ax_sub.quiver(0, 0, 0, *vec, color=color, linewidth=2.6, arrow_length_ratio=0.12)
        ax_sub.text(*(1.12 * vec), label, color=color, fontsize=11, weight="bold")
    ax_sub.scatter(0, 0, 0, color=SLATE, s=30, zorder=8)
    ax_sub.text(0.1, 0.1, 0.1, r"$\mathbf{0}$", color=SLATE, fontsize=9)
    ax_sub.set_xlim(-2.2, 2.2)
    ax_sub.set_ylim(-2.2, 2.2)
    ax_sub.set_zlim(-2.2, 2.2)
    ax_sub.set_xlabel(r"$x$")
    ax_sub.set_ylabel(r"$y$")
    ax_sub.set_zlabel(r"$z$")
    ax_sub.set_title(r"Subspace: $x+2y-z=0$", fontsize=12, weight="bold", pad=8)
    ax_sub.view_init(elev=20, azim=-55)
    ax_sub.grid(True, color=GRID, alpha=0.5)

    # Right panel: affine plane  x + 2y - z = 1  (shifted)
    ax_aff = fig.add_subplot(1, 2, 2, projection="3d")
    zz_aff = xx + 2.0 * yy - 1.0
    ax_aff.plot_surface(xx, yy, zz_aff, color=ORANGE, alpha=0.16, linewidth=0, shade=False)
    # particular solution
    xp = np.array([1.0, 0.0, 0.0])
    ax_aff.scatter(*xp, color=GREEN, s=50, zorder=8)
    ax_aff.text(*(xp + np.array([0.1, 0.1, 0.1])), r"$\mathbf{x}_p$", color=GREEN, fontsize=11, weight="bold")
    # show the shift from origin
    ax_aff.quiver(0, 0, 0, *xp, color=GREEN, linewidth=2.0, arrow_length_ratio=0.14)
    ax_aff.scatter(0, 0, 0, color=SLATE, s=24, zorder=7)
    ax_aff.set_xlim(-2.2, 2.2)
    ax_aff.set_ylim(-2.2, 2.2)
    ax_aff.set_zlim(-2.2, 2.2)
    ax_aff.set_xlabel(r"$x$")
    ax_aff.set_ylabel(r"$y$")
    ax_aff.set_zlabel(r"$z$")
    ax_aff.set_title(r"Affine: $x+2y-z=1$", fontsize=12, weight="bold", pad=8)
    ax_aff.view_init(elev=20, azim=-55)
    ax_aff.grid(True, color=GRID, alpha=0.5)

    fig.suptitle("A subspace passes through the origin; an affine plane does not", fontsize=15, weight="bold")
    return fig


def figure_span_and_dependence() -> Figure:
    """Span of one vector is a line; two independent vectors fill the plane; a redundant third adds nothing."""
    v1 = np.array([1.0, 2.0])
    v2 = np.array([1.0, -0.5])
    v3 = v1 + v2  # redundant

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.2), constrained_layout=True)

    # Panel 1: span of one vector = line
    style_plane(axes[0], xlim=(-3.0, 3.0), ylim=(-3.0, 3.0), title=r"$\operatorname{span}(\mathbf{v}_1)$: a line")
    t = np.linspace(-1.4, 1.4, 100)
    axes[0].plot(v1[0] * t, v1[1] * t, color=BLUE, linewidth=3.0, zorder=2)
    arrow(axes[0], np.zeros(2), v1, BLUE, r"$\mathbf{v}_1$")
    axes[0].scatter(0, 0, color=SLATE, s=24, zorder=7)

    # Panel 2: span of two independent vectors = R^2
    style_plane(axes[1], xlim=(-3.0, 3.0), ylim=(-3.0, 3.0), title=r"$\operatorname{span}(\mathbf{v}_1,\mathbf{v}_2)$: the plane")
    # fill a parallelogram to suggest coverage
    for s in np.linspace(-1.2, 1.2, 13):
        axes[1].plot(v1[0] * s + v2[0] * t, v1[1] * s + v2[1] * t, color=GRID, linewidth=0.5, alpha=0.6, zorder=1)
        axes[1].plot(v1[0] * t + v2[0] * s, v1[1] * t + v2[1] * s, color=GRID, linewidth=0.5, alpha=0.6, zorder=1)
    arrow(axes[1], np.zeros(2), v1, BLUE, r"$\mathbf{v}_1$")
    arrow(axes[1], np.zeros(2), v2, ORANGE, r"$\mathbf{v}_2$")
    axes[1].scatter(0, 0, color=SLATE, s=24, zorder=7)

    # Panel 3: adding v3 = v1 + v2 adds nothing
    style_plane(axes[2], xlim=(-3.0, 3.0), ylim=(-3.0, 3.0), title=r"$\mathbf{v}_3=\mathbf{v}_1+\mathbf{v}_2$: no new direction")
    arrow(axes[2], np.zeros(2), v1, BLUE, r"$\mathbf{v}_1$")
    arrow(axes[2], v1, v3, ORANGE, r"$\mathbf{v}_2$")
    arrow(axes[2], np.zeros(2), v3, GREEN, r"$\mathbf{v}_3$")
    axes[2].plot([0, v3[0]], [0, v3[1]], color="white", linewidth=0, zorder=0)
    axes[2].scatter(0, 0, color=SLATE, s=24, zorder=7)
    axes[2].text(0.97, 0.06, r"$\operatorname{span}(\mathbf{v}_1,\mathbf{v}_2,\mathbf{v}_3)$" + "\n" + r"$=\operatorname{span}(\mathbf{v}_1,\mathbf{v}_2)$",
                 transform=axes[2].transAxes, ha="right", va="bottom", fontsize=10, color=SLATE,
                 bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": GRID})

    fig.suptitle("Span grows only when a vector provides a genuinely new direction", fontsize=15, weight="bold")
    return fig


def figure_basis_coordinates() -> Figure:
    """The same vector has different coordinates under the standard basis and a non-standard basis."""
    x = np.array([5.0, 1.0])
    e1, e2 = np.eye(2)
    b1 = np.array([1.0, 1.0])
    b2 = np.array([1.0, -1.0])

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.4), constrained_layout=True)

    # Left: standard basis
    style_plane(axes[0], xlim=(-0.8, 6.8), ylim=(-1.6, 6.0), title=r"Standard basis $\{\mathbf{e}_1,\mathbf{e}_2\}$")
    arrow(axes[0], np.zeros(2), e1, BLUE, r"$\mathbf{e}_1$")
    arrow(axes[0], np.zeros(2), e2, ORANGE, r"$\mathbf{e}_2$")
    arrow(axes[0], np.zeros(2), x, GREEN, r"$\mathbf{x}$")
    axes[0].text(0.97, 0.94, r"$[\mathbf{x}]_{\mathcal{E}}=(5,\,1)$",
                 transform=axes[0].transAxes, ha="right", va="top", fontsize=13, color=GREEN,
                 bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": GRID})
    axes[0].scatter(0, 0, color=SLATE, s=24, zorder=7)

    # Right: non-standard basis  b1=(1,1), b2=(1,-1)
    style_plane(axes[1], xlim=(-0.8, 6.8), ylim=(-1.6, 6.0), title=r"Non-standard basis $\{\mathbf{b}_1,\mathbf{b}_2\}$")
    arrow(axes[1], np.zeros(2), b1, BLUE, r"$\mathbf{b}_1$")
    arrow(axes[1], np.zeros(2), b2, ORANGE, r"$\mathbf{b}_2$")
    # show 3*b1 + 2*b2 head-to-tail
    arrow(axes[1], np.zeros(2), 3.0 * b1, BLUE, r"$3\mathbf{b}_1$")
    arrow(axes[1], 3.0 * b1, x, ORANGE, r"$2\mathbf{b}_2$")
    arrow(axes[1], np.zeros(2), x, GREEN, r"$\mathbf{x}$")
    axes[1].text(0.97, 0.94, r"$[\mathbf{x}]_{\mathcal{B}}=(3,\,2)$",
                 transform=axes[1].transAxes, ha="right", va="top", fontsize=13, color=GREEN,
                 bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": GRID})
    axes[1].scatter(0, 0, color=SLATE, s=24, zorder=7)

    fig.suptitle(r"The same vector $\mathbf{x}$ has different coordinates under different bases", fontsize=15, weight="bold")
    return fig


def figure_rank_nullity_spaces() -> Figure:
    """Input space R^n splits into null space; output space R^m contains the column space."""
    fig, ax = plt.subplots(figsize=(11.2, 5.6), constrained_layout=True)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.5)
    ax.axis("off")

    box = {"boxstyle": "round,pad=0.5", "facecolor": "white", "edgecolor": GRID, "linewidth": 1.7}
    dim_box = {"boxstyle": "round,pad=0.4", "facecolor": "#eff6ff", "edgecolor": BLUE, "linewidth": 1.8}

    # Input space R^n
    ax.add_patch(Rectangle((0.4, 1.0), 4.0, 4.5, facecolor=BLUE, alpha=0.08, edgecolor=BLUE, linewidth=2.0))
    ax.text(2.4, 5.8, r"Input space $\mathbb{R}^n$", ha="center", fontsize=13, color=BLUE, weight="bold")
    # Null space inside
    ax.add_patch(Rectangle((0.8, 1.4), 1.6, 3.7, facecolor=ORANGE, alpha=0.16, edgecolor=ORANGE, linewidth=2.0))
    ax.text(1.6, 3.25, r"$\mathcal{N}(A)$", ha="center", va="center", fontsize=14, color=ORANGE, weight="bold")
    ax.text(1.6, 2.6, r"$\dim=n-r$", ha="center", va="center", fontsize=11, color=ORANGE)
    ax.text(3.5, 3.25, r"complement", ha="center", va="center", fontsize=10, color=SLATE)
    ax.text(3.5, 2.6, r"$\dim=r$", ha="center", va="center", fontsize=11, color=SLATE)

    # Arrow: A maps input to output
    ax.annotate("", xy=(7.6, 3.25), xytext=(4.6, 3.25), arrowprops={"arrowstyle": "-|>", "color": SLATE, "lw": 2.5})
    ax.text(6.1, 3.9, r"$\mathbf{x}\mapsto A\mathbf{x}$", ha="center", fontsize=13, color=SLATE, weight="bold")
    ax.text(6.1, 2.5, r"$A\in\mathbb{R}^{m\times n}$", ha="center", fontsize=11, color=SLATE)

    # Output space R^m
    ax.add_patch(Rectangle((7.8, 1.0), 4.0, 4.5, facecolor=GREEN, alpha=0.08, edgecolor=GREEN, linewidth=2.0))
    ax.text(9.8, 5.8, r"Output space $\mathbb{R}^m$", ha="center", fontsize=13, color=GREEN, weight="bold")
    # Column space inside
    ax.add_patch(Rectangle((8.2, 1.4), 3.2, 3.7, facecolor=GREEN, alpha=0.18, edgecolor=GREEN, linewidth=2.0))
    ax.text(9.8, 3.6, r"$\mathcal{C}(A)$", ha="center", va="center", fontsize=14, color=GREEN, weight="bold")
    ax.text(9.8, 2.9, r"$\dim=r$", ha="center", va="center", fontsize=11, color=GREEN)
    ax.text(9.8, 2.1, r"$=\operatorname{rank}(A)$", ha="center", va="center", fontsize=10, color=GREEN)

    # Rank-nullity equation at bottom
    ax.text(6.5, 0.3, r"$\operatorname{rank}(A)+\dim\,\mathcal{N}(A)=n$",
           ha="center", va="center", fontsize=15, color=SLATE, bbox=dim_box)

    fig.suptitle("Rank-nullity balances the input freedom against the output reach", fontsize=15, weight="bold")
    return fig


def figure_four_spaces_summary() -> Figure:
    """Three questions about a matrix correspond to three different spaces."""
    fig, ax = plt.subplots(figsize=(10.0, 5.8), constrained_layout=True)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")

    box = {"boxstyle": "round,pad=0.55", "facecolor": "white", "edgecolor": GRID, "linewidth": 1.7}

    rows = (
        (5.8, r"Which $\mathbf{b}$ are reachable?", r"$\mathcal{C}(A)\subseteq\mathbb{R}^m$", BLUE, r"column space"),
        (3.5, r"Which $\mathbf{x}$ vanish?", r"$\mathcal{N}(A)\subseteq\mathbb{R}^n$", ORANGE, r"null space"),
        (1.2, r"How many output directions?", r"$\operatorname{rank}(A)=\dim\mathcal{C}(A)$", GREEN, r"rank"),
    )

    for y_pos, question, space, color, label in rows:
        ax.text(1.8, y_pos, question, ha="center", va="center", fontsize=12, color=SLATE, weight="bold", bbox=box)
        ax.annotate("", xy=(4.3, y_pos), xytext=(3.05, y_pos), arrowprops={"arrowstyle": "->", "color": GRID, "lw": 1.8})
        ax.text(6.2, y_pos, space, ha="center", va="center", fontsize=14, color=color, weight="bold", bbox=box)
        ax.text(9.0, y_pos, label, ha="center", va="center", fontsize=10.5, color=color)

    ax.text(0.4, 6.5, r"$A\in\mathbb{R}^{m\times n}$", ha="left", va="center", fontsize=14, color=SLATE, weight="bold")
    ax.annotate("", xy=(5.0, 5.0), xytext=(5.0, 6.2), arrowprops={"arrowstyle": "->", "color": "#94a3b8", "lw": 1.5})
    ax.annotate("", xy=(5.0, 2.7), xytext=(5.0, 4.6), arrowprops={"arrowstyle": "->", "color": "#94a3b8", "lw": 1.5})
    ax.annotate("", xy=(5.0, 0.4), xytext=(5.0, 2.1), arrowprops={"arrowstyle": "->", "color": "#94a3b8", "lw": 1.5})

    fig.suptitle("Three questions about a matrix lead to three different spaces", fontsize=15, weight="bold")
    return fig


def normalize_svg(path: Path) -> None:
    """Remove generator-introduced trailing whitespace from an SVG file."""
    content = path.read_text(encoding="utf-8")
    normalized = "\n".join(line.rstrip() for line in content.splitlines()) + "\n"
    path.write_text(normalized, encoding="utf-8")


def save_figure(figure: Figure, stem: str, formats: tuple[str, ...]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for file_format in formats:
        target = OUTPUT_DIR / f"{stem}.{file_format}"
        figure.savefig(
            target,
            dpi=180,
            bbox_inches="tight",
            metadata={"Creator": "Matplotlib", "Date": "2026-07-20"},
        )
        if file_format == "svg":
            normalize_svg(target)
        print(f"generated {target.relative_to(ROOT)}")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--format",
        choices=("svg", "png", "both"),
        default="svg",
        help="output format (default: svg)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    formats = ("svg", "png") if args.format == "both" else (args.format,)
    save_figure(figure_basis_transformation(), "basis-transformation", formats)
    save_figure(figure_column_combination(), "column-combination", formats)
    save_figure(figure_linear_vs_affine(), "linear-vs-affine", formats)
    save_figure(figure_matrix_anatomy(), "matrix-anatomy", formats)
    save_figure(figure_row_column_views(), "row-column-views", formats)
    save_figure(figure_rectangular_linear_map(), "rectangular-linear-map", formats)
    save_figure(figure_batch_layout(), "batch-layout", formats)
    save_figure(figure_composition_spaces(), "composition-spaces", formats)
    save_figure(figure_matrix_product_entry(), "matrix-product-entry", formats)
    save_figure(figure_product_columns(), "product-columns", formats)
    save_figure(figure_noncommutativity(), "matrix-noncommutativity", formats)
    save_figure(figure_system_matrix_correspondence(), "system-matrix-correspondence", formats)
    save_figure(figure_row_operation_same_solution(), "row-operation-same-solution", formats)
    save_figure(figure_gaussian_elimination_steps(), "gaussian-elimination-steps", formats)
    save_figure(figure_parameter_system_outcomes(), "parameter-system-outcomes", formats)
    save_figure(figure_affine_solution_translation(), "affine-solution-translation", formats)
    save_figure(figure_two_by_two_determinant(), "two-by-two-determinant", formats)
    save_figure(figure_permutation_selections(), "permutation-selections", formats)
    save_figure(figure_inversion_crossings(), "inversion-crossings", formats)
    save_figure(figure_determinant_orientation(), "determinant-orientation", formats)
    save_figure(figure_invertibility_equivalences(), "invertibility-equivalences", formats)
    save_figure(figure_inverse_solves_basis(), "inverse-solves-basis", formats)
    save_figure(figure_cramer_column_replacement(), "cramer-column-replacement", formats)
    save_figure(figure_singular_system_two_rhs(), "singular-system-two-rhs", formats)
    save_figure(figure_rank_minor_pivots(), "rank-minor-pivots", formats)
    save_figure(figure_rank_consistency_criterion(), "rank-consistency-criterion", formats)
    save_figure(figure_subspace_vs_affine(), "subspace-vs-affine", formats)
    save_figure(figure_span_and_dependence(), "span-and-dependence", formats)
    save_figure(figure_basis_coordinates(), "basis-coordinates", formats)
    save_figure(figure_rank_nullity_spaces(), "rank-nullity-spaces", formats)
    save_figure(figure_four_spaces_summary(), "four-spaces-summary", formats)


if __name__ == "__main__":
    main()
