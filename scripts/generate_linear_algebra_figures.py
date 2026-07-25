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


if __name__ == "__main__":
    main()
