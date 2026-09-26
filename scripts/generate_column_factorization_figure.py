#!/usr/bin/env python3
"""Generate the A = CR column-factorization example and verify it exactly."""

from __future__ import annotations

import argparse
import html
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch

from generate_linear_algebra_figures import BLUE, GREEN, ORANGE, SLATE, configure_matplotlib, normalize_svg

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "images" / "linear-algebra"
COLORS = (BLUE, ORANGE, GREEN, "#7c3aed")
A = np.array([[1, 0, 1, 2], [0, 1, 1, -1], [1, 1, 2, 1]], dtype=object)
C = A[:, :2]
R = np.array([[1, 0, 1, 2], [0, 1, 1, -1]], dtype=object)


def verify_example() -> None:
    """Use Python integers, not numerical rank or a floating-point tolerance."""
    assert C.shape == (3, 2) and R.shape == (2, 4)
    assert np.array_equal(C[:2, :], np.eye(2, dtype=object))
    assert np.array_equal(A[:, 2], A[:, 0] + A[:, 1])
    assert np.array_equal(A[:, 3], 2 * A[:, 0] - A[:, 1])
    assert np.array_equal(C @ R, A)
    assert np.array_equal(A[2, :], A[0, :] + A[1, :])
    x = np.array([1, 2, -1, 1], dtype=object)
    assert np.array_equal(R @ x, [2, 0])
    assert np.array_equal(A @ x, [2, 0, 2])
    assert np.array_equal(C @ (R @ x), A @ x)
    for x in ([2, 0, 0, 0], [1, -1, 1, 0]):
        assert np.array_equal(A @ np.array(x, dtype=object), [2, 0, 2])


def draw_matrix(ax: Axes, data: np.ndarray, center: float, name: str, subtitle: str) -> None:
    rows, columns = data.shape
    spacing = 0.53
    left = center - columns * spacing / 2
    bottom, top = 4.05, 5.48
    ax.text(center, 5.98, name, ha="center", fontsize=23, color=SLATE)
    ax.text(center, 5.65, subtitle, ha="center", fontsize=12, color=SLATE)
    for j in range(columns):
        x = left + (j + 0.5) * spacing
        ax.add_patch(FancyBboxPatch(
            (x - 0.21, bottom + 0.07), 0.42, top - bottom - 0.14,
            boxstyle="round,pad=0.02", facecolor=COLORS[j], edgecolor="none", alpha=0.09,
        ))
        for i in range(rows):
            y = (bottom + top) / 2 + ((rows - 1) / 2 - i) * 0.40
            ax.text(x, y, str(data[i, j]), ha="center", va="center", fontsize=21, color=COLORS[j])
    for edge, inward in ((left - 0.08, 0.12), (left + columns * spacing + 0.08, -0.12)):
        ax.plot([edge + inward, edge, edge, edge + inward], [top, top, bottom, bottom], color=SLATE, lw=1.6)
    ax.text(center, 3.80, f"{rows} × {columns}", ha="center", fontsize=13, color=SLATE)


def figure_column_factorization() -> Figure:
    verify_example()
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(11.4, 6.8))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.03, top=0.97)
    ax.set(xlim=(0, 11.4), ylim=(0, 6.8))
    ax.axis("off")
    ax.text(5.7, 6.48, "Independent columns + coordinates = all columns", ha="center", fontsize=19, weight="bold", color=SLATE)
    draw_matrix(ax, A, 1.9, "$A$", "Original columns")
    draw_matrix(ax, C, 5.6, "$C$", "Keep the first two columns")
    draw_matrix(ax, R, 9.25, "$R$", "Coordinates of each column")
    ax.text(3.82, 4.77, "$=$", ha="center", va="center", fontsize=28, color=SLATE)
    ax.text(7.12, 4.77, "$\\times$", ha="center", va="center", fontsize=25, color=SLATE)
    ax.text(3.65, 3.29, "$a_3=a_1+a_2$", ha="center", fontsize=18, color=GREEN)
    ax.text(7.9, 3.29, "$a_4=2a_1-a_2$", ha="center", fontsize=18, color=COLORS[3])
    ax.plot([0.4, 11.0], [2.93, 2.93], color="#cbd5e1", lw=1)
    boxes = (
        (0.45, 2.25, "Four input weights", "$x\\in\\mathbb{R}^4$"),
        (4.35, 2.55, "Two coefficients", "$(s,t)^{\\mathsf{T}}$"),
        (8.42, 2.6, "Output in $C(A)$", "$(s,t,s+t)^{\\mathsf{T}}$"),
    )
    for left, width, title, formula in boxes:
        ax.add_patch(FancyBboxPatch(
            (left, 1.28), width, 1.12, boxstyle="round,pad=0.08",
            edgecolor="#cbd5e1", facecolor="#f8fafc", lw=1.3,
        ))
        ax.text(left + width / 2, 2.09, title, ha="center", fontsize=13, color=SLATE)
        ax.text(left + width / 2, 1.56, formula, ha="center", fontsize=19, color=SLATE)
    for start, end, label, color in ((2.88, 4.15, "$R$", ORANGE), (7.07, 8.22, "$C$", BLUE)):
        ax.annotate("", (end, 1.82), (start, 1.82), arrowprops={"arrowstyle": "-|>", "lw": 2.5, "color": color})
        ax.text((start + end) / 2, 2.1, label, ha="center", fontsize=19, color=color)
    ax.text(5.63, 0.94, "$s=x_1+x_3+2x_4$", ha="center", fontsize=15, color=SLATE)
    ax.text(5.63, 0.57, "$t=x_2+x_3-x_4$", ha="center", fontsize=15, color=SLATE)
    ax.text(9.72, 0.8, "$y_3=y_1+y_2$", ha="center", fontsize=17, color=GREEN)
    ax.text(1.58, 0.76, "$Ax=C(Rx)$", ha="center", fontsize=19, color=SLATE)
    ax.text(5.7, 0.13, "The output fills a plane in three-dimensional space.", ha="center", fontsize=13, color=SLATE)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--format", choices=("svg", "png"), default="svg")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig = figure_column_factorization()
    target = args.output_dir / f"column-factorization-cr.{args.format}"
    fig.savefig(target, dpi=180, metadata={"Creator": "Matplotlib", "Date": None} if args.format == "svg" else None)
    plt.close(fig)
    if args.format == "svg":
        content = target.read_text(encoding="utf-8")
        end = content.index(">", content.index("<svg")) + 1
        metadata = (
            "\n<title>A = CR: independent columns and their coordinates</title>"
            f"\n<desc>{html.escape('A = [[1,0,1,2],[0,1,1,-1],[1,1,2,1]]. C contains the first two columns. R = [[1,0,1,2],[0,1,1,-1]]. First apply R to obtain (s,t), then C to obtain (s,t,s+t). All outputs lie in the plane y3=y1+y2.')}</desc>"
        )
        target.write_text(content[:end] + metadata + content[end:], encoding="utf-8")
        normalize_svg(target)
    print(f"Exact checks passed; generated {target}")


if __name__ == "__main__":
    main()
