"""验证《排列、逆序数与行列式》中的基本定义和性质。"""

from itertools import permutations

import numpy as np


def inversion_pairs(permutation: tuple[int, ...]) -> list[tuple[int, int]]:
    """返回一行记号排列中的全部零基位置逆序对。"""
    expected = set(range(1, len(permutation) + 1))
    if set(permutation) != expected or len(set(permutation)) != len(permutation):
        raise ValueError("permutation 必须恰好包含 1 到 n")

    pairs: list[tuple[int, int]] = []
    for i in range(len(permutation)):
        for j in range(i + 1, len(permutation)):
            if permutation[i] > permutation[j]:
                pairs.append((i, j))
    return pairs


def permutation_sign(permutation: tuple[int, ...]) -> int:
    """根据逆序数奇偶性返回排列符号。"""
    return -1 if len(inversion_pairs(permutation)) % 2 else 1


def determinant_leibniz(matrix: np.ndarray) -> float:
    """使用 Leibniz 公式计算小型方阵的行列式。"""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix 必须是方阵")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("matrix 必须只含有限数值")

    order = matrix.shape[0]
    total = 0.0
    for zero_based_permutation in permutations(range(order)):
        one_based_permutation = tuple(index + 1 for index in zero_based_permutation)
        term = float(permutation_sign(one_based_permutation))
        for row, column in enumerate(zero_based_permutation):
            term *= matrix[row, column]
        total += term
    return total


def main() -> None:
    sigma = (3, 1, 4, 2)
    pairs = inversion_pairs(sigma)
    assert pairs == [(0, 1), (0, 3), (2, 3)]
    assert permutation_sign(sigma) == -1

    matrix_two = np.array([[2.0, 1.0], [4.0, 3.0]])
    matrix_three = np.array(
        [
            [1.0, 2.0, 0.0],
            [-1.0, 1.0, 3.0],
            [2.0, 0.0, 1.0],
        ]
    )

    determinant_two = determinant_leibniz(matrix_two)
    determinant_three = determinant_leibniz(matrix_three)
    np.testing.assert_allclose(determinant_two, 2.0)
    np.testing.assert_allclose(determinant_three, 15.0)
    np.testing.assert_allclose(determinant_three, np.linalg.det(matrix_three))

    row_swapped = matrix_three[[1, 0, 2]]
    row_added = matrix_three.copy()
    row_added[1] += 5.0 * row_added[0]
    row_scaled = matrix_three.copy()
    row_scaled[2] *= -2.0

    np.testing.assert_allclose(determinant_leibniz(row_swapped), -determinant_three)
    np.testing.assert_allclose(determinant_leibniz(row_added), determinant_three)
    np.testing.assert_allclose(determinant_leibniz(row_scaled), -2.0 * determinant_three)
    np.testing.assert_allclose(
        determinant_leibniz(matrix_three.T),
        determinant_three,
    )

    triangular = np.array(
        [
            [2.0, 1.0, -1.0],
            [0.0, 3.0, 4.0],
            [0.0, 0.0, -5.0],
        ]
    )
    np.testing.assert_allclose(
        determinant_leibniz(triangular),
        np.prod(np.diag(triangular)),
    )

    print(f"sigma = {sigma}")
    print(f"inversion pairs (zero-based positions) = {pairs}")
    print(f"det(matrix_two) = {determinant_two}")
    print(f"det(matrix_three) = {determinant_three}")
    print("排列符号、Leibniz 公式和初等行变换性质全部验证通过。")


if __name__ == "__main__":
    main()
