"""验证《行列式、可逆性与一般线性方程组》中的核心结论。"""

from itertools import combinations

import numpy as np


def validate_matrix(matrix: np.ndarray) -> None:
    if matrix.ndim != 2:
        raise ValueError("matrix 必须是二维数组")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("matrix 必须只含有限数值")


def numerical_rank(matrix: np.ndarray, relative_tolerance: float = 1e-12) -> int:
    """按相对于最大奇异值的阈值估计数值秩。"""
    validate_matrix(matrix)
    if not np.isfinite(relative_tolerance) or not 0 < relative_tolerance < 1:
        raise ValueError("relative_tolerance 必须是 0 与 1 之间的有限数")

    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if singular_values.size == 0 or singular_values[0] == 0:
        return 0
    threshold = relative_tolerance * singular_values[0]
    return int(np.count_nonzero(singular_values > threshold))


def cramers_rule(
    matrix: np.ndarray,
    rhs: np.ndarray,
    relative_tolerance: float = 1e-12,
) -> np.ndarray:
    """用 Cramer 法则求小型数值非奇异方阵方程组。"""
    validate_matrix(matrix)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Cramer 法则要求方阵")
    if rhs.shape != (matrix.shape[0],):
        raise ValueError("rhs 的长度必须等于矩阵行数")
    if not np.all(np.isfinite(rhs)):
        raise ValueError("rhs 必须只含有限数值")
    if numerical_rank(matrix, relative_tolerance) < matrix.shape[0]:
        raise ValueError("矩阵在给定相对容差下视为奇异，不能使用 Cramer 法则")

    denominator = float(np.linalg.det(matrix))

    solution = np.empty(matrix.shape[1], dtype=float)
    for column in range(matrix.shape[1]):
        replaced = matrix.astype(float).copy()
        replaced[:, column] = rhs
        solution[column] = np.linalg.det(replaced) / denominator
    return solution


def largest_nonzero_minor(
    matrix: np.ndarray,
    relative_tolerance: float = 1e-12,
) -> tuple[int, tuple[int, ...], tuple[int, ...], float]:
    """枚举小矩阵，返回数值满秩的最大方形子矩阵见证。"""
    validate_matrix(matrix)

    row_count, column_count = matrix.shape
    for order in range(min(row_count, column_count), 0, -1):
        for rows in combinations(range(row_count), order):
            for columns in combinations(range(column_count), order):
                submatrix = matrix[np.ix_(rows, columns)]
                if numerical_rank(submatrix, relative_tolerance) == order:
                    determinant = float(np.linalg.det(submatrix))
                    return order, rows, columns, determinant
    return 0, (), (), 0.0


def classify_by_rank(
    matrix: np.ndarray,
    rhs: np.ndarray,
    relative_tolerance: float = 1e-12,
) -> tuple[str, int, int]:
    """按照系数矩阵与增广矩阵的秩分类实线性方程组。"""
    validate_matrix(matrix)
    if rhs.shape != (matrix.shape[0],):
        raise ValueError("rhs 的长度必须等于矩阵行数")
    if not np.all(np.isfinite(rhs)):
        raise ValueError("rhs 必须只含有限数值")
    coefficient_rank = numerical_rank(matrix, relative_tolerance)
    augmented_rank = numerical_rank(
        np.column_stack((matrix, rhs)),
        relative_tolerance,
    )

    if coefficient_rank != augmented_rank:
        outcome = "no solution"
    elif coefficient_rank == matrix.shape[1]:
        outcome = "unique solution"
    else:
        outcome = "infinitely many solutions"
    return outcome, coefficient_rank, augmented_rank


def main() -> None:
    invertible = np.array([[2.0, 1.0], [1.0, 3.0]])
    rhs = np.array([5.0, 7.0])
    solution = cramers_rule(invertible, rhs)
    np.testing.assert_allclose(solution, [1.6, 1.8])
    np.testing.assert_allclose(solution, np.linalg.solve(invertible, rhs))

    inverse = np.linalg.solve(invertible, np.eye(2))
    np.testing.assert_allclose(invertible @ inverse, np.eye(2), atol=1e-12)
    np.testing.assert_allclose(inverse @ invertible, np.eye(2), atol=1e-12)
    assert not np.isclose(np.linalg.det(invertible), 0.0)
    assert numerical_rank(invertible) == 2

    scaled_well_conditioned = np.diag([1e-7, 1e-7])
    scaled_rhs = np.array([1e-7, 2e-7])
    assert numerical_rank(scaled_well_conditioned) == 2
    assert largest_nonzero_minor(scaled_well_conditioned)[0] == 2
    np.testing.assert_allclose(
        cramers_rule(scaled_well_conditioned, scaled_rhs),
        [1.0, 2.0],
    )

    singular = np.array([[1.0, 2.0], [2.0, 4.0]])
    homogeneous_direction = np.array([-2.0, 1.0])
    np.testing.assert_allclose(singular @ homogeneous_direction, np.zeros(2))

    on_image = np.array([3.0, 6.0])
    off_image = np.array([3.0, 7.0])
    assert classify_by_rank(singular, on_image) == (
        "infinitely many solutions",
        1,
        1,
    )
    assert classify_by_rank(singular, off_image) == ("no solution", 1, 2)

    rectangular = np.array(
        [
            [1.0, 2.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 1.0],
            [1.0, 3.0, 1.0, 2.0],
        ]
    )
    order, rows, columns, determinant = largest_nonzero_minor(rectangular)
    assert order == 2
    assert rows == (0, 1)
    assert columns == (0, 1)
    np.testing.assert_allclose(determinant, 1.0)
    assert np.linalg.matrix_rank(rectangular) == order

    rectangular_rhs = np.array([1.0, 2.0, 3.0])
    assert classify_by_rank(rectangular, rectangular_rhs) == (
        "infinitely many solutions",
        2,
        2,
    )

    print(f"Cramer solution = {solution}")
    print(f"det(invertible) = {np.linalg.det(invertible):g}")
    print(f"largest nonzero minor order = {order}")
    print("可逆性、Cramer 法则、子式秩与相容性判据全部验证通过。")


if __name__ == "__main__":
    main()
