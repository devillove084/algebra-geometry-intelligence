"""验证《线性方程组与 Gaussian 消元》中的核心计算。"""

import numpy as np


def parameter_system(lambda_value: float, mu_value: float) -> tuple[np.ndarray, np.ndarray]:
    """构造笔记中的三元参数方程组。"""
    coefficient = np.array(
        [
            [1.0, 1.0, 1.0],
            [2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0 + lambda_value],
        ]
    )
    rhs = np.array([2.0, 5.0, 3.0 + mu_value])
    return coefficient, rhs


def row_echelon(
    augmented_matrix: np.ndarray,
    tolerance: float = 1e-12,
) -> tuple[np.ndarray, list[int]]:
    """使用部分选主元把增广矩阵化为行阶梯形。"""
    if augmented_matrix.ndim != 2 or augmented_matrix.shape[1] < 2:
        raise ValueError("augmented_matrix 必须是至少含两列的二维数组")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance 必须是有限正数")
    if not np.all(np.isfinite(augmented_matrix)):
        raise ValueError("augmented_matrix 必须只含有限数值")

    matrix = augmented_matrix.astype(float).copy()
    row_count, column_count = matrix.shape
    variable_count = column_count - 1
    pivot_columns: list[int] = []
    pivot_row = 0

    for column in range(column_count):
        if pivot_row >= row_count:
            break

        candidates = np.abs(matrix[pivot_row:, column])
        selected_row = pivot_row + int(np.argmax(candidates))
        if abs(matrix[selected_row, column]) <= tolerance:
            continue

        matrix[[pivot_row, selected_row]] = matrix[[selected_row, pivot_row]]
        for row in range(pivot_row + 1, row_count):
            factor = matrix[row, column] / matrix[pivot_row, column]
            matrix[row] -= factor * matrix[pivot_row]

        matrix[np.abs(matrix) <= tolerance] = 0.0
        if column < variable_count:
            pivot_columns.append(column)
        pivot_row += 1

    return matrix, pivot_columns


def classify_echelon(
    echelon: np.ndarray,
    pivot_columns: list[int],
    tolerance: float = 1e-12,
) -> str:
    """在给定容差下，根据矛盾行和自由变量分类浮点方程组。"""
    if echelon.ndim != 2 or echelon.shape[1] < 2:
        raise ValueError("echelon 必须是增广矩阵")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance 必须是有限正数")

    variable_count = echelon.shape[1] - 1
    if pivot_columns != sorted(set(pivot_columns)):
        raise ValueError("pivot_columns 必须严格递增且不能重复")
    if any(column < 0 or column >= variable_count for column in pivot_columns):
        raise ValueError("pivot_columns 只能引用系数列")

    coefficient = echelon[:, :-1]
    rhs = echelon[:, -1]
    contradictory = np.any(
        np.all(np.abs(coefficient) <= tolerance, axis=1)
        & (np.abs(rhs) > tolerance)
    )
    if contradictory:
        return "no solution"
    if len(pivot_columns) == variable_count:
        return "unique solution"
    return "infinitely many solutions"


def main() -> None:
    initial_augmented = np.array(
        [
            [1.0, 1.0, 1.0, 2.0],
            [2.0, 3.0, 4.0, 5.0],
            [1.0, 2.0, 4.0, 5.0],
        ]
    )
    hand_echelon = initial_augmented.copy()
    hand_echelon[1] -= 2.0 * hand_echelon[0]
    hand_echelon[2] -= hand_echelon[0]
    hand_echelon[2] -= hand_echelon[1]

    expected_echelon = np.array(
        [
            [1.0, 1.0, 1.0, 2.0],
            [0.0, 1.0, 2.0, 1.0],
            [0.0, 0.0, 1.0, 2.0],
        ]
    )
    np.testing.assert_allclose(hand_echelon, expected_echelon)

    expected_classes = {
        (1.0, 2.0): "unique solution",
        (0.0, 0.0): "infinitely many solutions",
        (0.0, 1.0): "no solution",
    }
    for parameters, expected_class in expected_classes.items():
        coefficient, rhs = parameter_system(*parameters)
        augmented = np.column_stack((coefficient, rhs))
        echelon, pivots = row_echelon(augmented)
        actual_class = classify_echelon(echelon, pivots)
        assert actual_class == expected_class

    coefficient_unique, rhs_unique = parameter_system(1.0, 2.0)
    unique_solution = np.linalg.solve(coefficient_unique, rhs_unique)
    np.testing.assert_allclose(unique_solution, [3.0, -3.0, 2.0])

    coefficient_many, rhs_many = parameter_system(0.0, 0.0)
    particular = np.array([1.0, 1.0, 0.0])
    direction = np.array([1.0, -2.0, 1.0])
    np.testing.assert_allclose(coefficient_many @ particular, rhs_many)
    np.testing.assert_allclose(coefficient_many @ direction, np.zeros(3))
    for parameter in (-2.0, 0.0, 1.5):
        candidate = particular + parameter * direction
        np.testing.assert_allclose(coefficient_many @ candidate, rhs_many)

    coefficient_none, rhs_none = parameter_system(0.0, 1.0)
    none_echelon, none_pivots = row_echelon(
        np.column_stack((coefficient_none, rhs_none))
    )
    assert classify_echelon(none_echelon, none_pivots) == "no solution"

    # 增广列也参与阶梯化：矛盾行应被移到零行上方。
    contradiction_after_zero = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    edge_echelon, edge_pivots = row_echelon(contradiction_after_zero)
    np.testing.assert_allclose(edge_echelon[0], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(edge_echelon[1], [0.0, 0.0, 0.0])
    assert classify_echelon(edge_echelon, edge_pivots) == "no solution"

    # 同一个数值系统在不同容差下可能得到不同分类，这不等于精确代数结论改变。
    tiny_system = np.array([[1e-13, 1e-13]])
    tiny_echelon, tiny_pivots = row_echelon(tiny_system, tolerance=1e-15)
    assert classify_echelon(tiny_echelon, tiny_pivots, tolerance=1e-15) == "unique solution"

    for invalid_tolerance in (0.0, -1.0, float("nan")):
        try:
            row_echelon(initial_augmented, tolerance=invalid_tolerance)
        except ValueError:
            pass
        else:
            raise AssertionError("非法 tolerance 应触发 ValueError")

    print("手算阶梯形：")
    print(hand_echelon)
    print(f"唯一解：{unique_solution}")
    print("唯一解、无穷多解、无解和特解加齐次解全部验证通过。")


if __name__ == "__main__":
    main()
