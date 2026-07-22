"""验证《矩阵与线性映射》中的核心计算。"""

import numpy as np


def apply_linear_map(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """按列向量约定计算矩阵诱导的线性映射。"""
    if matrix.ndim != 2:
        raise ValueError("matrix 必须是二维数组")
    if vector.ndim != 1:
        raise ValueError("vector 必须是一维数组")
    if matrix.shape[1] != vector.shape[0]:
        raise ValueError("矩阵列数必须等于向量维数")
    return matrix @ vector


def apply_affine_batch(
    samples: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    """对按行存储的一批样本应用同一个仿射映射。"""
    if samples.ndim != 2 or weight.ndim != 2 or bias.ndim != 1:
        raise ValueError("samples、weight、bias 的维数不符合约定")
    if samples.shape[1] != weight.shape[1]:
        raise ValueError("样本特征数必须等于权重矩阵列数")
    if bias.shape[0] != weight.shape[0]:
        raise ValueError("偏置维数必须等于权重矩阵行数")
    return samples @ weight.T + bias


def main() -> None:
    matrix = np.array(
        [
            [1.0, 2.0],
            [-1.0, 3.0],
            [4.0, 0.0],
        ]
    )
    x = np.array([2.0, -1.0])
    z = np.array([-1.0, 3.0])

    y = apply_linear_map(matrix, x)
    row_view = np.array([row @ x for row in matrix])
    column_view = x[0] * matrix[:, 0] + x[1] * matrix[:, 1]

    np.testing.assert_allclose(y, [0.0, -5.0, 8.0])
    np.testing.assert_allclose(row_view, y)
    np.testing.assert_allclose(column_view, y)

    alpha = 1.5
    beta = -0.25
    left = apply_linear_map(matrix, alpha * x + beta * z)
    right = alpha * apply_linear_map(matrix, x) + beta * apply_linear_map(matrix, z)
    np.testing.assert_allclose(left, right)

    samples = np.array(
        [
            [2.0, -1.0],
            [0.0, 1.0],
            [-2.0, 0.5],
        ]
    )
    bias = np.array([0.5, -1.0, 2.0])
    batch_output = apply_affine_batch(samples, matrix, bias)
    expected = np.stack([apply_linear_map(matrix, sample) + bias for sample in samples])
    np.testing.assert_allclose(batch_output, expected)

    print(f"matrix.shape = {matrix.shape}")
    print(f"x.shape = {x.shape}")
    print(f"matrix @ x = {y}")
    print(f"batch_output.shape = {batch_output.shape}")
    print("行视角、列视角、线性性与批量仿射计算全部验证通过。")


if __name__ == "__main__":
    main()
