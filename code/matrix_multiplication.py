"""验证《矩阵乘法与线性映射复合》中的核心计算。"""

import numpy as np


def multiply_by_entries(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """按分量定义计算两个二维矩阵的乘积。"""
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("left 和 right 必须都是二维数组")
    if left.shape[1] != right.shape[0]:
        raise ValueError("左矩阵列数必须等于右矩阵行数")

    product = np.zeros((left.shape[0], right.shape[1]), dtype=np.result_type(left, right))
    for i in range(left.shape[0]):
        for j in range(right.shape[1]):
            for k in range(left.shape[1]):
                product[i, j] += left[i, k] * right[k, j]
    return product


def compose_affine(
    outer_weight: np.ndarray,
    outer_bias: np.ndarray,
    inner_weight: np.ndarray,
    inner_bias: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """合并先 inner、后 outer 的两个仿射映射。"""
    if outer_weight.ndim != 2 or inner_weight.ndim != 2:
        raise ValueError("权重必须是二维数组")
    if outer_bias.ndim != 1 or inner_bias.ndim != 1:
        raise ValueError("偏置必须是一维数组")
    if outer_weight.shape[1] != inner_weight.shape[0]:
        raise ValueError("两个仿射映射的中间维数不匹配")
    if inner_bias.shape[0] != inner_weight.shape[0]:
        raise ValueError("inner_bias 的维数与 inner_weight 输出不匹配")
    if outer_bias.shape[0] != outer_weight.shape[0]:
        raise ValueError("outer_bias 的维数与 outer_weight 输出不匹配")

    combined_weight = outer_weight @ inner_weight
    combined_bias = outer_weight @ inner_bias + outer_bias
    return combined_weight, combined_bias


def main() -> None:
    matrix_a = np.array(
        [
            [1.0, 2.0, 0.0],
            [-1.0, 0.0, 1.0],
        ]
    )
    matrix_b = np.array(
        [
            [1.0, 0.0, 2.0, -1.0],
            [0.0, 1.0, -1.0, 2.0],
            [2.0, -1.0, 0.0, 1.0],
        ]
    )
    x = np.array([1.0, 2.0, -1.0, 1.0])

    product = matrix_a @ matrix_b
    product_by_entries = multiply_by_entries(matrix_a, matrix_b)
    intermediate = matrix_b @ x
    sequential = matrix_a @ intermediate
    combined = product @ x

    expected_product = np.array(
        [
            [1.0, 2.0, 0.0, 3.0],
            [1.0, -1.0, -2.0, 2.0],
        ]
    )

    np.testing.assert_allclose(product, expected_product)
    np.testing.assert_allclose(product_by_entries, product)
    np.testing.assert_allclose(intermediate, [-2.0, 5.0, 1.0])
    np.testing.assert_allclose(sequential, [8.0, 3.0])
    np.testing.assert_allclose(combined, sequential)
    np.testing.assert_allclose(product.T, matrix_b.T @ matrix_a.T)

    sample_batch = np.stack([x, -x, 0.5 * x])
    batch_sequential = (sample_batch @ matrix_b.T) @ matrix_a.T
    batch_combined = sample_batch @ product.T
    np.testing.assert_allclose(batch_sequential, batch_combined)

    inner_bias = np.array([0.5, -1.0, 2.0])
    outer_bias = np.array([1.0, -0.5])
    combined_weight, combined_bias = compose_affine(
        matrix_a,
        outer_bias,
        matrix_b,
        inner_bias,
    )
    affine_sequential = (sample_batch @ matrix_b.T + inner_bias) @ matrix_a.T + outer_bias
    affine_combined = sample_batch @ combined_weight.T + combined_bias
    np.testing.assert_allclose(affine_sequential, affine_combined)

    print(f"matrix_a.shape = {matrix_a.shape}")
    print(f"matrix_b.shape = {matrix_b.shape}")
    print(f"product.shape = {product.shape}")
    print(f"matrix_b @ x = {intermediate}")
    print(f"matrix_a @ (matrix_b @ x) = {sequential}")
    print("分量定义、复合、转置、批量布局与仿射合并全部验证通过。")


if __name__ == "__main__":
    main()
