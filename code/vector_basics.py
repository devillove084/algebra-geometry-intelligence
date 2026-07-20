"""验证《标量与向量》中的基本运算。"""

import numpy as np


def main() -> None:
    x = np.array([1.0, 2.0, -1.0])
    y = np.array([3.0, 0.0, 4.0])
    alpha = 2.0
    beta = -0.5

    linear_combination = alpha * x + beta * y
    dot_product = x @ y
    norm = np.linalg.norm(x)

    np.testing.assert_allclose(linear_combination, [0.5, 4.0, -4.0])
    np.testing.assert_allclose(dot_product, -1.0)
    np.testing.assert_allclose(norm, np.sqrt(6.0))

    print(f"x.shape = {x.shape}")
    print(f"alpha * x + beta * y = {linear_combination}")
    print(f"x dot y = {dot_product}")
    print(f"||x||_2 = {norm:.6f}")
    print("全部验证通过。")


if __name__ == "__main__":
    main()
