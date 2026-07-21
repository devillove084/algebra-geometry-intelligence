# Algebra, Geometry, and Intelligence

## 代数、几何与智能

[![Read the notes](https://img.shields.io/badge/read-GitHub%20Pages-6f4e37?style=for-the-badge)](https://devillove084.github.io/algebra-geometry-intelligence/)
[![Deploy](https://github.com/devillove084/algebra-geometry-intelligence/actions/workflows/publish.yml/badge.svg)](https://github.com/devillove084/algebra-geometry-intelligence/actions/workflows/publish.yml)

这是一份持续整理和校正的个人学习笔记，记录一条从较少前置知识出发、面向现代学习系统的选择性数学路线。它不是完整教材，也不承诺覆盖相关学科的全部内容；现阶段主要记录三件事：

1. 从定义出发推导模型中真正使用的数学对象；
2. 用代数、几何与概率工具分析表示、训练和对齐；
3. 把架构直觉变成带假设、可验证、可反驳的问题。

## 数学路线图

![Algebra, Geometry, and Intelligence 的数学依赖图](images/mathematical-roadmap.svg)

箭头表示**前置依赖**，不是必须逐项走完的唯一顺序。浅色节点是共同基础，绿色节点是数学核心，红色节点是按研究目的选择的高级路线。

## 核心路线

我计划沿这条主线，从数学语言逐步走到能够推导简化的语言模型和强化学习算法：

```text
数、集合、函数与证明
  → 向量、矩阵、线性映射与内积
  → 极限、导数、偏导与链式法则
  → 随机变量、条件概率与期望
  → 矩阵微积分、信息论与优化
  → MLP、归一化、残差与反向传播
  → embedding、attention、RoPE 与 Transformer
  → MDP、Bellman 方程、Policy Gradient 与 PPO
  → 语言建模、偏好学习、奖励与验证
```

目标不是从数学“必然推出”某一种模型，而是能够区分：哪些结论来自定义，哪些结构来自设计，哪些行为只是经验观察。

## 高级路线

### A. 高等代数与多线性结构

```text
线性空间与线性映射
  → 像、核、商空间、对偶与伴随
  → Gram 矩阵、半正定矩阵与二次型
  → 谱定理、SVD、伪逆与主角度
  → 多线性映射、张量积、缩并与 einsum
  → Kronecker 结构、CP/Tucker/TT 分解
  → 条件数、随机化 SVD 与数值稳定
```

用于理解低秩更新、LoRA、表示子空间、张量化参数、多模态线性对齐以及大规模矩阵计算。Jordan 标准形与多项式理论会讲清结构意义，但不以机械手算为主。

### B. 抽象代数、对称性与表示论

```text
代数结构、同态与商结构
  → 群与群作用
  → 轨道、稳定子与置换群
  → 线性表示、子表示与交织算子
  → 不变性、等变性与群平均
  → 循环群、Fourier 表示与 RoPE
  → Lie 群、Lie 代数、SO(3) 与 SE(3)
  → 张量积表示、球谐与几何等变网络
```

用于严格描述序列置换、平移、旋转和三维几何中的对称性。完整的环论、域扩张、Galois 理论与同调代数不作为共同主线；只有出现明确模型问题时才进入专题。

### C. 高维几何与高维概率

```text
随机向量与各向同性
  → 浓缩不等式与范数集中
  → 高维球面与随机方向近似正交
  → Johnson–Lindenstrauss 随机投影
  → 距离集中、hubness 与近邻检索
  → 有效秩、内在维数与表示各向异性
  → 覆盖数、度量熵与流形假设
```

用于分析 attention 的尺度、embedding 检索和表示维度。每个结论都必须注明分布、独立性和各向同性假设；训练后的表示通常不会自动满足随机模型的条件。

### D. 谱理论与随机矩阵

```text
Gram 矩阵与协方差
  → 奇异谱、谱范数与 stable rank
  → Wigner 与 Marchenko–Pastur 随机基线
  → 尖峰协方差和低秩信号分离
  → Jacobian、Hessian 与经验谱估计
  → 初始化、梯度传播与训练动态诊断
```

用于建立噪声基线并分析权重、激活和梯度。谱中的离群值不自动代表语义或能力；解释必须结合任务、干预和对照实验。

### E. 信息几何与策略优化

```text
熵、交叉熵与 KL 散度
  → score 与 Fisher 信息
  → KL 的局部二阶展开
  → 自然梯度、阻尼与结构化近似
  → 镜像下降、信赖域与 TRPO
  → PPO、KL penalty 与 LLM 策略更新
```

用于描述参数变化如何引起输出分布变化。PPO 不是精确自然梯度，参数的 Euclidean 距离也不等于模型行为距离；这些差异会在推导中明确保留。

### F. 最优传输与多模态对齐

```text
概率测度、耦合与运输成本
  → Wasserstein 距离与 Kantorovich 对偶
  → 熵正则与 Sinkhorn 迭代
  → unbalanced OT 与 Gromov–Wasserstein
  → CCA、Procrustes、主角度与共享子空间
  → token–patch、集合与分布级多模态匹配
```

普通 row-softmax attention 不等于最优传输；只有成本、边际和约束都有明确含义时，OT 才是合适的模型。

### G. 几何学习与结构化多模态

```text
欧氏与仿射几何
  → 拓扑、流形、切空间与曲率
  → 谱图理论与图神经网络
  → 群表示与等变映射
  → 三维视觉、点云、分子与机器人状态
  → 几何视觉—语言—行动模型
```

这一方向主要服务三维视觉、机器人、分子和具身多模态，不会把适用于 $SE(3)$ 的理论泛化成所有图文模型的共同解释。

完整内容依赖、优先级和研究边界见 [`roadmap.qmd`](roadmap.qmd)。

## 写作准则

- 符号首次出现时声明类型、范围和含义；
- 矩阵与张量运算同时给出整体形式、分量形式和 shape check；
- 关键主题尽可能包含定义、推导、直觉、数值例子与代码核对；
- 明确区分定义、定理、证明、经验现象、工程惯例和研究假设；
- 高级理论必须写出适用条件、反例或失效边界。

## 当前进度

| 状态 | 内容 |
|---|---|
| 已整理 | 数学语言导论；数、集合与区间；函数与映射；命题、量词与证明；微积分核心定义；概率统计核心定义；标量与向量 |
| 整理中 | 矩阵、线性映射与矩阵乘法 |
| 之后补充 | 向量空间、内积、投影、SVD；系统性的微积分、概率论与统计推断 |
| 路线草案 | 核心路线与 A–G 七条高级路线 |

项目仍处在基础章节建设期。路线图表示内容边界与依赖，不表示所有章节已经完成。

## 阅读与构建

- **在线阅读**：<https://devillove084.github.io/algebra-geometry-intelligence/>

- **本地预览**：`make serve`
- **构建网站**：`make build`
- **构建 PDF**：`make pdf`
- **重新生成线性代数插图**：`make figures`
- **清理产物**：`make clean`
- **首次安装**：`bash scripts/setup.sh`
- **仅补装中文翻译**：`make translations`

环境、Pages 发布和脚本说明见 [`DEVELOPMENT.md`](DEVELOPMENT.md)。
