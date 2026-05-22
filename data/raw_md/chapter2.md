# 第二章 神经网络基础

## 章节概述

本章学习搭建神经网络所需的基本知识，包括从机器学习到神经网络的演进、神经网络训练方法（正向传播与反向传播）、网络设计原则、拓扑结构、激活函数、损失函数、过拟合与正则化、交叉验证等内容，为深入理解深度学习打下基础。

---

## 2.1 从机器学习到神经网络

### 2.1.1 人工智能、机器学习、神经网络与深度学习的关系

四者的包含关系为：**人工智能 $\supset$ 机器学习 $\supset$ 神经网络 $\supset$ 深度学习**。

### 2.1.2 机器学习的定义

- **Mitchell定义**：机器学习是对能通过经验自动改进的计算机算法的研究。
- **Alpaydin定义**：机器学习是用数据或以往的经验，以此提升计算机程序的能力。
- **周志华定义**：机器学习是研究如何通过计算的手段、利用经验来改善系统自身性能的一门学科。

典型的机器学习过程：训练数据 $\to$ 机器学习方法 $\to$ 模型函数 $\to$ 新数据 $\to$ 预测值。

### 2.1.3 符号说明

| 符号 | 含义 |
|------|------|
| $x$ | 输入数据 |
| $y$ | 真实值（实际值） |
| $\hat{y}$ | 计算值（模型输出值） |
| $H(x)$ | 模型函数 |
| $G(x)$ | 激活函数 |
| $L(x)$ | 损失函数 |
| 标量 | 斜体小写字母 $a, b, c$ |
| 向量 | 黑斜体小写字母 $\mathbf{a}, \mathbf{b}, \mathbf{c}$ |
| 矩阵 | 黑斜体大写字母 $\mathbf{A}, \mathbf{B}, \mathbf{C}$ |

---

## 2.2 线性回归模型

### 2.2.1 单变量线性回归（一元回归模型）

线性回归可以找到点集背后的规律：一个点集可以用一条直线来拟合，拟合出的直线参数特征即为点集背后的规律。

**单变量线性模型**：

$$H_w(x) = w_0 + w \cdot x$$

其中 $x$ 为特征（Feature），$H(x)$ 为假设函数（Hypothesis）。

**问题引入**：假设房屋销售中心有房屋面积和售价的数据，用 $x_1$ 表示房屋面积，$y$ 表示售价（万元）。设计回归程序预测新房屋面积的售价。

### 2.2.2 多变量线性回归模型

当影响目标值的特征有多个时（如房屋面积、楼层、朝向等），使用多变量线性模型：

**2个特征的多变量线性模型**：

$$H_w(x) = w_0 + w_1 x_1 + w_2 x_2$$

**n个特征的多变量线性模型**：

$$H_w(x) = \sum_{i=0}^{n} w_i x_i = \hat{\mathbf{w}}^T \mathbf{x}$$

其中 $\hat{\mathbf{w}} = [w_0; w_1; \ldots; w_n]$，$\mathbf{x} = [x_0; x_1; \ldots; x_n]$，$x_0 = 1$。

### 2.2.3 损失函数与参数优化

模型预测值 $\hat{y}$ 与真实值 $y$ 之间存在误差：

$$\varepsilon = y - \hat{y} = y - \hat{\mathbf{w}}^T \mathbf{x}$$

其中 $\varepsilon$ 满足高斯分布 $\mathcal{N}(0, \sigma^2)$：

$$p(\varepsilon) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{\varepsilon^2}{2\sigma^2}\right)$$

由此推导：

$$p(y \mid \mathbf{x}; \hat{\mathbf{w}}) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(y - \hat{\mathbf{w}}^T \mathbf{x})^2}{2\sigma^2}\right)$$

**损失函数（均方误差）**，通过最大似然函数得到：

$$L(\hat{\mathbf{w}}) = \frac{1}{2} \sum_{j=1}^{m} \left(H_w(\mathbf{x}^{(j)}) - y^{(j)}\right)^2 = \frac{1}{2} \sum_{j=1}^{m} \left(\hat{\mathbf{w}}^T \mathbf{x}^{(j)} - y^{(j)}\right)^2$$

**目标**：求出参数 $\hat{\mathbf{w}}$，使得损失函数 $L(\hat{\mathbf{w}})$ 取值最小。

**梯度下降法**：
- 初始给定参数 $\hat{\mathbf{w}}$（如零向量或随机向量）
- 沿着梯度下降方向迭代更新：

$$\hat{\mathbf{w}} \leftarrow \hat{\mathbf{w}} - \alpha \cdot \frac{\partial L(\hat{\mathbf{w}})}{\partial \hat{\mathbf{w}}}$$

- $\alpha$ 称为**学习率（步长）**
- 迭代至找到使 $L(\hat{\mathbf{w}})$ 最小的 $\hat{\mathbf{w}}$ 值停止

---

## 2.3 感知机模型

### 2.3.1 生物神经元与人工神经元

- **生物神经元**：有多个树突（接收信息）、一条轴突（传递信息），轴突末梢与其它神经元树突连接处称为"突触"。
- **人工神经元**：包含输入、输出与计算功能的模型。输入类比树突，输出类比轴突，计算类比细胞体。
- 关系类比：生物神经元 : 人工神经元 = 老鼠 : 米老鼠

### 2.3.2 感知机模型定义

感知机模型 $H(\mathbf{x}) = \text{sign}(\mathbf{w}^T \mathbf{x} + b)$ 对应一个超平面 $\mathbf{w}^T \mathbf{x} + b = 0$，目标是将线性可分的数据集正确分为两类。

其中符号函数为：

$$\text{sign}(x) = \begin{cases} +1, & x \geq 0 \\ -1, & x < 0 \end{cases}$$

### 2.3.3 感知机损失函数

对于训练数据集 $D = \{(\mathbf{x}_1, y_1), (\mathbf{x}_2, y_2), \ldots, (\mathbf{x}_m, y_m)\}$，其中 $\mathbf{x}_j \in \mathbb{R}^n$，$y_j \in \{+1, -1\}$。

样本点 $\mathbf{x}_j$ 到超平面 $S$ 的距离公式：

$$d = \frac{1}{\|\mathbf{w}\|} \left|\mathbf{w}^T \mathbf{x}_j + b\right|$$

数据集中误分类点满足条件：$-y_j(\mathbf{w}^T \mathbf{x}_j + b) > 0$。

去掉绝对值符号后：$d = -\frac{1}{\|\mathbf{w}\|} y_j (\mathbf{w}^T \mathbf{x}_j + b)$

所有误分类点到超平面 $S$ 的总距离为：

$$d = -\frac{1}{\|\mathbf{w}\|} \sum_{\mathbf{x}_j \in M} y_j (\mathbf{w}^T \mathbf{x}_j + b)$$

感知机损失函数（去掉 $\frac{1}{\|\mathbf{w}\|}$）：

$$L(\mathbf{w}, b) = -\sum_{\mathbf{x}_j \in M} y_j (\mathbf{w}^T \mathbf{x}_j + b)$$

其中 $M$ 为误分类点集合。

### 2.3.4 随机梯度下降优化

使用随机梯度下降法最小化损失函数。梯度：

$$\nabla_{\mathbf{w}} L(\mathbf{w}, b) = -\sum_{\mathbf{x}_j \in M} y_j \mathbf{x}_j$$

$$\nabla_{b} L(\mathbf{w}, b) = -\sum_{\mathbf{x}_j \in M} y_j$$

更新规则（随机选取误分类点 $(\mathbf{x}_j, y_j)$）：
- 更新权重：$\mathbf{w} \leftarrow \mathbf{w} + \alpha y_j \mathbf{x}_j$
- 更新偏置：$b \leftarrow b + \alpha y_j$
- 迭代至 $L(\mathbf{w}, b) \to 0$

---

## 2.4 多层感知机（两层神经网络）

### 2.4.1 网络结构

多层感知机是全连接的两层神经网络模型：
- 输入层 $\to$ 隐层 $\to$ 输出层
- 隐层计算：$\mathbf{h} = G(\mathbf{W}^{(1)T} \mathbf{x} + \mathbf{b}^{(1)})$
- 输出层计算：$\hat{\mathbf{y}} = G(\mathbf{W}^{(2)T} \mathbf{h} + \mathbf{b}^{(2)})$
- 除输出层外，每层都有一个偏置单元 $\mathbf{b}$，与后一层所有节点相连
- $\mathbf{W}$ 为权重，$\mathbf{b}$ 为偏置，$(\mathbf{W}, \mathbf{b})$ 合称为神经网络的参数

### 2.4.2 参数数量计算

对于一个 3输入 $\to$ 3隐层 $\to$ 2输出 的网络：
- 第一层参数数量：$3 \times 3 + 3 = 12$ 个
- 第二层参数数量：$3 \times 2 + 2 = 8$ 个
- 总参数数量：$20$ 个

### 2.4.3 浅层神经网络特点

- 优点：需要数据量小、训练快
- 局限：对复杂函数的表示能力有限，针对复杂分类问题泛化能力受制约
- Kurt Hornik 证明：理论上两层神经网络足以拟合任意函数

---

## 2.5 深度神经网络（深度学习）

### 2.5.1 深度学习的发展

- 2006年，Hinton在Science发表论文（Reducing the dimensionality of data with neural networks），提出"深度学习"概念
- Hinton、LeCun、Bengio 被称为深度学习三位开创者

### 2.5.2 深度学习成功的三要素（ABC）

- **A - Algorithm（算法）**：学习算法 $\to$ BP算法 $\to$ Pre-training、Dropout等方法不断涌现
- **B - Big Data（大数据）**：数据量从 $10 \to 10k \to 100M$ 不断增大
- **C - Computing（计算力）**：晶体管 $\to$ CPU $\to$ 集群/GPU $\to$ 智能处理器，计算能力不断提升

### 2.5.3 深度网络的层次特征提取

随着层数增加，每一层对前一层的抽象表示更深入：
- 第一隐层：提取"边缘"特征
- 第二隐层：提取"形状"特征
- 第三隐层：提取"图案"特征
- 第四隐层：提取"目标"特征

推导公式：

$$\mathbf{h}^{(1)} = G(\mathbf{W}_1^T \mathbf{x} + \mathbf{b}^{(1)})$$

$$\mathbf{h}^{(2)} = G(\mathbf{W}_2^T \mathbf{h}^{(1)} + \mathbf{b}^{(2)})$$

$$\hat{\mathbf{y}} = G(\mathbf{W}_3^T \mathbf{h}^{(2)} + \mathbf{b}^{(3)})$$

通过抽取更抽象的特征来获得更好的区分与分类能力。从单层到多层，配合激活函数的调整，神经网络拟合非线性分界的能力不断增强。

---

## 2.6 神经网络训练

### 2.6.1 正向传播（Forward Propagation）

正向传播是根据输入，经过权重和激活函数逐层计算，将输入特征从低级特征逐步提取为抽象特征，直到得到最终输出的过程。

**具体计算示例（使用Sigmoid激活函数）**：

输入到隐层计算：

$$\mathbf{v} = \mathbf{W}^{(1)T} \mathbf{x} + \mathbf{b}^{(1)}$$

$$\mathbf{h} = \frac{1}{1 + e^{-\mathbf{v}}} = \sigma(\mathbf{v})$$

隐层到输出层计算：

$$\mathbf{z} = \mathbf{W}^{(2)T} \mathbf{h} + \mathbf{b}^{(2)}$$

$$\hat{\mathbf{y}} = \frac{1}{1 + e^{-\mathbf{z}}} = \sigma(\mathbf{z})$$

### 2.6.2 反向传播（Back Propagation）

反向传播是根据输出结果和期望值计算损失函数，通过链式求导法则，从网络后端逐步修改权重，使输出与期望值差距最小化的过程。

**损失函数计算**（以均方误差为例）：

$$L(\mathbf{W}) = L_1 + L_2 = \frac{1}{2}(y_1 - \hat{y}_1)^2 + \frac{1}{2}(y_2 - \hat{y}_2)^2$$

**链式求导法则**（以隐层到输出层权重更新，以参数 $\omega = w_{2,1}^{(2)}$ 为例）：

$$\frac{\partial L(\mathbf{W})}{\partial \omega} = \frac{\partial L(\mathbf{W})}{\partial \hat{y}_1} \cdot \frac{\partial \hat{y}_1}{\partial z_1} \cdot \frac{\partial z_1}{\partial \omega}$$

其中：
- $\frac{\partial L(\mathbf{W})}{\partial \hat{y}_1} = -(y_1 - \hat{y}_1)$ （损失函数对输出的导数）
- $\frac{\partial \hat{y}_1}{\partial z_1} = \hat{y}_1 (1 - \hat{y}_1)$ （Sigmoid函数的导数：$\sigma'(z) = \sigma(z)(1 - \sigma(z))$）
- $\frac{\partial z_1}{\partial \omega} = h_2$ （因为 $z_1 = w_{1,1}^{(2)} h_1 + \omega \cdot h_2 + w_{3,1}^{(2)} h_3 + b_1^{(2)}$）

合并得：

$$\frac{\partial L(\mathbf{W})}{\partial \omega} = -(y_1 - \hat{y}_1) \cdot \hat{y}_1(1 - \hat{y}_1) \cdot h_2$$

**权重更新公式**：

$$\omega_{\text{new}} = \omega_{\text{old}} - \alpha \cdot \frac{\partial L(\mathbf{W})}{\partial \omega}$$

### 2.6.3 训练流程总结

正向传播 $\to$ 计算损失 $\to$ 反向传播更新权重 $\to$ 重复迭代，不断缩小计算值与真实值之间的误差。

当第一次反向传播完成后，网络模型参数得到更新，网络进行下一轮正向传播，如此反复迭代训练。

---

## 2.7 神经网络设计原则

### 2.7.1 网络拓扑结构设计

- **输入层**：神经元个数 = 特征维度
- **输出层**：神经元个数 = 分类类别数
- **隐层设计原则**：
  - 隐层节点太少：网络获取信息能力差，无法反映数据规律
  - 隐层节点太多：拟合能力过强，可能拟合噪声，导致泛化能力变差

神经网络结构一般为：输入层 $\times$ 隐层 $\times$ 输出层。给定训练样本后，输入和输出层节点数便已确定。

### 2.7.2 激活函数需具备的性质

- **可微性**：使用梯度优化方法时必须可微
- **输出值范围**：有限输出使梯度优化更稳定；无限输出使训练更高效但需要更小的学习率
- **非线性**：给神经元引入非线性因素，使神经网络可以逼近任意非线性函数

---

## 2.8 激活函数

### 2.8.1 Sigmoid 函数

**公式**：

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

**导数**：$\sigma'(x) = \sigma(x)(1 - \sigma(x))$

**特点**：
- 输出范围 $(0, 1)$，将连续实值映射到0和1之间
- 优点：最常见的非线性激活函数
- 缺点：
  - **非0均值输出**，导致梯度始终为正
  - 计算机进行指数运算速度慢
  - 存在**饱和性问题**及**梯度消失**现象（输入值很大或很小时，梯度接近0）

### 2.8.2 Tanh 函数

**公式**：

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = \frac{\sinh(x)}{\cosh(x)}$$

亦可表示为：$\tanh(x) = 2 \cdot \sigma(2x) - 1$

**特点**：
- 输出范围 $(-1, 1)$，**0均值输出**，解决了Sigmoid非0均值的问题
- 仍然存在梯度消失问题（输入很大或很小时，输出平滑，梯度很小）

### 2.8.3 ReLU 函数

**公式**：

$$f(x) = \max(0, x)$$

**特点**：
- $x > 0$ 时梯度不衰减（恒为1），有效**缓解梯度消失**问题
- 输出范围无限：$[0, +\infty)$
- 计算简单快速
- 缺点：**ReLU死亡问题**——学习率过大时参数可能变为负数，导致神经元完全不被激活（输入负数时ReLU输出为0，梯度为0）

### 2.8.4 Leaky ReLU / PReLU 函数

**公式**：

$$f(x) = \max(\alpha x, x), \quad \alpha \in (0, 1)$$

- 在负数区域有很小斜率，可以避免ReLU死亡问题
- **PReLU**中 $\alpha$ 为可调参数，每个通道有一个 $\alpha$，通过反向传播训练得到

### 2.8.5 ELU 函数 (Exponential Linear Unit)

**公式**：

$$f(x) = \begin{cases} x, & x > 0 \\ \alpha(e^x - 1), & x \leq 0 \end{cases}$$

**特点**：
- $\alpha$ 是可调参数，控制ELU在负值区间的饱和位置
- 输出均值接近零，**收敛速度更快**
- 右侧线性部分缓解梯度消失
- 左侧软饱和使ELU对输入变化或噪声更**鲁棒**，避免神经元死亡

---

## 2.9 损失函数

### 2.9.1 损失函数概述

损失函数 $L = f(\hat{y}, y)$，其中 $\hat{y} = H_w(x)$ 是模型预测值，是神经网络模型参数 $\mathbf{W}$ 的函数。从 $\mathbf{w}$ 角度看，损失函数可记为 $L(\mathbf{w}) = f(H_w(x), y)$。

损失函数可评价网络模型好坏，损失越小说明模型越符合训练样本。

### 2.9.2 均方差损失函数（MSE）

**公式**：

$$L = \frac{1}{2}(y - \hat{y})^2$$

以一个神经元的均方差损失函数为例（使用Sigmoid激活函数，$\hat{y} = \sigma(z)$，$z = wx + b$）：

梯度计算：

$$\frac{\partial L}{\partial w} = (y - \hat{y}) \cdot \sigma'(z) \cdot x$$

$$\frac{\partial L}{\partial b} = (y - \hat{y}) \cdot \sigma'(z)$$

**问题**：梯度中含有 $\sigma'(z)$，当神经元输出接近1时，$\sigma'(z)$ 趋于0，出现**梯度消失**，导致参数更新缓慢，学习效率下降。

### 2.9.3 交叉熵损失函数

**为什么需要交叉熵**：交叉熵损失 + Sigmoid激活函数可以解决输出层神经元学习率缓慢的问题。

**二分类交叉熵**：

$$L = -\frac{1}{m} \sum_{\mathbf{x} \in D} \left[y \ln \hat{y} + (1 - y) \ln(1 - \hat{y})\right]$$

**多分类交叉熵**：

$$L = -\frac{1}{m} \sum_{\mathbf{x} \in D} \sum_{i} y_i \ln \hat{y}_i$$

其中 $m$ 为训练样本总数，$i$ 为分类类别。

**为什么交叉熵能解决梯度消失问题**：

使用Sigmoid激活函数时，$\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}} = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$，交叉熵损失函数的梯度为：

$$\frac{\partial L}{\partial \mathbf{w}} = -\frac{1}{m} \sum_{\mathbf{x} \in D} \left[\frac{y}{\sigma(z)} - \frac{1-y}{1-\sigma(z)}\right] \cdot \sigma'(z) \cdot \mathbf{x}$$

化简后：

$$\frac{\partial L}{\partial \mathbf{w}} = \frac{1}{m} \sum_{\mathbf{x} \in D} (\sigma(z) - y) \cdot \mathbf{x}$$

$$\frac{\partial L}{\partial b} = \frac{1}{m} \sum_{\mathbf{x} \in D} (\sigma(z) - y)$$

Sigmoid的导数 $\sigma'(z)$ 被约掉，最后一层的梯度中不再含有 $\sigma'(z)$，有效避免了梯度消失。

### 2.9.4 损失函数的特性

- 同一算法的损失函数不唯一
- 损失函数是参数 $(\mathbf{w}, b)$ 的函数
- 损失函数可以评价网络模型的好坏，损失越小说明模型和参数越符合训练样本 $(x, y)$
- 损失函数是一个标量
- 选择损失函数时，需挑选对参数 $(\mathbf{w}, b)$ 可微的函数（全微分存在，偏导数一定存在）
- 损失函数又称为代价函数、目标函数

---

## 2.10 过拟合与正则化

### 2.10.1 欠拟合与过拟合

| 现象 | 特征 |
|------|------|
| **欠拟合（Underfitting）** | 训练考虑的维度太少，拟合函数无法满足训练集，误差较大 |
| **合适拟合（Good Fit）** | 模型适度拟合训练数据，泛化能力良好 |
| **过拟合（Overfitting）** | 训练考虑的维度太多，完美拟合训练数据但泛化能力差，对新数据预测能力不足 |

过拟合具体表现：训练集上误差很低，但验证集上误差很大。

### 2.10.2 正则化的基本思路

例如，拟合函数从 $w_0 + w_1 x + w_2 x^2$ 变为 $w_0 + w_1 x + w_2 x^2 + w_3 x^3 + w_4 x^4$，更复杂但可能过拟合。

解决方案：在损失函数中增加惩罚项，使 $w_3$、$w_4$ 足够小趋近于0：

$$\min_{\mathbf{w}} \frac{1}{2m} \sum_{i=1}^{m} \|\mathbf{y}_i - \hat{\mathbf{y}}_i\|^2 + C_1 \cdot w_3^2 + C_2 \cdot w_4^2$$

要使目标函数最小，则应有 $w_3 \approx 0$、$w_4 \approx 0$。

### 2.10.3 正则化的一般形式

正则化在损失函数中增加惩罚项 $\Omega(\mathbf{w})$：

$$\tilde{L}(\mathbf{w}; \mathbf{X}, \mathbf{y}) = L(\mathbf{w}; \mathbf{X}, \mathbf{y}) + \theta \cdot \Omega(\mathbf{w})$$

其中 $\theta$ 为正则化参数。神经网络中的参数包括权重 $\mathbf{w}$ 和偏置 $\mathbf{b}$，**正则化过程仅对权重 $\mathbf{w}$ 进行惩罚**。

### 2.10.4 L2 正则化（Ridge回归/权重衰减）

**惩罚项**：

$$\Omega(\mathbf{w}) = \frac{1}{2} \|\mathbf{w}\|_2^2$$

**目标函数**：

$$\tilde{L}(\mathbf{w}; \mathbf{X}, \mathbf{y}) = L(\mathbf{w}; \mathbf{X}, \mathbf{y}) + \frac{\theta}{2} \|\mathbf{w}\|^2$$

**梯度**：

$$\nabla_{\mathbf{w}} \tilde{L}(\mathbf{w}; \mathbf{X}, \mathbf{y}) = \nabla_{\mathbf{w}} L(\mathbf{w}; \mathbf{X}, \mathbf{y}) + \theta \mathbf{w}$$

**单步梯度更新**：

$$\mathbf{w} \leftarrow \mathbf{w} - \eta(\nabla_{\mathbf{w}} L(\mathbf{w}; \mathbf{X}, \mathbf{y}) + \theta \mathbf{w})$$

$$\mathbf{w} \leftarrow (1 - \eta\theta)\mathbf{w} - \eta \nabla_{\mathbf{w}} L(\mathbf{w}; \mathbf{X}, \mathbf{y})$$

通过L2正则化，$\mathbf{w}$ 权重值变小，网络复杂度降低，对数据拟合更好。L2正则化通过使权重衰减，减小模型复杂度来避免过拟合。

### 2.10.5 L1 正则化（Lasso回归）

**惩罚项**：

$$\Omega(\mathbf{w}) = \|\mathbf{w}\|_1 = \sum_{i} |w_i|$$

**目标函数**：

$$\tilde{L}(\mathbf{w}; \mathbf{X}, \mathbf{y}) = L(\mathbf{w}; \mathbf{X}, \mathbf{y}) + \theta \|\mathbf{w}\|_1$$

**梯度**：

$$\nabla_{\mathbf{w}} \tilde{L}(\mathbf{w}; \mathbf{X}, \mathbf{y}) = \nabla_{\mathbf{w}} L(\mathbf{w}; \mathbf{X}, \mathbf{y}) + \theta \cdot \text{sign}(\mathbf{w})$$

L1正则化通过加入符号函数 $\text{sign}(\mathbf{w})$，使得：
- 当 $w_i$ 为正时，更新后的 $w_i$ 变小
- 当 $w_i$ 为负时，更新后的 $w_i$ 变大
- 最终效果是让 $w_i$ 接近0，减小网络复杂度，防止过拟合

### 2.10.6 其他正则化方法

- **Dropout正则化**：训练时随机"删除"部分隐层单元（乘以0），阻止特征共同适应。输入单元采样概率约0.8，隐藏单元采样概率约0.5。
- **Bagging集成方法**：训练不同模型共同决策测试样例的输出。从原始数据集重复采样获取采样数据集（大小与原始数据集一致），分别训练多个模型，取平均输出。模型平均是减小泛化误差的可靠方法。
- **提前终止（Early Stopping）**：训练大网络时，训练误差降低但验证集误差会再次上升，在验证误差最低时返回参数设置，获得验证集误差更低的模型。
- **数据集增强（Data Augmentation）**：使用更多数据训练，对原数据集变换形成新数据添加到训练数据中。
- **多任务学习（Multi-Task Learning）**：通过合并多个任务的样例来减少神经网络的泛化误差。
- **参数共享（Parameter Sharing）**：强迫两个模型（监督和无监督模式）共享唯一的一组参数。
- **稀疏表示（Sparse Representation）**：惩罚神经网络中的激活单元，稀疏化激活单元。

---

## 2.11 交叉验证

### 2.11.1 简单划分法

将数据集分为训练集和测试集两部分。
- 缺点：结果依赖划分方式，不同划分下的MSE变动较大；只有部分数据参与训练。

### 2.11.2 留一交叉验证（Leave-One-Out Cross Validation）

- 每次取出一个数据作为测试集，其余 $n-1$ 个数据作为训练集
- 训练出 $n$ 个模型，得到 $n$ 个MSE
- 将这 $n$ 个MSE取平均得到最终test MSE
- 缺点：计算量过大，耗费时间长

### 2.11.3 K折交叉验证（K-Fold Cross Validation）

- 将数据集 $S$（含 $n$ 个数据）分成 $K$ 份
- 不重复地每次取其中1份做测试集，其余 $K-1$ 份做训练集
- 计算该模型在测试集上的 $MSE_i$
- 最终将 $K$ 次的 $MSE_i$ 取平均得到最终MSE：

$$\text{MSE} = \frac{1}{K} \sum_{i=1}^{K} MSE_i$$

- 特点：留一交叉验证是K折交叉验证在 $K = n$ 时的特例
- 优点：所有样本都作为了训练集和测试集，每个样本都被验证一次；相比留一法，计算成本降低，耗时减少

---

## 2.12 本章小结

本章核心知识点：
- **从机器学习到神经网络**：线性回归、多变量线性回归、感知机、神经元、激活函数、偏置等概念
- **正向传播与反向传播**：理解数据在神经网络中的正反向传播过程
- **激活函数**：了解Sigmoid、Tanh、ReLU、Leaky ReLU、ELU等常用激活函数的优缺点
- **损失函数**：了解MSE和交叉熵损失函数的作用、种类及应用场景
- **过拟合与正则化**：理解过拟合原因，掌握L1、L2、Dropout等正则化方法
- **交叉验证**：掌握K折交叉验证等验证方法的原理
