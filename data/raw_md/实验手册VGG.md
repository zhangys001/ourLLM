# 第3章 深度学习应用实验

## 3.1 基于 VGG19 实现图像分类

### 3.1.1 实验目的

掌握卷积神经网络的设计原理，掌握卷积神经网络的使用方法，能够使用 Python 语言实现 VGG19[3] 网络模型对给定的输入图像进行分类。具体包括：

1. 加深对深度卷积神经网络中卷积层、最大池化层等基本单元的理解。
2. 利用 Python 语言实现 VGG19 的前向传播计算，加深对 VGG19 网络结构的理解，为后续风格迁移中使用 VGG19 网络进行特征提取奠定基础。
3. 在第2.1节实验的基础上将三层神经网络扩展为 VGG19 网络，加深对神经网络工程实现中基本模块演变的理解，为后续实现更复杂的综合实验（如风格迁移）奠定基础。

**实验工作量**：约30行代码，约需3个小时。

### 3.1.2 背景介绍

#### 3.1.2.1 卷积神经网络中的基本单元

常见的卷积神经网络结构如图3.1所示。卷积层后面会使用 ReLU 等激活函数，几个卷积层后通常会使用一个最大池化层（也有使用平均池化的）。卷积和池化组合出现若干次之后，提取出来的卷积特征会经过若干个全连接层映射到若干个输出特征上；最后再经过一个全连接层或 Softmax 层来决定最终的输出。在第2.1节实验中，已经介绍了全连接层、ReLU激活函数、Softmax 层，本节介绍本实验中新增的基本单元：卷积层和最大池化层。更多关于卷积层和最大池化层的介绍详见《智能计算系统》教材第3.1节。

> **图3.1 常见卷积神经网络结构**（示意图：输入 → 卷积 → 激活 → 池化 → ... → 全连接 → 激活 → 全连接或Softmax → 输出）

##### 1) 卷积层

与全连接层类似，卷积层中的参数包括权重（即卷积核）和偏置。VGG19 中使用的都是多输入输出特征图的卷积运算。假设输入特征图 $X$ 的维度为 $N \times C_{in} \times H_{in} \times W_{in}$，其中 $N$ 是输入的样本个数（在本实验中 $N = 1$），$C_{in}$ 是输入的通道数，$H_{in}$ 和 $W_{in}$ 是输入特征图的高和宽。卷积核张量 $W$ 用四维矩阵表示，维度为 $C_{in} \times K \times K \times C_{out}$，其中 $K \times K$ 为卷积核的高度 $\times$ 宽度（也称为卷积窗口大小），$C_{out}$ 为输出特征图的通道数。卷积层的偏置用一维向量 $b$ 表示，维度为 $C_{out}$。同时定义输入特征图的边界扩充大小 $p$、卷积步长 $s$。输出特征图 $Y$ 由输入 $X$ 在滑动窗口内的数据与卷积核 $W$ 内积并加偏置 $b$ 计算得到，$Y$ 的维度为 $N \times C_{out} \times H_{out} \times W_{out}$，其中 $H_{out}$ 和 $W_{out}$ 是输出特征图的高和宽。

**前向传播计算时**，为了保证卷积之后的有效输出尺寸与输入尺寸一致，首先对卷积层的输入 $X$ 做边界扩充（padding），即在输入特征图的上下以及左右边界分别增加 $p$ 行以及 $p$ 列的 0。维度为 $N \times C_{in} \times H_{in} \times W_{in}$ 的输入特征图，经过大小为 $p$ 的边界扩充，得到扩充后的特征图 $X_{pad}$：

$$
X_{pad}(n, C_{in}, h, w) =
\begin{cases}
X(n, C_{in}, h - p, w - p) & \text{如果 } (p \leq h < p + H_{in}) \text{ 且 } (p \leq w < p + W_{in}) \\
0 & \text{其他}
\end{cases}
\tag{3.1}
$$

其中 $n \in [0, N)$，$C_{in} \in [0, C_{in})$，$h \in [0, H_{in})$，$w \in [0, W_{in})$ 分别表示输入特征图的样本号、通道号、行号、列号，均为整数。$X_{pad}$ 的维度为 $N \times C_{in} \times H_{pad} \times W_{pad}$，其中高度 $H_{pad}$ 和宽度 $W_{pad}$ 分别为：

$$
H_{pad} = H_{in} + 2p, \quad W_{pad} = W_{in} + 2p \tag{3.2}
$$

然后，在边界扩充后的特征图上滑动卷积窗口，依次计算窗口内的特征图数据与卷积核的矩阵内积再加上偏置得到输出特征图 $Y$：

$$
Y(n, C_{out}, h, w) = \sum_{C_{in}, k_h, k_w} W(C_{in}, k_h, k_w, C_{out}) \cdot X_{pad}(n, C_{in}, hs + k_h, ws + k_w) + b(C_{out}) \tag{3.3}
$$

其中 $n \in [0, N)$，$C_{out} \in [0, C_{out})$，$h \in [0, H_{out})$，$w \in [0, W_{out})$ 分别表示输出特征图的样本号、通道号、行号、列号；$k_h \in [0, K)$，$k_w \in [0, K)$ 表示卷积核的行号和列号；$C_{in} \in [0, C_{in})$ 表示输入特征图的通道号，这些符号的值均为整数。输出特征图 $Y$ 的高度和宽度分别是：

$$
H_{out} = \frac{H_{pad} - K}{s} + 1 = \frac{H_{in} + 2p - K}{s} + 1, \quad W_{out} = \frac{W_{pad} - K}{s} + 1 = \frac{W_{in} + 2p - K}{s} + 1 \tag{3.4}
$$

**反向传播计算时**，假设损失函数为 $L$，损失函数对本层输出的偏导为 $\nabla_Y L$，其维度与卷积层的输出特征图相同，均为 $N \times C_{out} \times H_{out} \times W_{out}$。根据链式法则，可以计算权重和偏置的梯度 $\nabla_W L$、$\nabla_b L$ 以及损失函数对边界扩充后的输入特征图的偏导 $\nabla_{X_{pad}} L$，计算公式为：

$$
\begin{aligned}
\nabla_{W(C_{in}, k_h, k_w, C_{out})} L &= \sum_{n, h, w} \nabla_{Y(n, C_{out}, h, w)} L \cdot X_{pad}(n, C_{in}, hs + k_h, ws + k_w) \\
\nabla_{b(C_{out})} L &= \sum_{n, h, w} \nabla_{Y(n, C_{out}, h, w)} L \\
\nabla_{X_{pad}(n, C_{in}, h', w')} L &= \sum_{C_{out}} \sum_{k_h=0}^{K-1} \sum_{k_w=0}^{K-1} \nabla_{Y(n, C_{out}, \lfloor h'/s \rfloor - k_h, \lfloor w'/s \rfloor - k_w)} L \cdot W(C_{in}, k_h, k_w, C_{out})
\end{aligned}
\tag{3.5}
$$

其中 $n \in [0, N)$，$C_{in} \in [0, C_{in})$，$C_{out} \in [0, C_{out})$，$h \in [0, H_{out})$，$w \in [0, W_{out})$，$k_h \in [0, K)$，$k_w \in [0, K)$，$h' \in [0, H_{pad})$，$w' \in [0, W_{pad})$。

之后剪裁掉 $\nabla_{X_{pad}} L$ 中扩充的边界，得到本层的 $\nabla_X L$，计算公式为：

$$
\nabla_{X(n, C_{in}, h, w)} L = \nabla_{X_{pad}(n, C_{in}, h+p, w+p)} L \tag{3.6}
$$

其中 $n \in [0, N)$，$C_{in} \in [0, C_{in})$，$h \in [0, H_{in})$，$w \in [0, W_{in})$。

##### 2) 最大池化层

假设最大池化层的输入特征图 $X$ 的维度为 $N \times C \times H_{in} \times W_{in}$，其中 $N$ 是输入的样本个数（在本实验中 $N = 1$），$C$ 是输入的通道数，$H_{in}$ 和 $W_{in}$ 是输入特征图的高和宽。池化窗口的高和宽均为 $K$，池化步长为 $s$，输出特征图 $Y$ 的维度为 $N \times C \times H_{out} \times W_{out}$，其中 $H_{out}$ 和 $W_{out}$ 是输出特征图的高和宽。

**前向传播计算时**，输出特征图 $Y$ 中某一位置的值是输入特征图 $X$ 的对应池化窗口内的最大值，计算公式为：

$$
Y(n, c, h, w) = \max_{k_h, k_w} X(n, c, hs + k_h, ws + k_w) \tag{3.7}
$$

其中 $n \in [0, N)$，$c \in [0, C)$，$h \in [0, H_{out})$，$w \in [0, W_{out})$ 分别表示输出特征图的样本号、通道号、行号、列号，$k_h \in [0, K)$，$k_w \in [0, K)$ 表示池化窗口内的坐标位置，均为整数。

**反向传播的计算过程**可以根据前向传播公式(3.7)推导获得。给定损失函数对本层输出的偏导 $\nabla_Y L$，其维度与最大池化层的输出特征图相同，均为 $N \times C \times H_{out} \times W_{out}$。由于最大池化层在前向传播后仅保留池化窗口内的最大值，因此在反向传播时，仅将后一层损失中对应该池化窗口的值传递给池化窗口内最大值所在位置，其他位置值置为 0。在反向传播时需先计算最大值所在位置 $p$，计算公式为：

$$
p(n, c, h, w) = \arg\max_{k_h, k_w} X(n, c, hs + k_h, ws + k_w) \tag{3.8}
$$

其中 $\arg\max$ 代表取最大值所在位置的函数，返回最大值位于池化窗口中的坐标向量 $p(n, c, h, w) = [p(0), p(1)]$，其中 $p(0)$ 对应 $h$ 方向的坐标，$p(1)$ 对应 $w$ 方向的坐标。$n \in [0, N)$，$c \in [0, C)$，$h \in [0, H_{out})$，$w \in [0, W_{out})$，$k_h \in [0, K)$，$k_w \in [0, K)$ 分别为输入输出特征图和池化窗口上的位置坐标。

利用最大值所在位置 $[p(0), p(1)]$ 可得最大池化层的 $\nabla_X L$，计算公式为：

$$
\nabla_{X(n, c, hs + p(0), ws + p(1))} L = \nabla_{Y(n, c, h, w)} L \tag{3.9}
$$

#### 3.1.2.2 VGG19 网络的基本结构

VGG19[3] 是经典的深度卷积神经网络结构，包含 6 个阶段共 16 个卷积层和 3 个全连接层，如表3.1所示。前 2 个阶段各有 2 个卷积层，第 3-5 个阶段各有 4 个卷积层。每个卷积层均使用 $3 \times 3$ 大小的卷积核，边界扩充大小为 1，步长为 1，即保持输入输出特征图的高和宽不变。每个阶段的卷积层的通道数在不断变化。每个阶段的第一个卷积层的输入通道数为上一个卷积层的输出通道数（第一个阶段的输入通道数为原始图像通道数）。5 个阶段的卷积层输出通道数分别为 64, 128, 256, 512, 512。每个阶段除第一个卷积层外，其他卷积层均保持输入和输出通道数相同。每个卷积层后面都跟随有 ReLU 层作为激活函数。每个阶段最后都跟随有一个最大池化层，将特征图的高和宽缩小为原来的 1/2。3 个全连接层中前 2 个全连接层后面也跟随有 ReLU 层。值得注意的是，第五阶段输出的特征图会进行变形，将四维特征图变形为二维矩阵作为全连接层的输入。网络最后是 Softmax 层计算分类概率。VGG19 的超参数配置详见表3.1，注意表中省略了卷积层和全连接层后的 ReLU 层。更多关于 VGG19 网络基本结构的介绍详见《智能计算系统》课程教材第3.2.2节。

**表3.1 VGG19 网络的基本结构**

| 名字 | 类型 | 卷积核/池化核 | 步长 | 边界扩充 | 输入通道数 | 输出通道数 | 输出特征图的高和宽 |
|------|------|:---:|:---:|:---:|:---:|:---:|:---:|
| conv1_1 | 卷积层 | 3 | 1 | 1 | 3 | 64 | 224 |
| conv1_2 | 卷积层 | 3 | 1 | 1 | 64 | 64 | 224 |
| pool1 | 最大池化层 | 2 | 2 | — | 64 | 64 | 112 |
| conv2_1 | 卷积层 | 3 | 1 | 1 | 64 | 128 | 112 |
| conv2_2 | 卷积层 | 3 | 1 | 1 | 128 | 128 | 112 |
| pool2 | 最大池化层 | 2 | 2 | — | 128 | 128 | 56 |
| conv3_1 | 卷积层 | 3 | 1 | 1 | 128 | 256 | 56 |
| conv3_2 | 卷积层 | 3 | 1 | 1 | 256 | 256 | 56 |
| conv3_3 | 卷积层 | 3 | 1 | 1 | 256 | 256 | 56 |
| conv3_4 | 卷积层 | 3 | 1 | 1 | 256 | 256 | 56 |
| pool3 | 最大池化层 | 2 | 2 | — | 256 | 256 | 28 |
| conv4_1 | 卷积层 | 3 | 1 | 1 | 256 | 512 | 28 |
| conv4_2 | 卷积层 | 3 | 1 | 1 | 512 | 512 | 28 |
| conv4_3 | 卷积层 | 3 | 1 | 1 | 512 | 512 | 28 |
| conv4_4 | 卷积层 | 3 | 1 | 1 | 512 | 512 | 28 |
| pool4 | 最大池化层 | 2 | 2 | — | 512 | 512 | 14 |
| conv5_1 | 卷积层 | 3 | 1 | 1 | 512 | 512 | 14 |
| conv5_2 | 卷积层 | 3 | 1 | 1 | 512 | 512 | 14 |
| conv5_3 | 卷积层 | 3 | 1 | 1 | 512 | 512 | 14 |
| conv5_4 | 卷积层 | 3 | 1 | 1 | 512 | 512 | 14 |
| pool5 | 最大池化层 | 2 | 2 | — | 512 | 512 | 7 |
| flatten | Flatten层 | — | — | — | $512\times7\times7$ | 25088 | — |
| fc6 | 全连接层 | — | — | — | 25088 | 4096 | — |
| fc7 | 全连接层 | — | — | — | 4096 | 4096 | — |
| fc8 | 全连接层 | — | — | — | 4096 | 1000 | — |
| Softmax | 损失层 | — | — | — | 1000 | 1000 | — |

### 3.1.3 实验环境

**硬件环境**：CPU。

**软件环境**：Python 编译环境及相关的扩展库，包括 Python 3.6.12，Pillow 7.2.0，Scipy 1.2.0，NumPy 1.19.5（本实验不需使用 PyTorch 等深度学习框架）。

**数据集**：ImageNet[4] 图像数据集；该数据集包括约 128 万训练图像和 5 万张测试图像；共有 1000 个不同的类别。本实验使用官方基于 ImageNet 数据集训练好的模型参数。不需要使用 ImageNet 数据集进行 VGG19 模型的训练。

### 3.1.4 实验内容

本实验使用 VGG19 网络进行图像分类。首先建立 VGG19 的网络结构，然后利用 VGG19 的官方模型参数对给定图像进行分类。VGG19 网络的模型参数是在 ImageNet[4] 数据集上训练获得的，VGG19 网络的输出结果对应 ImageNet 数据集中的 1000 个类别概率。

在工程实现中，依然按照第2.1节实验的模块划分方法，每个模块的具体实现基于第2.1节实验进行改进。由于本实验只涉及 VGG19 网络的推断过程，因此本实验仅包括**数据加载模块**、**基本单元模块**、**网络结构模块**、**网络推断模块**，不包括网络训练模块。

### 3.1.5 实验步骤

#### 3.1.5.1 数据加载模块

数据加载模块实现数据读取和预处理，如代码示例3.1所示。本实验采用 ImageNet 图像数据集，该数据集以 .jpg 或 .png 压缩文件格式存放每张 RGB 图像，且不同图像的尺寸可能不同。为了统一神经网络输入的大小，读入图像数据后，需要依次做以下处理：

首先，将图像缩放到 $224 \times 224$ 大小，并存储在矩阵中。

其次对输入图像做标准化，将输入值范围从 [0, 255] 标准化为均值为 0 的区间，从而提高神经网络的训练速度和稳定性。具体做法是图像的每个像素值减去 ImageNet 数据集的像素均值，该图像均值在加载 VGG19 模型参数的同时读入。本实验中使用 VGG19 模型中自带的图像均值进行输入图像标准化，是为了确保与官方使用 VGG19 网络时的预处理方式保持一致。

最后，将标准化后的图像矩阵转换为神经网络输入的统一维度，即 $N \times C \times H \times W$，其中 $N$ 是输入的样本数（由于图像是逐张读入的，因此 $N = 1$），$C$ 是输入的通道数（本实验输入图像是 RGB 彩色图像，因此 $C = 3$），$H$ 和 $W$ 分别表示输入图像的高和宽（缩放后的图像的高和宽均为 224）。

**代码示例3.1 VGG19 的数据加载模块的实现**

```python
# file: vgg_cpu.py

def load_image(self, image_dir):
    print("Loading and preprocessing image from", image_dir)
    self.input_image = scipy.misc.imread(image_dir)
    self.input_image = scipy.misc.imresize(self.input_image, [224, 224, 3])
    self.input_image = np.array(self.input_image).astype(np.float32)
    self.input_image = self.image_mean - self.input_image
    self.input_image = np.reshape(self.input_image, [1] + list(self.input_image.shape))
    # input_dim = [N, Channel, height, width]
    self.input_image = np.transpose(self.input_image, [0, 3, 1, 2])
```

#### 3.1.5.2 基本单元模块

本实验仅实现 VGG19 的推断过程，因此不需要实现反向传播计算和参数的更新。仅需实现层的初始化、参数初始化、前向传播计算、参数加载等基本操作。VGG19 网络包含卷积层、ReLU 层、最大池化层、全连接层和 Softmax 层。其中全连接层、ReLU 层和 Softmax 层可以直接使用第2.1节实验中已经实现的相应网络层，本节重点介绍卷积层和池化层的实现。此外还需实现一个 flatten（扁平化）层，用在 VGG19 中第一个全连接层之前，用于将最大池化层（pool5）输出的四维特征图矩阵变形为二维矩阵作为全连接层的输入。最大池化层和 flatten 层中没有参数，不包含参数初始化和参数加载操作。

##### 1) 卷积层

卷积层的实现如代码示例3.2所示，其中定义了以下成员函数：

- **层的初始化**：需要定义卷积的超参数，包括卷积核的高（或宽）$K$、输入特征图的通道数 $C_{in}$、输出特征图的通道数 $C_{out}$、特征图边界扩充大小 $p$、卷积步长 $s$ 等。

- **参数初始化**：卷积层的参数包括权重（卷积核）和偏置。与全连接层类似，通常用高斯随机数来初始化权重的值，而将偏置的所有值初始化为 0。

- **前向传播计算**：根据公式(3.1)和(3.3)可进行卷积层的前向传播计算。首先利用公式(3.1)对输入特征图进行边界扩充。之后利用公式(3.3)，在边界扩充后的特征图上滑动卷积窗口，依次计算每个窗口内的特征图数据与卷积核的内积并加上偏置得到输出特征图。在工程实现中，最简单直接的实现方式是利用四重循环计算输出特征图所有位置的值。由于 VGG19 网络中的所有卷积层都是 $3 \times 3$ 卷积核，即 $K = 3$，边界扩充大小 $p = 1$，步长 $s = 1$，因此 VGG19 网络中的所有卷积层输出特征图的高和宽与输入特征图相同。

- **参数加载**：从该函数的输入中读取本层的权重 $W$ 和偏置 $b$。

**代码示例3.2 卷积层的实现**

```python
# file: layer_2.py

class ConvolutionalLayer(object):
    def __init__(self, kernel_size, channel_in, channel_out, padding, stride):
        # 卷积层的初始化
        self.kernel_size = kernel_size
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.padding = padding
        self.stride = stride

    def init_param(self, std=0.01):   # 参数初始化
        self.weight = np.random.normal(loc=0.0, scale=std,
                        size=(self.channel_in, self.kernel_size,
                              self.kernel_size, self.channel_out))
        self.bias = np.zeros([self.channel_out])

    def forward(self, input):         # 前向传播的计算
        self.input = input             # [N, C, H, W]
        height = self.input.shape[2] + self.padding * 2
        width = self.input.shape[3] + self.padding * 2
        self.input_pad = np.zeros([self.input.shape[0],
                                   self.input.shape[1],
                                   height,
                                   width])
        self.input_pad[:, :,
            self.padding : self.padding + self.input.shape[2],
            self.padding : self.padding + self.input.shape[3]] = self.input

        height_out = (height - self.kernel_size) // self.stride + 1
        width_out  = (width - self.kernel_size) // self.stride + 1
        self.output = np.zeros([self.input.shape[0],
                                self.channel_out,
                                height_out,
                                width_out])
        for idxn in range(self.input.shape[0]):
            for idxc in range(self.channel_out):
                for idxh in range(height_out):
                    for idxw in range(width_out):
                        # TODO: 计算卷积层的前向传播，即特征图与卷积核的内积再加偏置
                        self.output[idxn, idxc, idxh, idxw] = \
                            np.sum(self.input_pad[idxn, :,
                                   idxh * self.stride : idxh * self.stride + self.kernel_size,
                                   idxw * self.stride : idxw * self.stride + self.kernel_size]
                                   * self.weight[:, :, :, idxc]) + self.bias[idxc]
        return self.output

    def load_param(self, weight, bias):    # 参数加载
        self.weight = weight
        self.bias = bias
```

##### 2) 最大池化层

最大池化层的实现如代码示例3.3所示，其中定义了以下成员函数：

- **层的初始化**：需要定义最大池化的超参数，包括池化窗口的高（或宽）$K$ 和池化步长 $s$。

- **前向传播计算**：根据公式(3.7)可计算最大池化层的前向传播结果，即输出特征图的某一位置的值为输入特征图的对应池化窗口中的最大值。由于输出特征图的每个位置的值都是输入特征图的对应池化窗口中的最大值，因此最简单直接的实现方式是用四重循环来计算输出特征图中所有位置的值。

**代码示例3.3 最大池化层的实现**

```python
# file: layer_2.py

class MaxPoolingLayer(object):
    def __init__(self, kernel_size, stride):     # 最大池化层的初始化
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, input):                     # 前向传播的计算
        start_time = time.time()
        self.input = input                         # [N, C, H, W]
        self.index = np.zeros(self.input.shape)
        height_out = (self.input.shape[2] - self.kernel_size) // self.stride + 1
        width_out  = (self.input.shape[3] - self.kernel_size) // self.stride + 1
        self.output = np.zeros([self.input.shape[0],
                                self.input.shape[1],
                                height_out,
                                width_out])
        for idxn in range(self.input.shape[0]):
            for idxc in range(self.input.shape[1]):
                for idxh in range(height_out):
                    for idxw in range(width_out):
                        # TODO: 计算最大池化层的前向传播，取池化窗口内的最大值
                        self.output[idxn, idxc, idxh, idxw] = \
                            np.max(self.input[idxn, idxc,
                                   idxh * self.stride : idxh * self.stride + self.kernel_size,
                                   idxw * self.stride : idxw * self.stride + self.kernel_size])
        return self.output
```

##### 3) Flatten 层

Flatten 层的实现如代码示例3.4所示，定义了以下成员函数：

- **层的初始化**：flatten 层用于改变特征图的维度，将输入特征图中每个样本的特征平铺成一个向量。初始化 flatten 层时需要定义输入特征图和输出特征图的维度。

- **前向传播计算**：假设输入特征图 $X$ 的维度为 $N \times C \times H \times W$，其中 $N$ 是输入的样本个数（在本实验中 $N = 1$），$C$ 是输入的通道数，$H$ 和 $W$ 是输入特征图的高和宽。将输入特征图中每个样本的特征平铺成一个向量后，输出特征图的维度变为 $N \times (CHW)$。

注意 VGG19 官方模型所使用的深度学习平台 MatConvNet[5] 的特征图存储方式与本实验不同。MatConvNet 中特征图维度为 $N \times H \times W \times C$，而本实验中特征图 $X$ 的维度为 $N \times C \times H \times W$。因此为避免使用官方模型计算出现错误，flatten 层在改变输入特征图的维度前，需要将输入特征图进行维度交换，保持与 MatConvNet 的特征图存储方式一致。

**代码示例3.4 Flatten 层的实现**

```python
# file: layer_2.py

class FlattenLayer(object):
    def __init__(self, input_shape, output_shape):
        # 层的初始化
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input):
        # 前向传播的计算
        self.input = np.transpose(input, [0, 2, 3, 1])
        # self.input: [N, height, width, Channel]
        self.output = self.input.reshape([self.input.shape[0]] + list(self.output_shape))
        return self.output
```

#### 3.1.5.3 网络结构模块

与第2.1节实验类似，本实验的网络结构模块也用一个类来定义 VGG19 神经网络，用类的成员函数来定义 VGG19 的初始化、建立网络结构、神经网络参数初始化等基本操作。VGG19 的网络结构模块的实现如代码示例3.5所示，其中定义了以下成员函数：

- **神经网络初始化**：确定神经网络相关的超参数。为方便起见，本实验在网络初始化时仅设定每层的名称，在建立网络结构时再设定每层的具体超参数。

- **建立网络结构**：定义整个神经网络的拓扑结构，设定每层的超参数，实例化基本单元模块中定义的网络层并将这些层堆叠，组成 VGG19 网络结构。根据表3.1中 VGG19 的网络结构和每层的超参数进行实例化。注意每个卷积层和前 2 个全连接层（fc6 层和 fc7 层）后面都跟随有 ReLU 激活函数层。此外，pool5 层和 fc6 层中间有一个 flatten 层，用于改变特征图的维度。最后是 Softmax 层计算分类概率。

- **神经网络参数初始化**：依次调用神经网络中包含参数的网络层的参数初始化函数。在本实验中，VGG19 中的 16 个卷积层和 3 个全连接层包含参数，因此需要依次调用其参数初始化函数。

**代码示例3.5 VGG19 的网络结构模块的实现**

```python
# file: vgg_cpu.py

class VGG19(object):
    def __init__(self, param_path='imagenet-vgg-verydeep-19.mat'):
        # 神经网络的初始化
        self.param_path = param_path
        self.param_layer_name = [
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
            'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2',
            'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
            'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2',
            'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
            'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2',
            'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5',
            'flatten', 'fc6', 'relu6', 'fc7', 'relu7', 'fc8',
            'softmax'
        ]

    def build_model(self):
        # 建立网络结构
        # TODO: 定义 VGG19 的网络结构
        self.layers = {}
        self.layers['conv1_1'] = ConvolutionalLayer(3, 3, 64, 1, 1)
        self.layers['relu1_1'] = ReLULayer()
        self.layers['conv1_2'] = ConvolutionalLayer(3, 64, 64, 1, 1)
        self.layers['relu1_2'] = ReLULayer()
        self.layers['pool1'] = MaxPoolingLayer(2, 2)
        # ... 省略中间层 ...
        self.layers['conv5_4'] = ConvolutionalLayer(3, 512, 512, 1, 1)
        self.layers['relu5_4'] = ReLULayer()
        self.layers['pool5'] = MaxPoolingLayer(2, 2)
        self.layers['flatten'] = FlattenLayer([512, 7, 7], [512*7*7])
        self.layers['fc6'] = FullyConnectedLayer(512*7*7, 4096)
        self.layers['relu6'] = ReLULayer()
        self.layers['fc7'] = FullyConnectedLayer(4096, 4096)
        self.layers['relu7'] = ReLULayer()
        self.layers['fc8'] = FullyConnectedLayer(4096, 1000)
        self.layers['softmax'] = SoftmaxLossLayer()

        self.update_layer_list = []
        for layer_name in self.layers.keys():
            if 'conv' in layer_name or 'fc' in layer_name:
                self.update_layer_list.append(layer_name)

    def init_model(self):
        # 神经网络参数初始化
        for layer_name in self.update_layer_list:
            self.layers[layer_name].init_param()
```

#### 3.1.5.4 网络推断模块

VGG19 的网络推断模块的实现如代码示例3.6所示。与第2.1节的实验类似，网络推断模块同样包含 VGG19 网络的前向传播、网络参数的加载、推断函数主体等操作，这些操作用 VGG19 神经网络类的成员函数来定义：

- **神经网络的前向传播**：前向传播的输入是预处理后的图像。首先将预处理后的图像输入到 VGG19 网络的第一层；然后根据之前定义的 VGG19 网络的结构，顺序依次调用每层的前向传播函数，每层的输出作为下一层的输入。由于 VGG19 中的网络层数较多，可以利用网络初始化时定义的层队列，建立循环实现前向传播。

- **神经网络参数的加载**：利用官方训练好的 VGG19 模型参数，依次将其中的参数加载到 VGG19 对应的层中。本实验使用的官方模型的下载地址为 http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat 。VGG19 中包含参数的网络层是卷积层和全连接层，可以根据层的编号依次读入对应卷积层和全连接层的权重和偏置。注意在本实验的神经网络初始化中，在 pool5 层和 fc6 层之间添加了 flatten 层来改变特征图的维度，而官方提供的模型不包含 flatten 层，因此 fc6 层及之后的层在读取参数时需要偏移。同时值得注意的是，VGG19 官方模型使用的深度学习平台 MatConvNet[5] 的卷积权重的存储方式与本实验不同：MatConvNet 中卷积权重维度为 $H \times W \times C_{in} \times C_{out}$，而本实验中权重的维度为 $C_{in} \times H \times W \times C_{out}$。为防止使用官方模型计算出现错误，在读取卷积层权重时需要对输入权重做维度交换，保持与 MatConvNet 的权重存储方式一致。此外还可以从该模型中读取预处理图像时使用的图像均值。

- **神经网络推断函数主体**：本实验仅需要对给定的一张图像进行分类，因此给定一张预处理好的图像，执行网络前向传播函数即可获得 VGG19 预测的 1000 个类别的分类概率；然后取其中概率最大的类别作为最终预测的分类类别。

**代码示例3.6 VGG19 的网络推断模块的实现**

```python
# file: vgg_cpu.py

def load_model(self):
    # 加载神经网络参数
    params = scipy.io.loadmat(self.param_path)
    self.image_mean = params['normalization'][0][0][0]
    self.image_mean = np.mean(self.image_mean, axis=(0, 1))
    for idx in range(43):
        if 'conv' in self.param_layer_name[idx]:
            weight, bias = params['layers'][0][idx][0][0][0][0]
            # MatConvNet: weights dim [height, width, in_channel, out_channel]
            # Ours:       weights dim [in_channel, height, width, out_channel]
            weight = np.transpose(weight, [2, 0, 1, 3])
            bias = bias.reshape(-1)
            self.layers[self.param_layer_name[idx]].load_param(weight, bias)
        if idx >= 37 and 'fc' in self.param_layer_name[idx]:
            weight, bias = params['layers'][0][idx-1][0][0][0][0]
            weight = weight.reshape([weight.shape[0] * weight.shape[1],
                                     weight.shape[2] * weight.shape[3]])
            self.layers[self.param_layer_name[idx]].load_param(weight, bias)

def forward(self):
    # 神经网络的前向传播
    current = self.input_image
    for idx in range(len(self.param_layer_name)):
        current = self.layers[self.param_layer_name[idx]].forward(current)
    return current

def evaluate(self):
    # 推断函数主体
    prob = self.forward()
    top1 = np.argmax(prob[0])
    print('Classification result: id = %d, Prob = %f' % (top1, prob[0, top1]))
```

#### 3.1.5.5 实验完整流程

完成 VGG19 的每个模块后，就可以用这些模块来实现给定图像的分类。VGG19 进行图像分类的完整流程的实现如代码示例3.7所示。首先实例化 VGG19 网络对应的类，建立 VGG19 的网络结构，并对每层的参数进行初始化，然后从官方模型中加载每层的参数，之后加载给定的图像并进行预处理，最后调用网络推断模块获得最终的图像分类结果。

**代码示例3.7 VGG19 进行图像分类的完整流程实现**

```python
# file: vgg_cpu.py

if __name__ == '__main__':
    vgg = VGG19()
    vgg.build_model()
    vgg.init_model()
    vgg.load_model()
    vgg.load_image('cat1.jpg')
    vgg.evaluate()
```

#### 3.1.5.6 实验运行

根据第3.1.5.1节~第3.1.5.5节的描述补全 `layer_1.py`、`layer_2.py`、`vgg_cpu.py` 代码，并通过 Python 运行 `.py` 代码。具体可以参考以下步骤。

**1) 环境申请**

申请实验环境并登录云平台，云平台上 `/opt/code_chap_2_3/code_chap_2_3_student/` 目录下是本实验的示例代码。

```bash
# 登录云平台
ssh root@XXX.XXX.XXX.XXX -p XXXXX

# 进入 code_chap_2_3_student 目录
cd /opt/code_chap_2_3/code_chap_2_3_student
```

**2) 代码实现**

补全 `stu_upload` 中的 `layer_1.py`、`layer_2.py`、`vgg_cpu.py` 文件。

```bash
# 进入实验目录
cd exp_3_1_VGG

# 补全 layer_1.py layer_2.py vgg_cpu.py
vim stu_upload/layer_1.py
vim stu_upload/layer_2.py
vim stu_upload/vgg_cpu.py
```

**3) 运行实验**

```bash
# 运行完整实验
python main_exp_3_1.py
```

### 3.1.6 实验评估

为验证实验代码的正确性，选择如图3.2所示猫咪的图像进行分类测试。该猫咪图像的真实类别为 tabby cat，对应 ImageNet 数据集类别编号为 281。本实验的正确结果是将该图像的类别编号判断为 281。通过查询 ImageNet 数据集类别编号对应的具体类别，编号 281 对应的具体类别为 tabby cat，说明利用 VGG19 网络推断得到了正确的图像类别。

> **图3.2 测试猫咪图像示例**（tabby cat，ImageNet 类别编号 281）

本实验的评估标准设定如下：

- **60分标准**：给定卷积层和池化层的前向传播输入矩阵和参数值，可以得到正确的前向传播输出矩阵。
- **80分标准**：建立 VGG19 网络后，给定 VGG19 的网络参数值和输入图像，可以得到正确的层输出结果。
- **100分标准**：建立 VGG19 网络后，给定 VGG19 的网络参数值和输入图像，可以得到正确的 Softmax 层输出结果和正确的图像分类结果。

### 3.1.7 实验思考

1. 在实现深度神经网络基本单元时，如何确保一个网络层的实现是正确的？

2. 在实现深度神经网络后，如何确保整个网络的实现是正确的？如果是网络中的某个层计算有误，如何快速定位到有错误的层？

3. 如何计算深度神经网络中每层的计算量（乘法数量和加法数量）？如何计算整个网络的前向传播时间和网络中每层的前向传播时间？深度神经网络中每层的计算量和每层的前向传播时间之间有什么关系？
