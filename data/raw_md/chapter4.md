# 第四章 编程框架使用

## 章节概述

本章学习使用深度学习编程框架实现算法的方法，以PyTorch为主要框架，涵盖编程框架概述、PyTorch核心概念、NumPy基础、PyTorch编程模型、张量操作、自动微分、模型构建、推理与训练实现、数据加载、模型保存与部署等实用技能。

---

## 4.1 编程框架概述

### 4.1.1 为什么需要编程框架

深度学习算法理论复杂、代码实现工作量大，有必要将算法中的常用操作封装成组件提供给程序员，以提高深度学习算法开发效率。

**编程框架的三层价值**：
1. 深度学习算法具有多层结构，每层运算由基本操作构成（卷积、池化、激活等），存在大量共性运算
2. 将这些共性运算封装起来，可提高编程实现效率
3. 硬件程序员可基于硬件特征有针对性地优化这些操作，充分发挥硬件效率

### 4.1.2 深度学习编程框架的定义

深度学习编程框架：将深度学习算法中的基本操作封装成一系列组件，构成一套深度学习框架。帮助算法开发人员更简单地实现已有算法或设计新算法，也有助于硬件程序员更有针对性地优化关键操作。

### 4.1.3 国内外主流编程框架

| 框架 | 发布者 | 首次发布时间 | 特点 |
|------|--------|-------------|------|
| **PyTorch** | Facebook(Meta) | 2017 | 动态计算图，Python优先，学术界主流 |
| **TensorFlow** | Google | 2015 | 静态计算图（2.0后支持动态图），工业界广泛应用 |
| **Keras** | Google | 2015 | 高层API，简单易用 |
| **Caffe** | BVLC | 2013 | 早期经典框架，图像处理领域 |
| **PaddlePaddle** | 百度 | 2018 | 中文文档丰富，国产框架代表 |
| **MindSpore** | 华为 | 2019 | 支持端-边-云全场景 |

---

## 4.2 PyTorch概述

### 4.2.1 PyTorch的起源

- **Torch**：瑞士IDIAP研究所2002年发布，基于Lua语言的机器学习框架，核心是易于使用的神经网络，在GPU上有较高性能
- **PyTorch = Python + Torch**：2017年Facebook AI Research开源

### 4.2.2 PyTorch的设计理念

1. **基于Python**：融入Python开源生态
2. **把研究人员放在首位**：易用、直观
3. **高性能**：底层C++/CUDA优化
4. **内部实现简单**：节省学习时间

### 4.2.3 PyTorch版本演进

| 版本 | 时间 | 关键特性 |
|------|------|---------|
| 0.1 | 2017.1 | 首个版本发布 |
| 1.0 | 2018.12 | 生产就绪，TorchScript |
| 1.3 | 2019.10 | 移动端支持，量化功能 |
| 1.13 | 2022.11 | BetterTransformer稳定版 |
| 2.0 | 2023.3 | 编译模式(torch.compile) |
| 2.4 | 2024.7 | 最新版本 |

### 4.2.4 PyTorch生态

- torchvision：视觉模型和数据集
- torchaudio：音频处理
- torchtext：自然语言处理
- torchrec：推荐系统
- PyTorch Lightning：高层训练框架
- HuggingFace Transformers：预训练模型库

---

## 4.3 NumPy基础

### 4.3.1 NumPy概述

NumPy是Python的高性能科学计算库，提供大量库函数和操作，支持高维数组的批量化处理，能够高效处理机器学习、计算机视觉任务。

NumPy中最重要的数组类为**ndarray**。

### 4.3.2 创建ndarray

```python
import numpy as np

# 从列表创建
my_data1 = np.array([1, 2])
my_data2 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], np.float32)

# 序列创建
my_data3 = np.arange(3)              # [0, 1, 2]
my_data4 = np.arange(3, 9, 3)        # [3, 6]

# 特殊数组
my_data5 = np.zeros((3, 2))          # 全0数组
my_data6 = np.ones((3, 2))           # 全1数组
my_data7 = np.full((3, 2), 7)        # 填充指定值
my_data8 = np.eye(3, 3)              # 单位矩阵
my_data9 = np.empty((3, 2))          # 未初始化数组
my_data10 = np.random.random((3, 2)) # 0到1随机值
```

### 4.3.3 np.eye用于One-Hot编码

```python
# one-hot编码标签
my_data = np.eye(3)[[0, 1, 0, 2]]
# 输出: [[1,0,0], [0,1,0], [1,0,0], [0,0,1]]
```

### 4.3.4 ndarray的重要属性

| 属性 | 说明 |
|------|------|
| ndarray.ndim | 数组维度（轴数） |
| ndarray.shape | 每个维度的尺寸 |
| ndarray.size | 元素总个数 |
| ndarray.dtype | 元素数据类型（默认float64） |
| ndarray.itemsize | 每个元素的字节数 |

---

## 4.4 PyTorch编程模型及基本用法

### 4.4.1 张量（Tensor）

张量是PyTorch中的基本数据结构，可理解为支持GPU加速和自动微分的多维数组。

**创建张量**：
- `torch.tensor(data)`：从数据创建
- `torch.zeros()`、`torch.ones()`、`torch.randn()`：特殊张量
- `torch.from_numpy(ndarray)`：从NumPy数组转换

**张量属性**：shape、dtype、device（所在设备CPU/GPU）

### 4.4.2 张量操作

- **索引与切片**：与NumPy类似
- **变形**：`reshape()`、`view()`、`transpose()`、`permute()`
- **数学运算**：加减乘除、矩阵乘法(`torch.matmul`、`@`)
- **设备迁移**：`.to('cuda')` / `.to('cpu')`
- **数据类型转换**：`.float()`、`.int()`、`.half()`

### 4.4.3 自动微分（Autograd）

自动微分是深度学习框架的核心特性：

```python
x = torch.randn(1, 10)
W = torch.randn(20, 10, requires_grad=True)  # 需要梯度
output = torch.matmul(W, x.t())
loss = output.sum()
loss.backward()  # 自动计算梯度
print(W.grad)    # 查看梯度
```

**关键方法**：
- `requires_grad=True`：标记需要计算梯度的张量
- `.backward()`：反向传播计算梯度
- `.grad`：查看累积的梯度
- `.zero_grad()`：清零梯度
- `torch.no_grad()`：上下文管理器，禁用梯度计算

### 4.4.4 神经网络模块（torch.nn）

**预定义层**：
- `nn.Linear(in_features, out_features)`：全连接层
- `nn.Conv2d(in_channels, out_channels, kernel_size)`：二维卷积
- `nn.MaxPool2d(kernel_size)`：最大池化
- `nn.BatchNorm2d(num_features)`：批归一化
- `nn.ReLU()` / `nn.Sigmoid()` / `nn.Tanh()`：激活函数
- `nn.Dropout(p)`：Dropout层
- `nn.Embedding(num_embeddings, embedding_dim)`：嵌入层

### 4.4.5 构建模型的三种方式

**1. Sequential（顺序模型）**：
```python
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)
```

**2. Module子类化（推荐）**：
```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)
```

**3. Functional API**：灵活定义多输入多输出模型。

### 4.4.6 损失函数

- `nn.MSELoss()`：均方误差损失
- `nn.CrossEntropyLoss()`：交叉熵损失（自动包含Softmax）
- `nn.BCELoss()`：二分类交叉熵损失
- `nn.BCEWithLogitsLoss()`：含Sigmoid的二分类交叉熵
- `nn.L1Loss()`：L1损失

### 4.4.7 优化器

- `torch.optim.SGD(params, lr, momentum)`：随机梯度下降
- `torch.optim.Adam(params, lr)`：Adam优化器
- `torch.optim.RMSprop(params, lr)`：RMSprop
- `torch.optim.AdamW(params, lr)`：解耦权重衰减的Adam

### 4.4.8 学习率调度器

- `StepLR`：固定步长衰减
- `CosineAnnealingLR`：余弦退火
- `ReduceLROnPlateau`：验证指标停滞时衰减
- `LambdaLR`：自定义函数调整

---

## 4.5 基于PyTorch的模型推理实现

### 4.5.1 推理流程

1. **加载模型**：实例化模型类并加载训练好的参数
2. **切换评估模式**：`model.eval()`
3. **数据预处理**：与训练时相同的预处理流程
4. **关闭梯度计算**：`with torch.no_grad():`
5. **前向传播**：输入数据获得预测结果
6. **后处理**：将模型输出转化为可读的预测结果

### 4.5.2 评估模式 vs 训练模式

- `model.eval()`：切换到评估模式
  - Dropout层被禁用
  - BatchNorm使用全局统计量（running mean/var）而非batch统计量
- `model.train()`：切换到训练模式

### 4.5.3 常用评估指标

- **准确率（Accuracy）**：正确预测数/总样本数
- **精确率（Precision）**：TP/(TP+FP)
- **召回率（Recall）**：TP/(TP+FN)
- **F1分数**：2*P*R/(P+R)
- **混淆矩阵（Confusion Matrix）**

---

## 4.6 基于PyTorch的模型训练实现

### 4.6.1 训练循环基本步骤

```python
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # 1. 梯度清零
        optimizer.zero_grad()

        # 2. 前向传播
        output = model(data)

        # 3. 计算损失
        loss = criterion(output, target)

        # 4. 反向传播
        loss.backward()

        # 5. 参数更新
        optimizer.step()
```

### 4.6.2 数据处理

**Dataset自定义**：
```python
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
```

**DataLoader**：
```python
train_loader = DataLoader(
    dataset, batch_size=32, shuffle=True,
    num_workers=4, pin_memory=True
)
```

关键参数：
- `batch_size`：批量大小
- `shuffle`：是否打乱数据
- `num_workers`：多进程加载
- `pin_memory`：锁页内存加速GPU传输

### 4.6.3 数据预处理与增强

**torchvision.transforms**：
- `transforms.ToTensor()`：转为张量并归一化到[0,1]
- `transforms.Normalize(mean, std)`：标准化
- `transforms.Resize(size)`：缩放
- `transforms.RandomCrop(size)`：随机裁剪
- `transforms.RandomHorizontalFlip()`：随机水平翻转
- `transforms.ColorJitter()`：颜色抖动
- `transforms.Compose()`：组合多个变换

### 4.6.4 混合精度训练

使用FP16加速训练并节省显存：
```python
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    output = model(data)
    loss = criterion(output, target)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 4.6.5 模型保存与加载

**保存和加载模型参数（推荐）**：
```python
# 保存
torch.save(model.state_dict(), 'model_weights.pth')
# 加载
model.load_state_dict(torch.load('model_weights.pth'))
```

**保存和加载完整模型**：
```python
# 保存
torch.save(model, 'model_full.pth')
# 加载
model = torch.load('model_full.pth')
```

**保存检查点**：
```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}
torch.save(checkpoint, 'checkpoint.pth')
```

---

## 4.7 模型部署

### 4.7.1 模型转换

- **TorchScript**：PyTorch的序列化优化表示，支持C++环境运行
- **ONNX（Open Neural Network Exchange）**：跨框架模型交换格式
- **TensorRT**：NVIDIA推理优化引擎

### 4.7.2 推理优化

- 批处理推理（Batched Inference）
- 混合精度推理（FP16/INT8）
- 算子融合和图优化
- TorchDynamo + torch.compile加速（PyTorch 2.0+）

### 4.7.3 部署方式

- **本地推理服务**：直接在Python环境运行
- **C++部署**：通过TorchScript或LibTorch
- **REST API服务**：结合Flask/FastAPI构建推理API
- **移动端部署**：PyTorch Mobile（Android/iOS）
- **边缘设备部署**：TensorRT、OpenVINO

---

## 4.8 本章小结

本章介绍了以PyTorch为代表的深度学习编程框架使用方法：

- **编程框架的价值**：将共性操作（卷积、池化等）封装为组件，提高开发效率
- **PyTorch核心概念**：张量(Tensor)、自动微分(Autograd)、神经网络模块(nn.Module)
- **模型构建三种方式**：Sequential、Module子类化、Functional API
- **训练流程五步骤**：梯度清零 -> 前向传播 -> 计算损失 -> 反向传播 -> 参数更新
- **数据处理**：Dataset自定义 + DataLoader批量加载 + transforms数据增强
- **模型评估与保存**：model.eval() + state_dict保存 + 评估指标
- **模型部署**：TorchScript、ONNX、C++部署、移动端部署等多种方式
