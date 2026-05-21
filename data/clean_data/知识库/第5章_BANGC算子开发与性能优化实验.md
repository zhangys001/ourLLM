# 第5章 智能编程语言

智能编程语言是连接智能编程框架和智能计算硬件的桥梁。本章将通过具体实验阐述智能编程语言的开发、优化和集成方法。具体而言，第5.1节介绍如何使用智能编程语言BANG C实现用户自定义的高性能库算子Sigmoid，并将其集成到PyTorch框架中。第5.2节进一步介绍如何使用智能编程语言进行矩阵乘性能优化以充分发挥MLU硬件潜力。

## 5.1 智能编程语言算子开发与集成实验（BANG C开发实验）

### 5.1.1 实验目的

掌握使用智能编程语言BANG C进行算子开发、编译扩展高性能库算子，并集成到PyTorch框架中的方法和流程。能够用BANG C实现Sigmoid算子，并集成进PyTorch的推断网络高效地运行在MLU硬件上。实验工作量：代码量约150行，实验时间约10小时。

### 5.1.2 背景介绍

#### 5.1.2.1 BANG C简介

BANG C语言采用异构混合编程和编译。一个完整的程序包括主机端（Host）程序和设备端（Device）程序，主机端程序和设备端程序可以写在同一份文件中。混合异构程序使用CNCC编译器进行编译，编译器会自动拆分主机端和设备端代码分别编译，最后链接成一个可执行程序。主机端程序主要调用运行时库接口执行内存申请、释放、拷贝，Kernel的控制执行；设备端程序使用BANG C特定的语法规则执行计算部分和并行任务。在BANG C算子开发中，用户可以在主机端输入数据，做一定处理后，通过一个Kernel启动函数将相应输入数据传给设备端，设备端进行计算后，再将计算结果拷回主机端。

#### 5.1.2.2 编译器（CNCC）

CNCC编译器将使用智能编程语言（BANG C）编写的程序编译成MLU架构指令。为了填补高层智能编程语言和底层MLU硬件指令间的鸿沟，MLU的编译器通过寄存器分配、地址空间推断、全局指令调度等技术进行编译优化，以提升最终二进制程序的性能。

CNCC的编译过程为：开发者使用BANG C开发出的异构混合程序首先通过CNCC分离为主机端和设备端代码，然后分别编译为对应架构的中间表示*.ll文件，经过优化后编译出汇编*.s文件，设备端汇编代码由CNAS汇编器生成MLU架构的二进制对象*.o文件，然后通过设备端链接器链接为多架构混合*.cnfatbin文件，最后设备端二进制会被链接进主机端二进制生成最终的二进制可执行文件或动态库。在实际使用中，CNCC编译器将自动完成上述过程，直接将BANG C代码编译、汇编、链接生成二进制机器码。

CNCC常用编译选项如下：

| 编译选项 | 说明 |
|---------|------|
| `--help` | 查看CNCC帮助信息 |
| `-E` | 编译器只执行预处理步骤，生成预处理文件 |
| `-S` | 编译器只执行预处理、编译步骤，生成汇编文件 |
| `-c` | 编译器只执行预处理、编译、汇编步骤，生成ELF格式的汇编文件 |
| `-o` | 将输出写入到指定的文件 |
| `-g` | 在编译时产生调试信息 |
| `-O` | 指定编译优化级别，其中-O0不做编译优化 |
| `--target=` | 指定可执行文件的目标主机平台架构，不指定时使用当前主机平台架构，例如x86_64-linux-gnu，aarch64-linux-gnu等 |
| `--bang-arch=` | 指定MLU的第几代架构，例如compute_30 |
| `--bang-mlu-arch=` | 指定MLU的具体架构号，例如mtp_372，可以同时指定多个架构进行fatbin编译 |
| `--bang-stack-on-ldram` | 栈是否放在LDRAM上，默认放在NRAM上。如果该选项开启，栈会放在LDRAM上 |
| `--bang-device-only=` | 面向设备侧编译程序，通常与-S选项配合使用，生成MLISA汇编代码 |
| `-emit-llvm` | 生成中间表示，通常与-S选项配合，生成LLVM IR文件 |
| `-###` | 显示编译的子命令行，用来查看异构混合编译的详细流程 |

#### 5.1.2.3 调试器（CNGDB）

CNGDB是面向智能编程语言的调试器，支持搭载MLU硬件的异构平台调试，即同时支持主机端C/C++代码和设备端BANG C代码的调试，同时两者调试过程的切换对于用户而言也是透明的。此外，针对多核MLU架构的特点，调试器可以支持单核和多核应用程序的调试。CNGDB解决了异构编程模型调试的问题，提升了应用程序开发的效率。

如果要使用CNGDB进行调试，需要在用CNCC编译BANG C文件时添加`-g`选项、选择`-O0`优化级别，以编译生成含有调试信息的二进制文件。编译命令示例：

```
cncc main.mlu -o a.out --bang-arch=compute_30 -g -O0
```

下面以BANG C编写的快速排序程序为例，介绍如何使用CNGDB调试程序。快速排序程序的设备端BANG C代码文件为recursion.mlu。使用CNGDB调试recursion.mlu程序的基本流程主要包含以下几个步骤：断点插入、程序执行、变量打印、单步调试和多核切换等。

#### 5.1.2.4 集成开发环境（CNToolkit和CNStudio）

CNToolkit是基于BANG异构计算平台的编译、调试、分析、运行的工具集。CNToolkit安装包内提供了各个组件的示例代码和用户手册，其中包含CNStudio组件。CNStudio是一款方便在Visual Studio Code（VSCode）中开发调试BANG C语言的编程插件。为了使BANG C语言在编写过程中更加方便快捷，CNStudio基于VSCode编辑器强大的功能和简便的可视化操作提供包括语法高亮、自动补全和程序调试等功能。安装包的具体下载地址参考网站（https://developer.cambricon.com）。

CNStudio插件只支持离线安装，安装CNToolkit后的插件位置为`/usr/local/neuware/data/cnstudio/cnstudio.vsix`。参考用户文档下载并安装CNToolkit后，即可完成CNStudio插件的安装。CNStudio插件安装完毕后，在左侧插件安装界面的搜索框中输入"@installed"即可查询全部插件，若显示CNStudio插件则说明安装成功。如果CNStudio的高亮颜色与VSCode背景颜色有冲突，可通过组合快捷键(Ctrl+k)(Ctrl+t)更改浅色主题。

在创建工程时（以新建一个MLU文件夹为例），每个project都包含三种类型的文件：设备端使用BANG C编写的Kernel程序源文件*.mlu（安装CNStudio插件后，VSCode会自动识别后缀名为mlu的文件），主机端的C++程序main.cpp，以及头文件kernel.h。通过VSCode工具栏中"File"→"Save Workspace As..."，将打开的MLU工程保存起来，方便下次直接打开工程文件。

#### 5.1.2.5 高性能算子库（CNNL和MLU-OPS）

CNNL高性能算子库是基于BANG C开发的算子库集合，CNNL为PyTorch、TensorFlow、PaddlePaddle等开源框架提供了运行在MLU硬件的完备算子集合，用户无需用BANG C开发即可通过CNNL运行主流网络的训练和推理并获得最优性能。MLU-OPS是CNNL算子库的开源版本，提供基于MLU使用C接口或者Python接口开发高性能算子的示例代码。MLU-OPS旨在通过提供示例代码，供开发者参考使用，可用于开发自定义算子，实现对应模型的计算。项目地址：https://github.com/Cambricon/mlu-ops。

一个高性能算子库的头文件mlu_ops.h需要提供Tensor描述符`mluOpTensorDescriptor_t`、矩阵乘运算描述符`mluOpMatMulDescriptor_t`等数据结构，还需要提供`mluOpAbs`和`mluOpMatMul`等算子运算接口，以及配套的数据结构和运行时相关的接口如`mluOpCreateTensorSetDescriptor`、`mluOpCreate`、`mluOpGetQueue`等。核心接口声明如下：

```c
typedef struct mluOpTensorStruct *mluOpTensorDescriptor_t;
typedef struct mluOpMatMulStruct *mluOpMatMulDescriptor_t;

mluOpStatus_t mluOpAbs(mluOpHandle_t handle,
    const mluOpTensorDescriptor_t x_desc,
    const void *x,
    const mluOpTensorDescriptor_t y_desc,
    void *y);

mluOpStatus_t mluOpMatMul(mluOpHandle_t handle,
    const bool is_trans_a,
    const bool is_trans_b,
    const void *alpha,
    const mluOpTensorDescriptor_t a_desc,
    const void *a,
    const mluOpTensorDescriptor_t b_desc,
    const void *b,
    const void *beta,
    const mluOpTensorDescriptor_t c_desc,
    void *c);

mluOpStatus_t mluOpCreateTensorDescriptor(mluOpTensorDescriptor_t *desc);
mluOpStatus_t mluOpCreate(mluOpHandle_t *handle);
mluOpStatus_t mluOpGetQueue(mluOpHandle_t handle, mluQueue_t *queue);
```

MLU-OPS的每个算子都包含主机端代码（如`mlu-ops/bangc-ops/kernels/abs/abs.cpp`文件）和异构混合BANG C代码（如`mlu-ops/bangc-ops/kernels/abs/abs_block.mlu`文件）。其中主机端代码主要完成算子参数处理、MLU-OPS接口封装和Kernel并行规模策略计算等工作。设备端代码包含BANG C源码和Kernel的异构`<<<>>>`核函数调用，实现主要的计算逻辑。MLU-OPS的主要目录结构如下：

```
mlu-ops/bangc-ops/
├── cmake
├── core
├── kernels
│   ├── abs
│   │   ├── abs.cpp
│   │   ├── abs.h
│   │   └── abs_block.mlu
│   ├── ...
├── scripts
├── test
├── CMakeLists.txt
├── README.md
├── build.sh
├── kernel_depends.toml
└── mlu_ops.h
```

更多关于智能编程语言的介绍，详见《智能计算系统》教材第8章。

#### 5.1.2.6 PyTorch框架

MLU版本的PyTorch借助PyTorch自身提供的设备扩展接口，将MLU后端中所包含的算子操作动态注册到PyTorch中，MLU后端可处理MLU上的张量和算子的运算。PyTorch会基于CNNL库在MLU后端实现一些常用算子，并完成一些数据拷贝。

MLU版本的PyTorch兼容原生PyTorch的Python编程接口和原生PyTorch网络模型，支持以在线逐层方式进行训练和以JIT融合方式进行推理。网络权重可以从pth格式文件读取，已支持的分类和检测网络结构由Torchvision管理，可以从Torchvision中读取。对于训练任务，支持float32及定点量化模型。

为了能在Torch模块方便使用MLU设备，MLU版的PyTorch在后端进行了以下扩展：

1. 通过Torch模块可调用MLU后端支持的网络运算。
2. 对MLU暂不支持的算子，并且该算子在MLU后端库中已添加注册，支持该类算子自动切换到CPU上运行。
3. Torch模块中与MLU相关的接口的语义与CPU和GPU的接口语义保持一致。
4. 支持CPU和MLU之间的无缝切换。

### 5.1.3 实验环境

- 硬件平台：MLU云平台环境。
- 软件环境：编程框架PyTorch、CNNL高性能算子库、BANG异构计算平台的开发工具包CNToolkit。

### 5.1.4 实验内容

在BANG C算子开发与集成实验中，在第4.4节自定义PyTorch CPU算子实验的基础上，进一步用智能编程语言BANG C来实现自定义算子Sigmoid的计算逻辑（Kernel函数），通过PyTorch的自定义算子扩展机制（MLUExtension，参考CUDAExtension），将自定义算子Sigmoid集成到编程框架PyTorch中，最后与第4.4节的实现进行精度对比。实验流程主要包括：

1. **BANG C自定义算子的Kernel实现**：采用智能编程语言BANG C实现自定义算子Sigmoid的计算逻辑并进行正确性测试，包括使用BANG C的内置向量函数实现Kernel函数，通过主机端C++代码调用Kernel函数运行并测试功能正确性。

2. **框架算子集成**：通过PyTorch的自定义算子扩展机制对Sigmoid算子进行封装，使其调用方式和高性能库原有MLU算子一致，然后将封装后的MLU算子（BANG C自定义算子和CNNL内置算子统称为MLU算子）集成到PyTorch框架中并进行测试，保证其精度和功能正确。

3. **MLU算子和CPU算子对比测试**：调用PyTorch框架的CPU Sigmoid算子，和MLU Sigmoid自定义算子做精度对比测试。

### 5.1.5 实验步骤

在BANG C算子开发与集成实验中，首先介绍使用BANG C语言开发和集成自定义算子的主要原理和流程，然后分步骤介绍实现一个Sigmoid自定义算子的主程序与核函数、通过pybind11封装自定义算子接口、使用setuptools对自定义算子编译和集成、自定义算子和框架原生算子如何对比测试。

首先，PyTorch编程框架的设计理念为"Python First"，所以集成自定义算子的基本思想就是使用Python语言的胶水能力将自定义算子嵌入到PyTorch框架的算子调用流程中。

在BANG C算子开发与集成实验中，PyTorch框架自定义算子的主要流程如下：

1. **实现主机端和设备端代码**：实现主机端的算子主程序和设备端的算子核函数，其中主程序确定算子的输入输出张量并定义算子接口，核函数实现算子的计算。

2. **添加MLUExtension扩展**：在PyTorch框架的`torch.utils.cpp_extension`模块中添加MLUExtension函数定义，添加方式类似CUDAExtension或CppExtension。CUDAExtension和CppExtension是返回`setuptools.Extension`类的函数。其中，CppExtension扩展的对象为CPU，支持的语言是C++，提供了一些头文件和PyTorch C++相关的静态链接库、动态链接库；而CUDAExtension扩展的对象是GPU，支持的语言是CUDAC++。因此对于DLP的BCL编程语言，需要添加DLPExtension来完成编译器查找、编译参数指定等一系列操作。

3. **编写setup.py脚本并编译**：编写setup.py脚本，使用setuptools工具将BCL源码编译为动态库，并通过pybind11将算子的API从BCL语言封装为Python语言。

然后，结合实验代码分步骤实现、编译、集成、测试一个Sigmoid算子（函数名称为`active_sigmoid_mlu`），算子为单输入单输出，函数接口为`torch::Tensor active_sigmoid_mlu(torch::Tensor x)`。

BANG C算子开发与集成实验中Sigmoid算子的代码目录结构如下：

```
# Sigmoid自定义算子实验根目录
├── README.md：描述算子功能的说明文档
├── mlu_custom_ext：生成的module模块用于在python层导入
│   ├── __init__.py：python包固有文件
│   ├── mlu：mlu代码文件，根据实际情况自己创建，在setup.py中修改即可
│   │   ├── include：头文件目录（头文件和实现分离，属于代码习惯，建议采用此布局）
│   │   │   ├── bang_sigmoid_sample.h：实现对mlu函数的封装
│   │   │   ├── kernel.h：BANG C代码中的宏，良好的组织代码的需要
│   │   │   └── custom_ops.h：算子对外头文件
│   │   └── src
│   │       ├── bang_sigmoid.cpp：对PyTorch层面Tensor的封装，和自定义算子中xxx_internal的实现类似
│   │       └── bang_sigmoid_sample.mlu：核心BangC实现
│   └── mlu_functions：合理的组织自己的代码方便后续调用
│       ├── __init__.py：包必备文件
│       └── mlu_functions.py：对C++代码的封装
├── setup.py：构建包的脚本
└── tests
    └── test_sigmoid.py：对绑定代码的python侧测试
```

Sigmoid实验中各文件的函数释义如下：

| 文件名 | 函数名 | 释义 |
|--------|--------|------|
| test_sigmoid.py | test_forward_with_shape() | pytest测试函数 |
| test_sigmoid.py | test_backward_with_shape() | pytest测试函数 |
| mlu_functions.py | forward() | 继承torch.autograd.function类的正向sigmoid接口 |
| mlu_functions.py | backward() | 继承torch.autograd.function类的反向sigmoid接口 |
| bang_sigmoid.cpp | active_sigmoid_mlu() | C++函数接口，属于torch_mlu命名空间，操作的是PyTorch的Tensor，被pybind11封装进libmlu_custom_ext库 |
| bang_sigmoid_sample.mlu | bang_sigmoid_kernel_entry() | BANG C编程中主机端的C++函数入口，被PyTorch的C++接口调用 |
| bang_sigmoid_sample.mlu | bang_sigmoid_kernel() | BANG C编程中设备端核函数，被主机端程序使用<<<>>>核函数调用 |
| bang_sigmoid_sample.mlu | bang_sigmoid_sample() | C++测试用例封装的函数接口，仅供C++测试使用，PyTorch自定义算子中并未调用 |

在完成实验代码补全后，需要执行的命令步骤如下：

1. 在根目录下执行`python setup.py install`，完成定义算子的编译和安装。
2. 进入tests目录，执行`python test_sigmoid.py`测试完成精度测试。
3. 对于性能测试，在test_sigmoid.py文件中添加计时函数后，执行`python test_sigmoid.py`。

#### 5.1.5.1 实现Sigmoid主程序

在BANG C算子开发与集成实验中，Sigmoid主程序通过PyTorch提供的能力获取PyTorch Tensor提供的Tensor数据指针，数据指针在Host侧无法操作，因此需要实现一个Device函数计算Sigmoid。主程序调用了核函数`bang_sigmoid_kernel_entry`实现对Device上的数据计算。

Sigmoid主程序的C++实现（bang_sigmoid.cpp文件）如下：

```cpp
// filename: bang_sigmoid.cpp
// 所属实验：5.1 BANG C算子开发与集成实验 - Sigmoid主程序
// 核心任务：在PyTorch C++层封装Sigmoid算子，获取Tensor数据指针并调用MLU核函数
// 对应原理：通过PyTorch的Tensor接口获取MLU设备端数据指针，然后调用bang_sigmoid_kernel_entry核函数在MLU上执行Sigmoid计算

#include "bang_sigmoid_sample.h"
#include "custom_ops.h"
#include "ATen/Tensor.h"
#include "aten/operators/bang/bang_kernel.h"
#include "aten/operators/bang/internal/bang_internal.h"

using namespace torch_mlu;

torch::Tensor active_sigmoid_mlu(torch::Tensor x) {
    // 获取连续存储的输入Tensor
    auto x_contiguous = torch_mlu::cnnl_contiguous(x);
    auto x_impl = getMluTensorImpl(x_contiguous);
    auto x_ptr = x_impl->mlu_data_ptr();

    // 创建与输入同形状的输出Tensor
    auto y = at::empty_like(x_contiguous);
    auto y_contiguous = torch_mlu::cnnl_contiguous(y);
    auto y_impl = getMluTensorImpl(y_contiguous);
    auto y_ptr = y_impl->mlu_data_ptr();

    // 获取Tensor元素总数
    int32_t size = x_contiguous.numel();
    cnrtQueue_t queue = getCurQueue();

    // 调用核函数入口，在MLU设备端执行Sigmoid计算
    bang_sigmoid_kernel_entry(queue,
        reinterpret_cast<float *>(y_ptr),
        reinterpret_cast<float *>(x_ptr),
        size);

    return y;
}

// 通过pybind11将C++函数暴露给Python层
PYBIND11_MODULE(libmlu_custom_ext, m) {
    m.def("active_sigmoid_mlu", &active_sigmoid_mlu);
}
```

主程序的执行流程为：

1. 获取输入Tensor的连续存储版本，并通过`getMluTensorImpl`获取内部实现指针。
2. 通过`mlu_data_ptr()`获取输入数据在MLU设备端的指针。
3. 创建一个与输入同形状的空输出Tensor。
4. 获取当前运行时队列`cnrtQueue_t`。
5. 调用`bang_sigmoid_kernel_entry`核函数入口，传入队列、输出指针、输入指针和元素个数。
6. 返回计算结果Tensor。
7. 通过pybind11的`PYBIND11_MODULE`宏将`active_sigmoid_mlu`函数暴露给Python。

#### 5.1.5.2 实现Sigmoid核函数

在BANG C算子开发与集成实验中，需要在`bang_sigmoid_sample.mlu`文件中实现`bang_sigmoid_kernel_entry`函数并通过头文件对外暴露。

在Sigmoid核函数的实现中，为了充分利用MLU硬件计算能力，使用了向量计算函数来完成Sigmoid的运算。为了使用向量计算函数必须满足两个前提：第一是调用计算函数时数据的输入和输出存放位置必须在NRAM上，因此必须在计算前使用memcpy将数据从GDRAM拷贝到NRAM上，在计算完成后将结果从NRAM拷贝到GDRAM上；第二是向量操作的输入规模如果不能被多核和多次循环整除时需要增加分支来处理余数部分。

由于NRAM大小的限制，不能一次性将所有数据全部拷贝到NRAM上执行，因此需要对原输入数据进行分块。分块的规模在满足NRAM大小和函数对齐要求的前提下由用户指定，这里设置为`NRAM_LIMIT_SIZE = FLOOR_ALIGN(MAX_NRAM_SIZE / 2, 64)`。分块的重点在于余数段的处理：由于通常情况下输入不一定是NRAM_LIMIT_SIZE的倍数，所以最后会有一部分长度小于NRAM_LIMIT_SIZE、大于0的余数段，需要特别注意该部分数据的处理逻辑。

Sigmoid核函数的BANG C实现（bang_sigmoid_sample.mlu文件）如下：

```c
// filename: bang_sigmoid_sample.mlu
// 所属实验：5.1 BANG C算子开发与集成实验 - Sigmoid核函数
// 核心任务：在MLU设备端实现Sigmoid算子的向量化计算，支持多核任务并行
// 对应原理：使用BANG C内置向量函数__bang_active_sigmoid在NRAM上执行Sigmoid运算，通过分块策略处理NRAM容量限制

#include <bang_sigmoid_sample.h>
#include <kernel.h>

// NRAM缓冲区声明，用于存放输入和输出数据
__nram__ char NRAM_BUFFER[MAX_NRAM_SIZE];

template <typename T>
__mlu_global__ void bang_sigmoid_kernel(T *d_dst, T *d_src, int N) {
    // 计算NRAM分块大小：取NRAM总大小的一半并对64字节对齐
    const int NRAM_LIMIT_SIZE = FLOOR_ALIGN(MAX_NRAM_SIZE / 2, 64);
    int nram_limit = NRAM_LIMIT_SIZE / sizeof(T);

    // 对列数据切分：计算每个任务的元素数
    int32_t num_per_core = N / taskDim;
    int32_t repeat = num_per_core / nram_limit;   // 完整分块的循环次数
    int32_t rem = num_per_core % nram_limit;       // 余数部分
    T *d_input_per_task = d_src + taskId * nram_limit;
    T *d_output_per_task = d_dst + taskId * nram_limit;
    T *nram_out = (T *)NRAM_BUFFER;                // NRAM输出缓冲区
    T *nram_in = (T *)(NRAM_BUFFER + NRAM_LIMIT_SIZE); // NRAM输入缓冲区
    const int align_rem = CEIL_ALIGN(rem, 64);     // 余数对齐

    int i = 0;
    // 处理完整的NRAM分块（repeat次循环）
    for (; i < repeat; i++) {
        // 将输入数据从GDRAM异步拷贝到NRAM
        __memcpy_async(nram_in, d_input_per_task + i * nram_limit,
                       NRAM_LIMIT_SIZE, GDRAM2NRAM);
        __sync_io();
        // 在NRAM上执行Sigmoid向量计算
        __bang_active_sigmoid(nram_out, nram_in, nram_limit);
        __sync_compute();
        // 将计算结果从NRAM异步拷贝回GDRAM
        __memcpy_async(d_output_per_task + i * nram_limit, nram_out,
                       NRAM_LIMIT_SIZE, NRAM2GDRAM);
        __sync_io();
    }
    // 处理余数部分（当输入元素数不能被NRAM分块整除时）
    if (rem > 0) {
        // 将余数数据从GDRAM拷贝到NRAM
        __memcpy_async(nram_in, d_input_per_task + i * nram_limit,
                       rem * sizeof(T), GDRAM2NRAM);
        __sync_io();
        // 在NRAM上执行Sigmoid向量计算（使用对齐后的余数大小以避免越界）
        __bang_active_sigmoid(nram_out, nram_in, align_rem);
        __sync_compute();
        // 将余数结果从NRAM拷贝回GDRAM
        __memcpy_async(d_output_per_task + i * nram_limit, nram_out,
                       rem * sizeof(T), NRAM2GDRAM);
        __sync_io();
    }
}

// 核函数入口：主机端调用，负责设置并行规模和启动核函数
template <typename T>
void bang_sigmoid_kernel_entry(cnrtQueue *queue, T *d_dst, T *d_src,
                               int elem_count) {
    cnrtDim3_t dim = {1, 1, 1};             // 默认单核并行规模
    int taskDims = dim.x * dim.y * dim.z;
    cnrtFunctionType_t c = CNRT_FUNC_TYPE_BLOCK; // Block任务类型（单核）
    if (elem_count < taskDims) {
        dim.x = 1;
        dim.y = 1;
    }
    // 启动核函数：<<<并行规模, 任务类型, 队列>>>
    bang_sigmoid_kernel<<<dim, c, queue>>>(d_dst, d_src, elem_count);
    cnrtQueueSync(queue);
}

// C++测试接口：用于纯C++环境下的功能测试（非PyTorch集成路径）
template <typename T>
void bang_sigmoid_sample(T *h_dst, T *h_src, const int elem_count) {
    T *d_src, *d_dst;
    cnrtQueue_t queue;
    cnrtQueueCreate(&queue);
    cnrtRet_t ret;
    // 在设备端分配输入和输出内存
    ret = cnrtMalloc(reinterpret_cast<void **>(&d_src),
                     elem_count * sizeof(T));
    ret = cnrtMalloc(reinterpret_cast<void **>(&d_dst),
                     elem_count * sizeof(T));
    // 将输入数据从主机端拷贝到设备端
    ret = cnrtMemcpy(d_src, h_src, elem_count * sizeof(T),
                     CNRT_MEM_TRANS_DIR_HOST2DEV);
    // 启动核函数
    bang_sigmoid_kernel_entry(queue, d_dst, d_src, elem_count);
    cnrtQueueSync(queue);
    // 将计算结果从设备端拷贝回主机端
    ret = cnrtMemcpy(h_dst, d_dst, elem_count * sizeof(T),
                     CNRT_MEM_TRANS_DIR_DEV2HOST);
    ret = cnrtQueueDestroy(queue);
}

// 模板实例化
template void bang_sigmoid_sample(float *, float *, int);
template void bang_sigmoid_kernel_entry(cnrtQueue *, float *, float *, int);
```

Sigmoid核函数的计算流程为：

1. **确定分块参数**：根据NRAM总大小计算每个分块能处理的元素数量`nram_limit`。
2. **任务切分**：利用`taskDim`和`taskId`将总数据平均分配给各任务，每个任务处理`num_per_core`个元素。
3. **整块循环处理**：对于完整的NRAM分块，循环执行"GDRAM→NRAM拷贝 → Sigmoid向量计算 → NRAM→GDRAM写回"，每次循环间通过`__sync_io()`和`__sync_compute()`进行同步。
4. **余数处理**：如果元素数不能被分块大小整除，对剩余元素单独拷贝和计算，使用`CEIL_ALIGN`对齐余数以避免向量函数越界。
5. **核函数启动**：在`bang_sigmoid_kernel_entry`入口函数中，设置并行规模`cnrtDim3_t`和任务类型`CNRT_FUNC_TYPE_BLOCK`，通过`<<<>>>`语法启动核函数。

#### 5.1.5.3 通过pybind11暴露Op接口

在BANG C算子开发与集成实验中，通过pybind11将C++函数封装为Python可调用的接口。在`bang_sigmoid.cpp`文件末尾使用`PYBIND11_MODULE`宏定义Python模块，将`active_sigmoid_mlu`函数注册到`libmlu_custom_ext`模块中：

```cpp
PYBIND11_MODULE(libmlu_custom_ext, m) {
    m.def("active_sigmoid_mlu", &active_sigmoid_mlu);
}
```

这样，在Python层就可以通过`import libmlu_custom_ext`导入模块后，直接调用`libmlu_custom_ext.active_sigmoid_mlu(x)`实现算子计算。

#### 5.1.5.4 使用setuptools编译和安装

在BANG C算子开发与集成实验中，使用`python setup.py install`将.cpp和.mlu文件通过不同的编译器编译，生成最终的动态库。主要包括以下步骤：

1. CNCC将.mlu代码编译为.o文件。
2. 将.cpp代码编译为.so文件，链接.mlu文件编译的.o。

编译完成后，在本地会生成一个动态库，一般格式为`name.cpython_version-abi.so`，例如：`libmlu_custom_ext.cpython-37m-x86_64-linux-gnu.so`。编译和安装成功后，即可在Python中导入并使用Sigmoid自定义算子。

setup.py编译脚本的实现如下：

```python
# filename: setup.py
# 所属实验：5.1 BANG C算子开发与集成实验 - 编译安装脚本
# 核心任务：使用setuptools和MLUExtension将BANG C代码和C++代码编译为Python可调用的动态库
# 对应原理：通过torch_mlu.utils.cpp_extension提供的MLUExtension，分别调用CNCC编译.mlu文件和GCC编译.cpp文件，最终链接为动态库

import os
import sys
from setuptools import setup, find_packages
from torch.utils import cpp_extension
from torch_mlu.utils.cpp_extension import MLUExtension, BuildExtension
import glob
import shutil
from setuptools.dist import Distribution

mlu_custom_src = "mlu_custom_ext"
cpath = os.path.join(
    os.path.abspath(os.path.dirname(__file__)),
    os.path.join(mlu_custom_src, "mlu"))

def source(src):
    # 收集.cpp和.mlu源文件
    cpp_src = glob.glob("{}/*.cpp".format(src))
    mlu_src = glob.glob("{}/*.mlu".format(src))
    cpp_src.extend(mlu_src)
    return cpp_src

def main():
    # 创建MLUExtension实例，指定模块名、源文件、头文件目录和编译选项
    mlu_extension = MLUExtension(
        name="libmlu_custom_ext",
        sources=source(os.path.join(cpath, 'src')),
        include_dirs=[os.path.join(cpath, "include")],
        verbose=True,
        extra_cflags=['-w'],
        extra_link_args=['-w'],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++14"],
            "cncc": ["-O3", "-I{}".format(os.path.join(cpath, "include"))]
        })

    dist = Distribution()
    dist.script_name = os.path.basename(sys.argv[0])
    dist.script_args = sys.argv[1:]
    if dist.script_args == ["clean"]:
        if os.path.exists(os.path.abspath('build')):
            shutil.rmtree('build')

    setup(name="mlu_custom_ext",
          version="0.1",
          packages=find_packages(),
          ext_modules=[mlu_extension],
          cmdclass={
              "build_ext":
              BuildExtension.with_options(no_python_abi_suffix=True)
          })

if __name__ == "__main__":
    main()
```

setup.py的关键配置说明：

1. **MLUExtension**：torch_mlu提供的扩展类，负责管理.mlu和.cpp混合编译。`name`参数指定生成的动态库名称，`sources`指定源文件列表，`include_dirs`指定头文件搜索路径。
2. **extra_compile_args**：分别指定C++编译器（cxx）和CNCC编译器（cncc）的编译选项，包括优化级别-O3和C++14标准。
3. **BuildExtension.with_options**：使用`no_python_abi_suffix=True`选项，避免Python ABI后缀导致的兼容性问题。

#### 5.1.5.5 精度对比测试

在BANG C算子开发与集成实验中，`mlu_custom_ext`安装后可以通过pip工具查看安装情况。接下来调用PyTorch框架的CPU Sigmoid算子和MLU Sigmoid自定义算子做精度对比。需要注意的是，`x_cpu.sigmoid()`调用的算子是PyTorch框架的原生算子，而不是第4.4小节实现的CPU Sigmoid自定义算子。也就是说，Sigmoid实验中实现的MLU自定义算子和第4.4节实现的CPU自定义算子都以PyTorch框架原生算子的精度作为对比的真值。

精度对比测试代码（test_sigmoid.py）如下：

```python
# filename: test_sigmoid.py
# 所属实验：5.1 BANG C算子开发与集成实验 - 精度测试
# 核心任务：对比MLU自定义Sigmoid算子与PyTorch原生Sigmoid算子的前向和反向精度
# 对应原理：使用numpy.testing.assert_array_almost_equal验证MLU实现与CPU参考值在decimal=3精度下一致

import torch
import numpy as np
import torch_mlu
import copy
from mlu_custom_ext import mlu_functions
import unittest

class TestSigmoid(unittest.TestCase):
    """
    test sigmoid
    """
    def test_forward_with_shape(self, shapes=[(3, 4)]):
        """前向传播精度测试：对比MLU Sigmoid与PyTorch原生Sigmoid的输出"""
        for shape in shapes:
            x_cpu = torch.randn(shape)
            x_mlu = x_cpu.to('mlu')
            # 调用MLU自定义Sigmoid算子
            y_mlu = mlu_functions.forward(x_mlu)
            y_cpu = x_cpu.sigmoid()  # PyTorch原生Sigmoid作为真值
            # 验证精度：decimal=3表示小数点后3位一致
            np.testing.assert_array_almost_equal(y_mlu.cpu(), y_cpu, decimal=3)

    def test_backward_with_shape(self, shapes=[(3, 4)]):
        """反向传播精度测试：对比MLU Sigmoid的梯度与理论梯度值"""
        for shape in shapes:
            x_mlu = torch.randn(shape, requires_grad=True, device='mlu')
            # 前向计算
            y_mlu = mlu_functions.forward(x_mlu)
            z_mlu = torch.sum(y_mlu)
            z_mlu.backward()  # 反向传播计算梯度
            grad_mlu = x_mlu.grad
            with torch.no_grad():
                # Sigmoid梯度理论值：y * (1 - y)
                grad_cpu = (y_mlu * (1 - y_mlu)).cpu()
            np.testing.assert_array_almost_equal(
                grad_mlu.detach().cpu(), grad_cpu, decimal=3)

if __name__ == '__main__':
    unittest.main()
```

#### 5.1.5.6 性能测试

在BANG C算子开发与集成实验中，精度测试通过后，调用`torch.mlu.Event`接口来做性能测试。性能测试时需要注意的是，由于测试代码调用PyTorch框架和自定义算子时会触发底层运行时对设备的初始化和核函数二进制模型的加载等动作，所以为了更准确地使用`torch.mlu.Event`对算子计时，需要在正式计时前预先执行一遍相同的测试，称为**预热（Warmup）**。

性能测试代码（test_sigmoid_benchmark.py）如下：

```python
# filename: test_sigmoid_benchmark.py
# 所属实验：5.1 BANG C算子开发与集成实验 - 性能测试
# 核心任务：使用torch.mlu.Event测量MLU自定义Sigmoid算子的前向和反向耗时
# 对应原理：通过预热消除首次调用的初始化开销，使用Event的hardware_time接口获取精确的硬件执行时间（毫秒级）

import torch
import numpy as np
import torch_mlu
import copy
from mlu_custom_ext import mlu_functions
import unittest

class TestSigmoidBenchmark(unittest.TestCase):
    """
    test sigmoid benchmark
    """
    def test_forward_with_shape(self, shapes=[(3, 4)]):
        # 预热：提前执行一次相同计算，消除设备初始化和核函数加载开销
        for shape in shapes:
            x_mlu_warmup = torch.randn(shape).to('mlu')
            _ = mlu_functions.forward(x_mlu_warmup)

        for shape in shapes:
            event_start = torch.mlu.Event()
            event_end = torch.mlu.Event()
            event_start.record()
            x_cpu = torch.randn(shape)
            x_mlu = x_cpu.to('mlu')
            y_mlu = mlu_functions.forward(x_mlu)
            y_cpu = x_cpu.sigmoid()
            np.testing.assert_array_almost_equal(y_mlu.cpu(), y_cpu, decimal=3)
            event_end.record()
            event_end.synchronize()
            print('forward time: ', event_start.hardware_time(event_end), 'ms')

    def test_backward_with_shape(self, shapes=[(3, 4)]):
        # 预热：提前执行一次反向传播计算
        for shape in shapes:
            x_mlu_warmup = torch.randn(shape, requires_grad=True, device='mlu')
            y_warmup = mlu_functions.forward(x_mlu_warmup)
            torch.sum(y_warmup).backward()

        for shape in shapes:
            event_start = torch.mlu.Event()
            event_end = torch.mlu.Event()
            event_start.record()
            x_mlu = torch.randn(shape, requires_grad=True, device='mlu')
            y_mlu = mlu_functions.forward(x_mlu)
            z_mlu = torch.sum(y_mlu)
            z_mlu.backward()
            grad_mlu = x_mlu.grad
            with torch.no_grad():
                grad_cpu = (y_mlu * (1 - y_mlu)).cpu()
            np.testing.assert_array_almost_equal(
                grad_mlu.detach().cpu(), grad_cpu, decimal=3)
            event_end.record()
            event_end.synchronize()
            print('backward time: ', event_start.hardware_time(event_end), 'ms')

if __name__ == '__main__':
    unittest.main()
```

性能测试的关键要点：

1. **预热机制**：在正式计时前，使用相同shape的数据执行一次完整的前向（和反向）计算，确保MLU设备已完成初始化和核函数加载。
2. **Event计时**：通过`torch.mlu.Event()`创建开始和结束事件，在计算前后分别调用`record()`记录时间点。
3. **同步等待**：调用`event_end.synchronize()`确保异步计算完成后再读取时间。
4. **硬件时间**：使用`event_start.hardware_time(event_end)`获取纯硬件执行时间，单位为毫秒。

### 5.1.6 推荐实践内容

在BANG C算子开发与集成实验中，主要关注BANG C自定义算子的实现与验证、与框架的集成以及完整的模型推断。模型推断的性能和精度应同时作为主要参考指标。推荐实践目标如下：

- **基础实践**：实现Sigmoid主程序、核函数、pybind接口，使用setuptools编译并安装成功。
- **进阶实践**：在基础实践基础上，完成精度对比测试用例，使用numpy的`assert_array_almost_equal`接口评估精度误差在decimal=3以内。
- **高级实践**：在进阶实践基础上，完成性能测试，前向Sigmoid测试耗时小于25ms，反向Sigmoid测试耗时小于70ms。

### 5.1.7 实验思考

1. 在BANG C算子开发与集成实验中实现的MLU Sigmoid自定义算子与第4.4小节实现的CPU Sigmoid自定义算子精度是否一致？
2. 使用BANG C如何实现一个和CPU精度一致的MLU Sigmoid算子？
3. CPU和MLU算子精度不一致（以Sigmoid算子为例）对神经网络的推理精度有何影响？

---

## 5.2 智能编程语言性能优化实验

### 5.2.1 实验目的

掌握使用智能编程语言优化算法性能的原理，掌握智能编程语言的调试和调优方法，能够使用智能编程语言在MLU上加速矩阵乘的计算。实验工作量：代码量约700行，实验时间约6小时。

### 5.2.2 背景介绍

#### 5.2.2.1 智能编程模型

智能计算系统的层次化抽象将多卡的DLP服务器抽象为五个层次，即服务器级（Server）、板卡级（Card）、芯片级（Chip）、核心簇级（Cluster）和核心级（Core）：

1. **服务器级（Server）**：整个服务器系统包含若干CPU构成的控制单元，以及片外存储器构成的存储单元（主机端内存），由PCIe总线互连的若干DLP板卡作为该层的计算单元。
2. **板卡级（Card）**：每块DLP板卡上包含片外存储器，板卡上可以有多个DLP芯片通过芯粒（Chiplet）封装，多个DLP芯片共享片外存储器，每个DLP芯片作为计算和控制单元。
3. **芯片级（Chip）**：每个芯片包含多个核心簇作为计算单元，核心簇间共享高速缓存。
4. **核心簇级（Cluster）**：核心簇内封装了单个或多个DLP核心作为控制和计算单元，核心簇内有核心簇级的存储器做片上的多核数据通信，相比片外存储器可以极大降低访存延迟。
5. **核心级（Core）**：每个DLP核心包含功能单元、寄存器、以及神经元存储器和权重存储器等片上高速存储器。

该架构可以很方便地通过增加板卡、芯片、核心簇或者核心等方式提升整个系统的计算能力。

从服务器级依次到处理器核级，存储单元的数据访问延迟依次递减，数据访问带宽依次递增，存储单元的空间大小依次递减。在编程实现时，如果需要将数据从GDRAM拷贝到SRAM，只需调用智能编程语言中的Memcpy函数，同时指定拷贝方向为GDRAM2SRAM。

在矩阵乘性能优化实验的编程实践中，可以在程序中指定运行一次任务调用的计算资源数量。特别地，称一次执行只调用一个Core的任务为**BLOCK任务**。一次执行只调用一个Cluster的任务为**UNION1任务**，对应调用两个Cluster与四个Cluster的任务分别为**UNION2**和**UNION4**。关于智能编程模型更详细的介绍，请参考《智能计算系统》第8章。

#### 5.2.2.2 MLU并行编程

智能编程语言提供了与Kernel函数内部任务切分相关的内置变量，方便开发者有效利用MLU资源。

**Core变量**：
- `coreDim`（核维数）：表示一个Cluster包含的Core个数，例如MLU370上等于4。
- `coreId`（核序号）：表示每个Core在Cluster内的逻辑ID，例如MLU370上的取值范围为[0-3]。

**Cluster变量**：
- `clusterDim`（簇维数）：表示启动Kernel时指定的UNION类型任务调用的Cluster个数，例如UNION4时等于4。
- `clusterId`（簇序号）：表示clusterDim内某个Cluster的逻辑ID，例如UNION4时其取值范围是[0-3]。

**Task变量**：
- `taskDimX`/`taskDimY`/`taskDimZ`：分别表示1个任务在X/Y/Z方向的任务规模，其值等于主机端所指定的任务规模。
- `taskDim`（任务维数）：表示用户指定任务的总规模，$taskDim = taskDimX \times taskDimY \times taskDimZ$。
- `taskIdX`/`taskIdY`/`taskIdZ`：分别表示程序运行时所分配的逻辑规模在X/Y/Z方向的任务ID。
- `taskId`（任务序号）：表示程序运行时所分配的任务ID，其值为对逻辑规模降维后的任务ID，$taskId = taskIdZ \times taskDimY \times taskDimX + taskIdY \times taskDimX + taskIdX$。

以下是一个实际的内置变量取值示例。当程序调用8个计算核（UNION2类型，即2个Cluster×4个Core=8核）时，`{taskDimX, taskDimY, taskDimZ}`设为`{8, 1, 1}`，每个核上的并行变量取值如下：

| taskId | taskIdX | taskIdY | taskIdZ | clusterDim | coreDim | coreId | clusterId | taskDimX | taskDimY | taskDimZ | taskDim |
|--------|---------|---------|---------|------------|---------|--------|-----------|----------|----------|----------|---------|
| 0 | 0 | 0 | 0 | 2 | 4 | 0 | 0 | 8 | 1 | 1 | 8 |
| 1 | 1 | 0 | 0 | 2 | 4 | 1 | 0 | 8 | 1 | 1 | 8 |
| 2 | 2 | 0 | 0 | 2 | 4 | 2 | 0 | 8 | 1 | 1 | 8 |
| 3 | 3 | 0 | 0 | 2 | 4 | 3 | 0 | 8 | 1 | 1 | 8 |
| 4 | 4 | 0 | 0 | 2 | 4 | 0 | 1 | 8 | 1 | 1 | 8 |
| 5 | 5 | 0 | 0 | 2 | 4 | 1 | 1 | 8 | 1 | 1 | 8 |
| 6 | 6 | 0 | 0 | 2 | 4 | 2 | 1 | 8 | 1 | 1 | 8 |
| 7 | 7 | 0 | 0 | 2 | 4 | 3 | 1 | 8 | 1 | 1 | 8 |

从表中可以看出：`clusterDim=2`表示2个Cluster，每个Cluster有`coreDim=4`个Core，总共8个计算核。前4个task（taskId 0-3）运行在clusterId=0的Cluster上，coreId依次为0-3；后4个task（taskId 4-7）运行在clusterId=1的Cluster上，coreId同样依次为0-3。

#### 5.2.2.3 Notifier接口

在矩阵乘性能优化实验中，不涉及深度学习框架集成等系统开发内容，侧重使用智能编程语言进行程序优化。主机端CPU代码使用`gettimeofday`来统计矩阵乘运算函数的耗时，设备端MLU的硬件耗时可以使用**Notifier（通知）接口**。

Notifier是一种轻量级任务，不像计算任务那样占用计算资源，而是通过驱动从硬件读取一些运行参数，只占用很少的执行时间（几乎可以忽略不计）。Notifier可以像计算任务一样放入Queue（队列）中执行，在队列中均遵循FIFO（先进先出）调度原则。可以使用Notifier来统计Kernel计算任务的硬件执行时间。`cnrtNotifierElapsedTime`接口返回的耗时单位是毫秒。

使用Notifier机制统计Kernel执行时间的代码示例如下：

```c
// Notifier机制：在核函数前后放置Notifier，统计MLU硬件执行时间
cnrtNotifier_t start, end;
CNRT_CHECK(cnrtNotifierCreate(&start));
CNRT_CHECK(cnrtNotifierCreate(&end));
CNRT_CHECK(cnrtPlaceNotifier(start, queue));
Kernel<<<...>>>(...);              // 待测核函数
CNRT_CHECK(cnrtPlaceNotifier(end, queue));
CNRT_CHECK(cnrtSyncQueue(queue));
CNRT_CHECK(cnrtNotifierElapsedTime(start, end, &mlu_time_used));
printf("MLU Time taken: %.3f ms\n", mlu_time_used);
```

### 5.2.3 实验环境

- 硬件平台：MLU云平台环境。
- 软件环境：CNToolkit开发工具包。

### 5.2.4 实验内容

在矩阵乘性能优化实验中，对于矩阵乘运算，首先使用BANG C语言实现一个标量版本，然后利用NRAM存储优化、向量化、多核并行优化、三级流水优化、五级流水优化，逐步实现一个高性能矩阵乘。每一个优化步骤除了和CPU版本做性能对比外，还和上一步的BANG C优化做对比，从而逐步理解智能编程语言的优化技巧：

1. 实现一个CPU版本的标量矩阵乘，将标量矩阵乘实现迁移到MLU上，做性能对比。
2. 利用BANG C的NRAM地址空间加速标量版本的矩阵乘，对比性能提升。
3. 利用BANG C提供的向量化接口`__bang_matmul`加速矩阵乘，对比性能提升。
4. 利用BANG C提供的Block任务，使用MLU的多核做并行加速，对比性能提升。
5. 利用BANG C提供的异步拷贝接口`__memcpy_async`做三级流水优化，对比性能提升。
6. 利用BANG C提供的Union任务和SRAM地址空间，做五级流水优化，对比性能提升。

### 5.2.5 实验步骤

#### 5.2.5.1 矩阵乘标量实现

在矩阵乘性能优化实验中，首先实现主机端程序负责为矩阵乘生成随机初始值，然后在主机端实现一个CPU版本的标量矩阵乘作为性能基准。

MLU版本的标量矩阵乘实现和CPU一致。由于核函数是执行在MLU上的，所以核函数需要在声明时加入特殊的`__mlu_entry__`属性，并在主机端使用`<<<>>>`核函数调用语法。本实验步骤MLU标量实现和CPU一样只使用一个计算核心，所以在`<<<>>>`核函数调用时需要指定核函数任务类型`cnrtFunctionType_t`和核函数并行规模`cnrtDim3_t`。对于CPU使用`gettimeofday`函数计时，精度为微秒级；对于MLU使用`cnrtNotifierElapsedTime`函数接口统计核函数的硬件耗时，精度同样为微秒级。

选用较小规模的矩阵乘（M×N×K=128×256×128）的原因是标量单核版本的实现在MLU上执行过慢会触发运行时的超时检查，后续实现步骤中随着优化加速，会选用较大规模的矩阵乘和CPU版本做性能对比。

在矩阵乘性能优化实验的标量实现阶段，CPU和MLU的乘累加顺序一致，所以计算精度误差为0。在后续的向量化优化阶段中，由于使用了BANG C提供的向量化运算接口，乘累加的顺序和基准版本的CPU实现不一致，所以会导致校验精度时误差不为0，因此测试代码中应采用相对误差来评估精度。

矩阵乘标量实现的主机端完整代码如下：

```cpp
// 矩阵乘主机端程序：包括CPU基线实现、MLU调用、Notifier计时和精度校验
int main() {
    int m = M, n = N, k = K;
    printf("\nM = %d, N = %d, K = %d\n", m, n, k);

    // 主机端内存分配
    float *left = (float *)malloc(m * n * sizeof(float));
    float *right = (float *)malloc(n * k * sizeof(float));
    float *result = (float *)malloc(m * k * sizeof(float));

    // MLU设备端内存分配
    float *left_mlu = NULL, *right_mlu = NULL, *result_mlu = NULL;
    CNRT_CHECK(cnrtMalloc((void **)&left_mlu, m * n * sizeof(float)));
    CNRT_CHECK(cnrtMalloc((void **)&right_mlu, n * k * sizeof(float)));
    CNRT_CHECK(cnrtMalloc((void **)&result_mlu, m * k * sizeof(float)));

    // 初始化左矩阵随机值
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            left[i * n + j] = generateRandomFloat(1.0f, 1.1f);
        }
    }
    // 初始化右矩阵随机值
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            right[i * k + j] = generateRandomFloat(1.0f, 1.1f);
        }
    }

    // 将左右矩阵从主机端拷贝到MLU设备端
    CNRT_CHECK(cnrtMemcpy(left_mlu, left, m * n * sizeof(float),
                          cnrtMemcpyHostToDev));
    CNRT_CHECK(cnrtMemcpy(right_mlu, right, n * k * sizeof(float),
                          cnrtMemcpyHostToDev));

    // CPU版本计时和计算
    struct timeval st_cpu, et_cpu;
    gettimeofday(&st_cpu, NULL);
    multiplyMatricesCPU(left, right, result, m, n, k);
    gettimeofday(&et_cpu, NULL);
    float cpu_time_used = (et_cpu.tv_sec - st_cpu.tv_sec) * 1e3
                        + (et_cpu.tv_usec - st_cpu.tv_usec) / 1e3;
    printf("\nCPU Time taken: %.3f ms\n", cpu_time_used);

    // MLU版本：创建Notifier用于计时
    cnrtNotifier_t st_mlu, et_mlu;
    CNRT_CHECK(cnrtNotifierCreate(&st_mlu));
    CNRT_CHECK(cnrtNotifierCreate(&et_mlu));
    cnrtQueue_t queue;
    CNRT_CHECK(cnrtQueueCreate(&queue));
    cnrtDim3_t dim = {1, 1, 1};
    cnrtFunctionType_t func_type = CNRT_FUNC_TYPE_BLOCK;

    // 放置开始Notifier并启动核函数
    CNRT_CHECK(cnrtPlaceNotifier(st_mlu, queue));
    multiplyMatricesMLU<<<dim, func_type, queue>>>(
        left_mlu, right_mlu, result_mlu, m, n, k);
    // 放置结束Notifier
    CNRT_CHECK(cnrtPlaceNotifier(et_mlu, queue));
    CNRT_CHECK(cnrtQueueSync(queue));

    // 计算MLU硬件耗时（毫秒）
    float mlu_time_used = 0.0f;
    CNRT_CHECK(cnrtNotifierElapsedTime(st_mlu, et_mlu, &mlu_time_used));
    printf("\nMLU Time taken: %.3f ms\n", mlu_time_used);

    // 将MLU计算结果拷回主机端
    float *result_actual = (float *)malloc(m * k * sizeof(float));
    CNRT_CHECK(cnrtMemcpy(result_actual, result_mlu, m * k * sizeof(float),
                          cnrtMemcpyDevToHost));

    // 精度校验：使用相对误差判断CPU和MLU结果是否一致
    bool is_passed = true;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            float diff_rel = relativeError(result[i * k + j],
                                           result_actual[i * k + j]);
            if (diff_rel > 0.1) {
                printf("diff_rel = %.3f\n", diff_rel);
                printf("[%d, %d]: cpu = %f, mlu = %f\n",
                       i, j, result[i * k + j], result_actual[i * k + j]);
                is_passed = false;
            }
        }
    }
    printf(is_passed ? "\nPASSED\n" : "\nFAILED\n");

    // 资源释放
    free(left);  CNRT_CHECK(cnrtFree(left_mlu));
    free(right); CNRT_CHECK(cnrtFree(right_mlu));
    free(result); free(result_actual);
    CNRT_CHECK(cnrtFree(result_mlu));
    CNRT_CHECK(cnrtNotifierDestroy(st_mlu));
    CNRT_CHECK(cnrtNotifierDestroy(et_mlu));
    CNRT_CHECK(cnrtQueueDestroy(queue));
    return 0;
}
```

#### 5.2.5.2 矩阵乘标量NRAM实现

在矩阵乘性能优化实验的标量NRAM优化阶段，目的是利用BANG C的NRAM地址空间来加速标量版本的矩阵乘。原理是通过BANG C的静态数组在NRAM上为左右矩阵和结果矩阵声明空间，将小规模的矩阵一次性拷贝至NRAM，在NRAM上运算完成后一次性拷出到GDRAM空间。由于标量读写NRAM空间的性能远高于GDRAM空间，所以性能会有几十倍的提升。

标量NRAM实现的核函数代码如下：

```c
// filename: 02_scalar_nram.mlu
// 所属实验：5.2 矩阵乘性能优化 - 标量NRAM实现
// 核心任务：将矩阵数据从GDRAM拷贝到NRAM，在NRAM上执行标量矩阵乘，利用NRAM高带宽加速计算
// 对应原理：NRAM是片上高速存储器，读写延迟远低于GDRAM，将小规模矩阵整体放入NRAM可大幅提升访存性能

#define M 128
#define N 256
#define K 128

__mlu_entry__ void multiplyMatricesMLU(float *left, float *right,
                                        float *result, int m, int n, int k) {
    // 在NRAM上声明静态数组存放左右矩阵和结果矩阵
    __nram__ float left_nram[M * N];
    __nram__ float right_nram[N * K];
    __nram__ float result_nram[M * K];

    // 将左矩阵从GDRAM拷贝至NRAM
    __memcpy(left_nram, left, M * N * sizeof(float), GDRAM2NRAM);
    // 将右矩阵从GDRAM拷贝至NRAM
    __memcpy(right_nram, right, N * K * sizeof(float), GDRAM2NRAM);

    // 在NRAM上执行标量矩阵乘计算
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            float sum = 0.0f;
            for (int p = 0; p < N; p++) {
                sum += left_nram[i * N + p] * right_nram[p * K + j];
            }
            result_nram[i * K + j] = sum;
        }
    }

    // 将结果矩阵从NRAM写回GDRAM
    __memcpy(result, result_nram, M * K * sizeof(float), NRAM2GDRAM);
}
```

#### 5.2.5.3 矩阵乘向量NRAM实现

在矩阵乘性能优化实验的向量NRAM优化阶段，MLU硬件架构支持SIMD指令集，所以为了发挥算力优势，必须调用BANG C封装的Builtin函数`__bang_matmul`进行向量化加速。向量NRAM优化阶段仍然用较小规模的单核矩阵乘，仅仅将标量循环实现替换为调用`__bang_matmul`函数实现，性能将有几千倍量级的提升。

向量NRAM实现的核函数代码如下：

```c
// filename: 03_vector_nram.mlu
// 所属实验：5.2 矩阵乘性能优化 - 向量NRAM实现
// 核心任务：使用BANG C内置向量函数__bang_matmul替代标量三重循环，充分利用MLU的SIMD指令集
// 对应原理：__bang_matmul是BANG C封装的矩阵乘向量化接口，底层利用MLU的向量运算单元并行计算

#define M 128
#define N 256
#define K 128

__mlu_entry__ void multiplyMatricesMLU(float *left, float *right,
                                        float *result, int m, int n, int k) {
    // NRAM存放左矩阵，WRAM存放右矩阵（利用不同存储体提高并行度）
    __nram__ float left_nram[M * N];
    __wram__ float right_wram[N * K];
    __nram__ float result_nram[M * K];

    // 将左矩阵从GDRAM拷贝至NRAM
    __memcpy(left_nram, left, M * N * sizeof(float), GDRAM2NRAM);
    // 将右矩阵从GDRAM拷贝至WRAM
    __memcpy(right_wram, right, N * K * sizeof(float), GDRAM2WRAM);

    // 调用BANG C向量化矩阵乘接口，替代标量三重循环
    __bang_matmul(result_nram, left_nram, right_wram, m, n, k);

    // 将结果矩阵从NRAM写回GDRAM
    __memcpy(result, result_nram, M * K * sizeof(float), NRAM2GDRAM);
}
```

#### 5.2.5.4 矩阵乘多核向量NRAM实现

在矩阵乘性能优化实验中，对于多核向量实现，128×256×128的矩阵乘规模过小（上一个实验步骤中MLU的耗时已经降到了50微秒量级，计时函数的精度为微秒量级），所以将M扩大到`M_PER_BLOCK × BLOCKS = 524288`，其中`M_PER_BLOCK`表示每个任务处理的M维度的元素个数，`BLOCKS`表示`<<<>>>`核函数的并行任务数。

多核加速的原理是将超大规模的矩阵乘在M维度上拆分为BLOCKS个可并行的小矩阵乘，然后利用BANG C提供的`taskId`、`taskDim`等并行变量将核函数改写为并行版本，改写后相比CPU的标量单核版本基线要快几千至上万倍。由于片上NRAM和WRAM空间容量有限，所以对于M=524288这样大规模的矩阵乘，需要首先做多核拆分$M = M\_PER\_BLOCK \times BLOCKS$，其次还可以在单核内做循环拆分$M\_PER\_BLOCK = M\_PER\_LOOP \times LOOPS$。

多核拆分的BLOCKS不是越多越好，因为BLOCKS个并行任务最终要映射至有限个MLU计算核心，每启动一次MLU核心都会有一定的硬件和软件开销，软件开销与核函数的实现有关（例如核函数中循环体外部代码）。如果启用的BLOCKS过多，每个MLU核心都要执行一遍重复的运算，所以最优的BLOCKS数量应该等于MLU芯片的物理核心数，然后将并行运算代码放在循环体内。

多核向量NRAM实现的核函数代码如下：

```c
// filename: 04_vector_nram_blocks.mlu
// 所属实验：5.2 矩阵乘性能优化 - 多核向量NRAM实现
// 核心任务：利用BANG C的taskId/taskDim并行变量，将大矩阵在M维度拆分到多个计算核并行执行
// 对应原理：M_PER_BLOCK × BLOCKS拆分策略，每个核处理M_PER_LOOP × LOOPS的子矩阵，最优BLOCKS=物理核心数

#define LOOPS 32
#define BLOCKS 128
#define M_PER_LOOP 128
#define M_PER_BLOCK (M_PER_LOOP * LOOPS)
#define M (M_PER_BLOCK * BLOCKS)
#define N 256
#define K 128

__mlu_entry__ void multiplyMatricesMLU(float *left, float *right,
                                        float *result, int m, int n, int k) {
    // NRAM数组：存放当前任务处理的左矩阵子块和结果矩阵子块
    __nram__ float left_nram[M_PER_LOOP * N];
    __wram__ float right_wram[N * K];
    __nram__ float result_nram[M_PER_LOOP * K];

    int m_per_block = m / taskDim;          // 每个任务处理的M维度元素数
    int m_per_loop = m_per_block / LOOPS;   // 每轮循环处理的M维度元素数

    // 将右矩阵从GDRAM拷贝至WRAM（所有核共享同一右矩阵）
    __memcpy(right_wram, right, N * K * sizeof(float), GDRAM2WRAM);

    // 循环处理：每轮读取一块左矩阵子块，计算一块结果子块
    for (int loop = 0; loop < LOOPS; loop++) {
        // 将当前左矩阵子块从GDRAM拷贝至NRAM
        __memcpy(left_nram,
                 left + taskId * m_per_block * n + loop * m_per_loop * n,
                 m_per_loop * n * sizeof(float), GDRAM2NRAM);
        // 向量化矩阵乘计算
        __bang_matmul(result_nram, left_nram, right_wram, m_per_loop, n, k);
        // 将结果子块从NRAM写回GDRAM
        __memcpy(result + taskId * m_per_block * k + loop * m_per_loop * k,
                 result_nram, m_per_loop * k * sizeof(float), NRAM2GDRAM);
    }
}
```

主机端多核调用代码的关键配置如下：

```cpp
// 多核并行配置：设置BLOCKS个并行任务，每个任务处理一个M维度子块
cnrtDim3_t dim = {BLOCKS, 1, 1};  // BLOCKS=128个并行任务
cnrtFunctionType_t func_type = CNRT_FUNC_TYPE_BLOCK;

// Notifier计时
CNRT_CHECK(cnrtPlaceNotifier(st_mlu, queue));
multiplyMatricesMLU<<<dim, func_type, queue>>>(
    left_mlu, right_mlu, result_mlu, m, n, k);
CNRT_CHECK(cnrtPlaceNotifier(et_mlu, queue));
CNRT_CHECK(cnrtQueueSync(queue));
CNRT_CHECK(cnrtNotifierElapsedTime(st_mlu, et_mlu, &mlu_time_used));
```

#### 5.2.5.5 矩阵乘多核向量NRAM三级流水实现

在矩阵乘性能优化实验中，MLU架构的NRAM、WRAM、GDRAM等各级存储之间支持异步的DMA数据拷贝，而且访存单元和运算单元之间是可以并行执行的，这就为软件流水优化提供了可能。BANG C提供了异步拷贝接口`__memcpy_async`和同步接口`__sync`，通过三级流水实现可以进一步缓解访存瓶颈，即缓解"访存墙"带来的喂不饱MLU运算单元的情况。

简单来说，软件流水就是对循环中的操作进行调整，使尽可能多的操作可以并行执行。在向量乘法案例中，整体逻辑可以分为三部分：Load（数据加载）、Compute（计算）和Store（结果写回）。为了避免Load和Store产生冲突，需要引入两块独立的Ping-Pong缓冲区。由于NRAM大小有限，要进行多次普通的向量计算过程：顺序执行[L0→C0→S0]→[L1→C1→S1]→[L2→C2→S2]。如果通过软流水技术重新排列，则变成[L0]→[S0+C1+L2]→[S1+C2+L3]→[S2+C3+L4]的形式，可以写成新的循环形式：同时进行上一轮的结果写回、当前轮的计算和下一轮的数据加载。

多核向量NRAM三级流水实现的核函数代码如下：

```c
// filename: 05_vector_nram_blocks_pipe3.mlu
// 所属实验：5.2 矩阵乘性能优化 - 三级流水实现
// 核心任务：利用Ping-Pong缓冲区和异步拷贝，将Load/Compute/Store三级操作流水线化
// 对应原理：__memcpy_async异步拷贝不阻塞计算单元，__sync同步点确保数据依赖正确；
//          STAGES=2提供双缓冲，使当前轮计算、上轮写回、下轮加载可同时进行

#define STAGES 2    // Ping-Pong双缓冲
#define LOOPS 32
#define BLOCKS 128
#define M_PER_LOOP 128
#define M_PER_BLOCK (M_PER_LOOP * LOOPS)
#define M (M_PER_BLOCK * BLOCKS)
#define N 256
#define K 128

__mlu_entry__ void multiplyMatricesMLU(float *left, float *right,
                                        float *result, int m, int n, int k) {
    // STAGES=2：使用双缓冲存放左右矩阵和结果矩阵的子块
    __nram__ float left_nram[STAGES * M_PER_LOOP * N];
    __wram__ float right_wram[N * K];
    __nram__ float result_nram[STAGES * M_PER_LOOP * K];

    int m_per_block = m / taskDim;
    int m_per_loop = m_per_block / LOOPS;

    // 将右矩阵从GDRAM异步拷贝至WRAM
    __memcpy_async(right_wram, right, N * K * sizeof(float), GDRAM2WRAM);

    // 三级流水主循环：每轮迭代同时执行上一轮Store、当前轮Compute、下一轮Load
    for (int loop = 0; loop < (LOOPS + STAGES); loop++) {
        // Stage 1 (Load): 异步加载下一块左矩阵数据
        if (loop < LOOPS) {
            __memcpy_async(
                left_nram + (loop % STAGES) * m_per_loop * n,
                left + taskId * m_per_block * n + loop * m_per_loop * n,
                m_per_loop * n * sizeof(float), GDRAM2NRAM);
        }
        // Stage 2 (Compute): 对上一轮已加载的数据执行矩阵乘计算
        if (loop >= 1 && loop <= LOOPS) {
            __bang_matmul(
                result_nram + m_per_loop * k * ((loop - 1) % STAGES),
                left_nram + m_per_loop * n * ((loop - 1) % STAGES),
                right_wram, m_per_loop, n, k);
        }
        // Stage 3 (Store): 将再上一轮的计算结果异步写回GDRAM
        if (loop >= STAGES) {
            __memcpy_async(
                result + taskId * m_per_block * k
                       + (loop - STAGES) * m_per_loop * k,
                result_nram + m_per_loop * k * ((loop - STAGES) % STAGES),
                m_per_loop * k * sizeof(float), NRAM2GDRAM);
        }
        // 同步：确保本轮所有异步操作完成后再进入下一轮
        __sync();
    }
}
```

三级流水的执行逻辑说明：

1. **第0轮（loop=0）**：仅执行Load操作，加载第0块左矩阵（Ping缓冲区）。
2. **第1轮（loop=1）**：同时执行Load第1块（Pong缓冲区）、Compute第0块（Ping数据）、无Store（还需等数据就绪）。
3. **第2轮到第LOOPS轮**：同时执行Load（loop%STAGES缓冲区）、Compute（(loop-1)%STAGES缓冲区）、Store（(loop-STAGES)%STAGES缓冲区）。
4. **第LOOPS+1到LOOPS+STAGES-1轮**：无新Load，仅完成最后剩余的Compute和Store。

#### 5.2.5.6 矩阵乘多核向量SRAM五级流水实现

在矩阵乘性能优化实验中，五级流水相比三级流水多出一级SRAM地址空间。SRAM是核心簇级共享存储器，可以在Cluster内的多个Core之间共享数据。五级流水在三级流水（Load→Compute→Store）的基础上增加了SRAM层级的数据传输，数据流动路径变为：GDRAM→SRAM→NRAM→Compute→NRAM→SRAM→GDRAM，形成五级流水线。

多核向量SRAM五级流水实现的核函数代码如下：

```c
// filename: 06_vector_sram_unions_pipe5.mlu
// 所属实验：5.2 矩阵乘性能优化 - 五级流水实现
// 核心任务：利用SRAM簇内共享存储和Union任务类型，实现GDRAM→SRAM→NRAM→Compute→NRAM→SRAM→GDRAM五级流水
// 对应原理：SRAM是Cluster级存储，延迟介于GDRAM和NRAM之间，通过Union任务类型可使用多个Core协作

#define STAGES 2
#define LOOPS 32
#define BLOCKS 128
#define M_PER_LOOP 64
#define M_PER_LOOP_BLOCK (M_PER_LOOP * STAGES)
#define M_PER_LOOP_UNION (M_PER_LOOP_BLOCK * 4)  // coreDim=4
#define M_PER_BLOCK (M_PER_LOOP_BLOCK * LOOPS)
#define M (M_PER_BLOCK * BLOCKS)
#define N 256
#define K 128

// Cluster内部核函数：负责NRAM级别的三级流水（SRAM→NRAM→Compute→NRAM→SRAM）
__mlu_func__ void multiplyMatrices(float *left, float *right_wram,
                                    float *result, int m, int n, int k) {
    __nram__ float left_nram[STAGES * M_PER_LOOP * N];
    __nram__ float result_nram[STAGES * M_PER_LOOP * K];

    int m_per_block = m / coreDim;       // 每个Core处理的数据量
    int m_per_loop = m_per_block / STAGES;
    int loops = STAGES;

    for (int loop = 0; loop < (loops + STAGES); loop++) {
        if (loop < loops) {
            __sync_io();
            // 将左矩阵子块从SRAM异步拷贝至NRAM
            __memcpy_async(
                left_nram + (loop % STAGES) * m_per_loop * n,
                left + coreId * m_per_block * n + loop * m_per_loop * n,
                m_per_loop * n * sizeof(float), SRAM2NRAM);
        }
        if (loop >= 1 && loop <= loops) {
            // 在NRAM上执行向量化矩阵乘
            __bang_matmul(
                result_nram + m_per_loop * k * ((loop - 1) % STAGES),
                left_nram + m_per_loop * n * ((loop - 1) % STAGES),
                right_wram, m_per_loop, n, k);
        }
        if (loop >= STAGES) {
            // 将结果从NRAM写回SRAM
            __memcpy_async(
                result + coreId * m_per_block * k
                       + (loop - STAGES) * m_per_loop * k,
                result_nram + m_per_loop * k * ((loop - STAGES) % STAGES),
                m_per_loop * k * sizeof(float), NRAM2SRAM);
        }
        __sync_move();
    }
}

// 主核函数：负责GDRAM↔SRAM级别的数据搬运
__mlu_entry__ void multiplyMatricesMLU(float *left, float *right,
                                        float *result, int m, int n, int k) {
    // SRAM数组（__mlu_shared__）：Cluster内所有Core共享
    __mlu_shared__ float left_sram[STAGES * M_PER_LOOP_UNION * N];
    __mlu_shared__ float right_sram[N * K];
    __wram__ float right_wram[N * K];
    __mlu_shared__ float result_sram[STAGES * M_PER_LOOP_UNION * K];

    int m_per_block = m / taskDim;
    int m_per_union = m_per_block * coreDim;          // Cluster内所有Core处理的总量
    int m_per_loop_union = m_per_union / LOOPS;       // 每轮循环Cluster处理量

    // 将右矩阵从GDRAM→SRAM→WRAM
    __memcpy_async(right_sram, right, N * K * sizeof(float), GDRAM2SRAM);
    __sync_cluster();
    __memcpy(right_wram, right_sram, N * K * sizeof(float), SRAM2WRAM);

    // 五级流水主循环
    for (int loop = 0; loop < (LOOPS + STAGES); loop++) {
        if (loop < LOOPS) {
            // Stage 1: GDRAM→SRAM 加载左矩阵子块
            __memcpy_async(
                left_sram + (loop % STAGES) * m_per_loop_union * n,
                left + taskId * m_per_block * n + loop * m_per_loop_union * n,
                m_per_loop_union * n * sizeof(float), GDRAM2SRAM);
        }
        if (loop >= 1 && loop <= LOOPS) {
            // Stage 2-4: 调用Cluster内部核函数（SRAM→NRAM→Compute→NRAM→SRAM）
            multiplyMatrices(
                left_sram + m_per_loop_union * n * ((loop - 1) % STAGES),
                right_wram,
                result_sram + m_per_loop_union * k * ((loop - 1) % STAGES),
                m_per_loop_union, n, k);
        }
        if (loop >= STAGES) {
            // Stage 5: SRAM→GDRAM 写回结果
            __memcpy_async(
                result + taskId * m_per_block * k
                       + (loop - STAGES) * m_per_loop_union * k,
                result_sram + m_per_loop_union * k * ((loop - STAGES) % STAGES),
                m_per_loop_union * k * sizeof(float), SRAM2GDRAM);
        }
        __sync_cluster();  // Cluster级同步
    }
}
```

主机端Union任务调用配置：

```cpp
// Union任务配置：使用UNION1类型，一个Cluster内的多个Core协作处理
cnrtDim3_t dim = {CNRT_FUNC_TYPE_UNION1, BLOCKS, 1};
cnrtFunctionType_t func_type = CNRT_FUNC_TYPE_UNION1;

CNRT_CHECK(cnrtPlaceNotifier(st_mlu, queue));
multiplyMatricesMLU<<<dim, func_type, queue>>>(
    left_mlu, right_mlu, result_mlu, m, n, k);
CNRT_CHECK(cnrtPlaceNotifier(et_mlu, queue));
CNRT_CHECK(cnrtQueueSync(queue));
CNRT_CHECK(cnrtNotifierElapsedTime(st_mlu, et_mlu, &mlu_time_used));
```

#### 5.2.5.7 性能对比

在矩阵乘性能优化实验中，测试环境参数如下：

- CPU：Intel(R) Xeon(R) Gold 6330 CPU @ 2.00GHz
- MLU：MLU370-X8@1000MHz
- MLU Driver：v5.10.26
- MLU Firmware：v1.1.6
- CNToolkit：v3.9.0

各优化阶段的性能对比如下：

| 实验步骤 | 矩阵乘规模 (M×N×K) | CPU耗时 (ms) | MLU耗时 (ms) |
|----------|---------------------|-------------|-------------|
| 标量实现 | 128×256×128 | 5.857 | 2716.140 |
| 标量NRAM实现 | 128×256×128 | 5.847 | 126.067 |
| 向量NRAM实现 | 128×256×128 | 5.904 | 0.050 |
| 多核向量NRAM实现 | 524288×256×128 | 26075.229 | 4.563 |
| 多核向量NRAM三级流水实现 | 524288×256×128 | 26076.031 | 3.824 |
| 多核向量SRAM五级流水实现 | 524288×256×128 | 26081.332 | 4.073 |

从性能对比数据可以观察到各优化阶段的效果：

1. **标量→标量NRAM**：利用NRAM片上存储，MLU耗时从2716ms降至126ms，约21.5倍加速，体现了NRAM高带宽的优势。
2. **标量NRAM→向量NRAM**：利用`__bang_matmul`向量化接口，MLU耗时从126ms降至0.05ms，约2520倍加速，体现了SIMD向量化的巨大收益。
3. **向量NRAM→多核向量NRAM**：利用128核并行，矩阵规模扩大4096倍（M从128→524288），MLU耗时仅4.563ms，体现了多核并行的高效扩展性。
4. **多核向量→三级流水**：通过Load/Compute/Store流水线重叠，MLU耗时从4.563ms降至3.824ms，约16%提升，缓解了访存瓶颈。
5. **三级流水→五级流水**：引入SRAM层级后MLU耗时为4.073ms，与三级流水在同一数量级，主要优势在于支持更大规模的矩阵运算。

### 5.2.6 推荐实践内容

在矩阵乘性能优化实验中，每升高一级实践目标，不但要实现当前级别的代码，还要实现上一级别的代码。即最高级实践要求实现"标量NRAM实现"、"向量NRAM实现"、"多核向量NRAM实现"、"多核向量NRAM三级流水实现"、"多核向量SRAM五级流水实现"共5个版本的BANG C代码。推荐实践目标如下：

- **基础实践（标量NRAM）**：M×N×K = 128×256×128规模，实现标量NRAM矩阵乘，MLU计算结果与CPU计算结果误差为0。
- **向量化实践（向量NRAM）**：M×N×K = 128×256×128规模，实现向量NRAM矩阵乘，精度达标，MLU性能相比标量NRAM版本有千倍以上提升。
- **多核并行实践（多核向量NRAM）**：M×N×K = 524288×256×128规模，实现多核向量NRAM矩阵乘，精度达标，MLU性能大幅超越CPU版本。
- **流水线优化实践（三级流水）**：M×N×K = 524288×256×128规模，实现多核向量NRAM三级流水矩阵乘，精度达标，MLU性能相比无流水版本有进一步提升。
- **高级流水线实践（五级流水）**：M×N×K = 524288×256×128规模，实现多核向量SRAM五级流水矩阵乘，精度达标，性能和三级流水在同一数量级。

### 5.2.7 实验思考

1. CPU上实现矩阵乘有哪些可以加速的方法？请尝试改写CPU版本的矩阵乘实现和MLU重新做性能对比并分析峰值性能的上限由什么因素决定。
2. 为什么矩阵乘的标量实现中CPU和MLU的精度一致，而使用了向量接口`__bang_matmul`后精度不一致？是BANG C提供的向量加速接口计算错误吗？
3. 什么是分块矩阵乘？分块矩阵乘和不分块时精度有何差异？精度差异的来源是什么？
