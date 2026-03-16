**参考文献以及网络资源**：【【2025/Minimind】Only三小时！Pytorch从零手敲大模型，架构到训练全教程】 https://www.bilibili.com/video/BV1T2k6BaEeC/?p=11&share_source=copy_web&vd_source=322e0d48ab2dd50c95f3d4768a25b22f（主要参考）

​					https://blog.csdn.net/Dust_Evc/article/details/127502272（前置知识：点积、叉积、内积、外积）

​					https://docs.pytorch.ac.cn/docs（前置知识：pytorch官方文档）

​					https://www.runoob.com/python3（菜鸟教程 python3）

​					https://github.com/jingyaogong/minimind（minimind源项目）

# 前置知识

模型训练步骤：

![](C:\Users\86198\AppData\Roaming\Typora\typora-user-images\image-20260305223210152.png)

## 点积、叉积、内积、外积

https://blog.csdn.net/Dust_Evc/article/details/127502272

## pytorch

### 1.pytorch介绍

​	pytorch是为python设计的用于构建和训练深度神经网络的工具包，pytorch识别数字分步：准备数据、定义模型、训练模型、评估模型、做出预测。PyTorch 是一个开源的深度学习框架，以其灵活性和动态计算图而广受欢迎。PyTorch 主要有以下几个基础概念：张量（Tensor）、自动求导（Autograd）、神经网络模块（nn.Module）、优化器（optim）等。

* **张量（Tensor）**：PyTorch 的核心数据结构，支持多维数组，并可以在 CPU 或 GPU 上进行加速计算。

- **自动求导（Autograd）**：PyTorch 提供了自动求导功能，可以轻松计算模型的梯度，便于进行反向传播和优化。
- **神经网络（nn.Module）**：PyTorch 提供了简单且强大的 API 来构建神经网络模型，可以方便地进行前向传播和模型定义。
- **优化器（Optimizers）**：使用优化器（如 Adam、SGD 等）来更新模型的参数，使得损失最小化。
- **设备（Device）**：可以将模型和张量移动到 GPU 上以加速计算。

#### PyTorch 架构总览

PyTorch 采用模块化设计，由多个相互协作的核心组件构成。理解这些组件的作用和相互关系，是掌握 PyTorch 的关键。

##### PyTorch 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    PyTorch 生态系统                          │
├─────────────────────────────────────────────────────────────┤
│  torchvision  │  torchtext  │  torchaudio  │  其他专业库     │
├─────────────────────────────────────────────────────────────┤
│                     PyTorch 核心                            │
├───────────────┬─────────────────┬───────────────────────────┤
│   torch.nn    │   torch.optim   │      torch.utils          │
│   (神经网络)   │   (优化器)      │      (工具函数)           │
├───────────────┼─────────────────┼───────────────────────────┤
│               │                 │   torch.utils.data        │
│  torch 核心   │  autograd       │   (数据加载)              │
│  (张量计算)   │  (自动微分)     │                           │
└───────────────┴─────────────────┴───────────────────────────┘
```

PyTorch 采用**分层架构**设计，从上层到底层依次为：

**1、Python API（顶层）**

- `torch`：核心张量计算（类似NumPy，支持GPU）。
- `torch.nn`：神经网络层、损失函数等。
- `torch.autograd`：自动微分（反向传播）。
- 开发者直接调用的接口，简单易用。

**2、C++核心（中层）**

- **ATen**：张量运算核心库（400+操作）。
- **JIT**：即时编译优化模型。
- **Autograd引擎**：自动微分的底层实现。
- 高性能计算，连接Python与底层硬件。

**3、基础库（底层）**

- **TH/THNN**：C语言实现的基础张量和神经网络操作。
- **THC/THCUNN**：对应的CUDA（GPU）版本。
- 直接操作硬件（CPU/GPU），极致优化速度。

**执行流程**：
Python代码 → C++核心计算 → 底层CUDA/C库加速 → 返回结果。
既保持易用性，又确保高性能。

![img](https://www.runoob.com/wp-content/uploads/2024/12/iGWbOXL.png)

#### 张量 tensor

张量（Tensor）是 PyTorch 中的核心数据结构，用于存储和操作多维数组。

张量可以视为一个多维数组，支持加速计算的操作。

在 PyTorch 中，张量的概念类似于 NumPy 中的数组，但是 PyTorch 的张量可以运行在不同的设备上，比如 CPU 和 GPU，这使得它们非常适合于进行大规模并行计算，特别是在深度学习领域。

- **维度（Dimensionality）**：张量的维度指的是数据的多维数组结构。例如，一个标量（0维张量）是一个单独的数字，一个向量（1维张量）是一个一维数组，一个矩阵（2维张量）是一个二维数组，以此类推。
- **形状（Shape）**：张量的形状是指每个维度上的大小。例如，一个形状为`(3, 4)`的张量意味着它有3行4列。
- **数据类型（Dtype）**：张量中的数据类型定义了存储每个元素所需的内存大小和解释方式。PyTorch支持多种数据类型，包括整数型（如`torch.int8`、`torch.int32`）、浮点型（如`torch.float32`、`torch.float64`）和布尔型（`torch.bool`）。

————————————————————————————————

​	定义模型前，在识别图像中**卷积层**是将图像变成一系列0-1数据矩阵，**池化层**则可以理解为将原来很大的图像矩阵压缩为更小更容易计算的图像矩阵，**全连接层**则将图像矩阵展开为一段数字标识图像。

​	训练模型时，**损失函数**衡量模型预测的准确性，**优化器**用于调整模型的参数

​	实现特征提取使用到的技术有卷积和池化：

* 卷积是一种线性运算，即将一组权重与输入相乘，卷积层通常用于捕捉边缘、颜色、基本几何形状等基本特征。卷积层数增多之后，模型能够提取更高级的特征

![image-20260305203153391](C:\Users\86198\AppData\Roaming\Typora\typora-user-images\image-20260305203153391.png)

![image-20260305203234786](C:\Users\86198\AppData\Roaming\Typora\typora-user-images\image-20260305203234786.png)

连接多个“聚焦镜头”（卷积核），就可以表示出与训练数据中的已知数据相匹配的复杂图案

![image-20260305203340736](C:\Users\86198\AppData\Roaming\Typora\typora-user-images\image-20260305203340736.png)

* 池化层能够缩小表示的空间的大小，提高计算效率，池化层会单独对每个特征图进行运算。池化层中常用的方法是最大池化 ，即捕捉数组的最大值，从而减少计算所需的值的数量

### 2.单层CNN网络构建

![image-20260305204016296](C:\Users\86198\AppData\Roaming\Typora\typora-user-images\image-20260305204016296.png)

* 3 继承自torch.CNN.Module，这是Pytorch中所有神经网络模块的基类（又称父类，允许创建可重用的、模块化的代码，并且通过继承机制，可以扩展和定制基类的功能），为构建和训练神经网络提供了基础
* 5 self是构造函数固定使用的参数，n_channels代表输入图像的通道数（彩色图像有三个通道：红绿蓝；而灰度图像只有一个通道 ）
* 6 调用了父类Module的初始化方法，确保正确地初始化模型（前三行代码相当于 定义了一个方法，之后想使用直接调用其名字即可，无需设置）
* 8 第一个卷积层接收 n_channels个通道，并输出32个特征图（即32个卷积层用于处理训练数据集，每个卷积核可以提取一种 特征，最终得到32个特征映射，输出32个特征图），使用的是3×3的卷积核进行卷积运算；Conv2d是Pytorch中用于创建二维卷积层的类
* 9 调用kaiming_uniform_方法并设置激活函数为ReLu，初始化卷积层的权重（深度学习模型训练过程的本质是对Weight即参数进行更新，self.hidden1.weight表示卷积层1的张量，后面的参数则说明指明的激活函数为ReLU）
* 10 创建了一个 ReLU函数并将其赋值给act1属性（激活函数相当于是神经元之间决定是否传递信息的开关，ReLU输入信息是负值->输出为0；输入信息是正值->输出为原数值，优点是处理简单，计算速度快）
* 12 创建了一个最大池化层，它将输入的特征图分割为2×2大小的区域，并从每个区域中取最大值。 stride(2, 2)是池化操作的步长，决定 了池化核在输入特征图上滑动的速度，通过池化在保留最重要特征的同时，图的尺寸进一步缩小，减少了模型的计算量和参数数量，同时让模型不至于非常符合训练的模型但不能用于其他的模型，即减少过拟合，提高泛化能力
* 综上：卷积层需要设置通道数 、卷积核数目、卷积核大小；池化层需要设置池化方法，池化核大小和步长，这些参数的选择会影响到卷积神经网络的性能和泛化能力

### 3.构建多层神经网络

方法即在已构建的单层CNN后添新的卷积层和池化层即可

* ![image-20260305210929997](C:\Users\86198\AppData\Roaming\Typora\typora-user-images\image-20260305210929997.png)

* 代码解读：

  * ![image-20260305211032238](C:\Users\86198\AppData\Roaming\Typora\typora-user-images\image-20260305211032238.png)

    * 卷积层2仍使用ReLU激活函数以优化使用效果，在此再次被用来引入非线性，帮助网络捕捉复杂的输入模式
    * 池化层2的作用是通过减少特征图的空间尺寸来降低后续网络层的计算负载并在一定程度上增强模型的抗噪声能力 

  * 通过多层将原始输入转化为高级特征从而为后续的分类或其他任务提供必要的信息，这种层序结构是深度学习中处理图像任务的一种常见且有效的策略

  * 经过两层卷积和池化（图像的特征已被有效提取并压缩）后下一步是进入全连接层：对前面卷积层提取的特征进行整合以进行最终的分类决策，类似与神经网络中的“连接器”，它能够将前一层的所有神经元与当前层的每个神经元的相连接，从而使网络能够学习并理解更加复杂和抽象的特征  

    ![image-20260305212124947](C:\Users\86198\AppData\Roaming\Typora\typora-user-images\image-20260305212124947.png)![image-20260305212244663](C:\Users\86198\AppData\Roaming\Typora\typora-user-images\image-20260305212244663.png)

​		softmax激活函数将决策中心的输出转化为概率分布，评估每个	决策可能性的方式，并根据概率选择最高的类别来进行决策

![image-20260308225508808](C:\Users\86198\AppData\Roaming\Typora\typora-user-images\image-20260308225508808.png)

* 完整代码 ：

  ![image-20260305212734138](C:\Users\86198\AppData\Roaming\Typora\typora-user-images\image-20260305212734138.png)

### 4.前向传播

* 定义：在神经网络的训练过程中，它指的是数据在神经网络中的流动方向，即从输入层经过隐藏层

![image-20260305214613611](C:\Users\86198\AppData\Roaming\Typora\typora-user-images\image-20260305214613611.png)



![image-20260305215112887](C:\Users\86198\AppData\Roaming\Typora\typora-user-images\image-20260305215112887.png)

​								例图

![](C:\Users\86198\AppData\Roaming\Typora\typora-user-images\image-20260305215349412.png)



​	首先在输入层输入图片，斑马的图片先被转化为数字格式，这些数字代表了图片中的每个像素点的颜色值，这些像素值作为输入数据被送入神经网络的输入层进行特征提取，输入数据提取完特征，进入网络后会通过一系列的隐藏层，在每个隐藏层中，输入数据会与该层的权重矩阵相乘并加上偏置项，此过程可表示为（其中W是权重矩阵，x是输入数据，b是偏置项）：

![image-20260305215510990](C:\Users\86198\AppData\Roaming\Typora\typora-user-images\image-20260305215510990.png)

乘加操作后的结果会通过一个激活函数（引入非线性，使得神经网络能够学习和模拟更加复杂的函数映射）

![image-20260305215804709](C:\Users\86198\AppData\Roaming\Typora\typora-user-images\image-20260305215804709.png)

​	上述过程会在网络中每个隐藏层中重复进行直到数据到达输出层，输出层的激活函数通常是Softmax会将输出值转换为概率形式表示输出图片属于各个类别的概率（在最初的图片中，可能有一个神经元对应“斑马”类别，其输出值越高，表示神经网络认为输入图片是斑马）

​	神经网络的输出与实际标签（如斑马）进行比较，通过损失函数（如交叉熵损失）计算两者之间的差异

#### 附  反向传播

​	在前向传播之后进行的，神经网络会通过反向传播算法来调整权重和偏置以减少损失函数的值，目的是计算损失函数关于网络参数的梯度以便通过梯度下降等优化算法更新网络的权重，从而改进模型的性能

![image-20260305220632746](C:\Users\86198\AppData\Roaming\Typora\typora-user-images\image-20260305220632746.png)

​	在斑马的例图中，前向传播是神经网络处理输入数据并生成预测结果的过程，在此过程中，输入数据经过一系列的数学运算包括卷积、激活函数和池化最终在输出层生成预测结果

![image-20260305220919105](C:\Users\86198\AppData\Roaming\Typora\typora-user-images\image-20260305220919105.png)

![image-20260305220945067](C:\Users\86198\AppData\Roaming\Typora\typora-user-images\image-20260305220945067.png)

模型架构定义了前向传播的路径和操作顺序，前向传播路径上执行具体的数据处理、层与层之间的数据传传递和参数计算 

* 代码实现：
  * ![image-20260305221226077](C:\Users\86198\AppData\Roaming\Typora\typora-user-images\image-20260305221226077.png)
  * 在这个函数中，每个语句的输入输出都是训练集X，每个语句都控制训练集在之前定义的模型层中进行训练，此处的self表示在上一步中定义的模型实例（即hidden1，act1都是多层CNN中已定义的）
    * 其中第12行所进行的是数据扁平化处理：将多维数据结构转换为一维的向量形式，这样做是为了让数据能够被神经网络处理（大多数都是设计为处理一维数据的）
  * 在前向传播的过程中，每个步骤都是一个神经网络层，数据流转方式相应的由我们定义的方法所确定并发生相应的变化

* 总结：整个前向传播的过程即数据从输入层开始，经过一系列的变换（加权求和、偏置、激活函数）最终在输出层产生一个预测结果，这个结果的好坏将决定网络是否需要通过反向传播进行调整

### 5.计算误差、参数更新

* 损失函数（Loss Function）：用于衡量模型预测值与真实值之间的差距，回答"我现在表现得有多差？"这个问题。它是模型训练的目标函数，损失值越小表示模型预测越准确。

* 优化器（Optimizer）负责根据损失函数计算出的梯度更新模型参数，回答"我该怎么做才能变好？"这个问题。其核心目标是以最有效的方式更新参数以**最小化损失函数**。

* 代码实现（损失函数：交叉熵损失；优化器：随机梯度下降）：

  * ![image-20260305224332787](C:\Users\86198\AppData\Roaming\Typora\typora-user-images\image-20260305224332787.png)SGD参数：modles.parameters()决定模型使用的所有参数，学习率lr决定了每次参数更新的步长大小，动量momentum通过在更新方向上加速帮助优化器在鞍点附近更快地收敛并减小震荡，加速收敛并避免陷入局部最优解
  * ![image-20260305224847268](C:\Users\86198\AppData\Roaming\Typora\typora-user-images\image-20260305224847268.png)一个epoch指的是算法在整个训练数据集上完整地进行一次前向和后向传播的过程，多次遍历可以使模型更好学习数据的特征，提高泛化能力，每个epoch可以看作是一次完整地下降过程（训练损失是指模型在训练数据集上的平均损失值，越低表示训练数据的拟合度越好；训练准确率使衡量模型在训练数据集上预测正确的比例；测试准确率是衡量模型在测试数据集上的预测准确率，真实反映模型对未知数据的泛化能力)
  * ![image-20260305225425550](C:\Users\86198\AppData\Roaming\Typora\typora-user-images\image-20260305225425550.png)每个mini_batch包含一小部分训练样本
  * ![image-20260305225550208](C:\Users\86198\AppData\Roaming\Typora\typora-user-images\image-20260305225550208.png)
  * ![image-20260305225626312](C:\Users\86198\AppData\Roaming\Typora\typora-user-images\image-20260305225626312.png)
  * ![image-20260305225650229](C:\Users\86198\AppData\Roaming\Typora\typora-user-images\image-20260305225650229.png)
  * ![image-20260305225716194](C:\Users\86198\AppData\Roaming\Typora\typora-user-images\image-20260305225716194.png)
  * ![image-20260305225740989](C:\Users\86198\AppData\Roaming\Typora\typora-user-images\image-20260305225740989.png)


## CS336（待更新）

### 1. token和Tokenizer

​	Tokenizer分词算法是[NLP](https://zhida.zhihu.com/search?content_id=260527376&content_type=Article&match_order=1&q=NLP&zhida_source=entity)大模型最基础的组件，基于Tokenizer可以将文本转换成独立的token列表，进而转换成输入的向量成为计算机可以理解的输入形式。本文将对分词器进行系统梳理，包括分词模型的演化路径，可用的工具，并手推每个tokenizer的具体实现.

​	简单来说，Tokenizer 就是将连续的文本拆分成模型能处理的基本单位——token 的工具，而 token 是模型理解和生成文本的最小单位。对于计算机来说，处理原始文本是非常困难的，因此我们需要一个中间层，把文字转换为一系列的数字序列（即，一个个离散的 token），这些 token 既可以是单个字符、词语，也可以是子词（subword）。而这个转换过程正是由 Tokenizer 完成的。

​	在传统的自然语言处理中，我们可能直接按照单词（每个token是一个完整单词易于理解处理，但对语料库要求高增加存储开销）或字符（不存在OOV未登录词但丢失语义信息）来分割文本；而在大模型中，常见的方法则是采用子词级别（subword-level）的分割方式。这种方式既能保证足够细致（能够捕捉到拼写变化、罕见词等信息），又不会使得词表过大，进而影响模型的效率和泛化能力。而subword-level方式中最常用的便是BPE：

#### BPE（Byte-Pair Encoding）

##### 原理介绍

​	“ The BPE algorithm is "byte-level" because it runs on UTF-8 encoded strings.”核心是一个**贪心迭代合并**过程，经过Pre_tokenization后，BPE会确保最常见的词在token列表中表示为单个token，而罕见的词被分解为两个及多个subword tokens（如lowereset分解为low和est）。

​	假设我们有一个语料库，其中包含单词（pre-tokenization之后）—— old, older, highest, 和 lowest，我们计算这些词在语料库中的出现频率。假设这些词出现的频率如下：

{“old”: 7, “older”: 3, “finest”: 9, “lowest”: 4}

让我们在每个单词的末尾添加一个特殊的结束标记“</w>”。

{“old</w>”: 7, “older</w>”: 3, “finest</w>”: 9, “lowest</w>”: 4}

在每个单词的末尾添加“</w>”标记以标识单词边界能够让算法知道**每个单词的结束位置**（因为我们统计相邻字符对时不能把**分别位于两个单词**中的字符对算进去），这有助于算法查看每个字符并找到频率最高的字符配对。稍后我们将看到“</w>”也能被算作字符对的一部分。

接下来，我们将每个单词拆分为字符并计算它们的出现次数。初始token将是所有字符和“</w>”标记的集合。

![img](https://pic2.zhimg.com/v2-3172eb05aa528f23e90fb0bb2cc62839_1440w.jpg)

BPE 算法的下一步是寻找最频繁的字符对，合并它们，并一次又一次地执行相同的迭代，直到达到我们预先设置的token数限制或迭代限制。

合并字符可以让你**用最少的token来表示语料库**，这也是 BPE 算法的主要目标，即**数据的压缩**。为了合并，BPE 寻找最常出现的字节对。在这里，我们将字符视为与字节等价。当然，这只是英语的用法，其他语言可能有所不同。现在我们将最常见的字节对合并成一个token，并将它们添加到token列表中，并重新计算每个token出现的频率。这意味着我们的频率计数将在每个合并步骤后发生变化。我们将继续执行此合并步骤，直到达到我们预先设置的token数限制或迭代限制。

**为了更清晰的理解，看完下面完整的迭代过程：**

**迭代 1：**我们将从第二常见的标记“e”开始。 在我们的语料库中，最常见的带有“e”的字节对是“e”和“s”（在finest和lowest两个词中），它们出现了 9 + 4 = 13 次。 我们将它们合并以形成一个新的token“es”并将其频率记为 13。我们还将从单个token（“e”和“s”）中减少计数 13，从而我们知道剩余的“e”或“s”token数。 我们可以看到“s”不会单独出现，“e”出现了 3 次。 这是更新后的表格：

![img](https://pic3.zhimg.com/v2-79975ec59166e44cbe4b17cb83571e28_1440w.jpg)

**迭代 2：**我们现在将合并token“es”和“t”，因为它们在我们的语料库中出现了 13 次。 因此，我们有一个频率为 13 的新token“est”，我们会将“es”和“t”的频率减少 13。

![img](https://pic3.zhimg.com/v2-7cac758046b37d44a7d07f206601ca7e_1440w.jpg)

**迭代 3：**让我们现在考虑“</w>”token，我们看到字节对“est”和“</w>”在我们的语料库中出现了 13 次。

***注意：\***合并停止token“</w>”非常重要。 这有助于算法理解“estimate”和“highest”等词之间的区别。 这两个词都有一个共同的“est”，但一个词在结尾有一个“est”token，一个在开头。 因此，像“est”和“est</w>”这样的token将被不同地处理。 如果算法看到token“est</w>”，它就会知道它是“highest”这个词的token，而不是“estate”的。

![img](https://pic1.zhimg.com/v2-a368671aa1cac4bfa6f1524ff10aef06_1440w.jpg)

**迭代 4：**查看其他token，我们看到字节对“o”和“l”在我们的语料库中出现了 7 + 3 = 10 次。

![img](https://picx.zhimg.com/v2-08a053c3f996c9bd037b29b027e253d9_1440w.jpg)

**迭代 5：**我们现在看到字节对“ol”和“d”在我们的语料库中出现了 10 次。

![img](https://pic4.zhimg.com/v2-28f99e69748d8c9ac538225d9e09bb1d_1440w.jpg)

​	现在我们发现“f”、“i”和“n”的频率为 9，但我们只有一个单词包含这些字符，因此我们没有将它们合并。 为了本文的简单起见，让我们现在停止迭代并。看看我们最终的token列表：

![img](https://pic1.zhimg.com/v2-52a138eb8975c213840785e2c6850d60_1440w.jpg)

频率计数为 0 的token已从表中删除。我们现在可以看到列表的总token数为 11，这比我们最初列表的token数 12 个要少，说明token列表被有效压缩了。

​	你是不是想说这算法好像也没那么nb，token列表才减了1个token？但是你要知道，这是一个很小的语料库。在实际中，我们的预料库通常要大得多，从而我们能通过更多的迭代次数将token列表缩小更多的比例。

​	你一定也注意到，当我们添加一个token时，我们的计数既能增加也能减少，也能保持不变。在实际中，token计数先增加然后减少。算法的停止标准可以是token的计数或固定的迭代次数。我们选择一个最合适的停止标准，以便我们的数据集可以以最有效的方式分解为token。

## Transformer

3Blue1Brown 视频解析：

#### 一、GPT基础概念

**GPT**全称**Generative Pre-trained Transformer**，包含三个核心含义：

- **生成式（Generative）**：模型能够根据输入文本生成新的连贯文本，通过“预测-采样-重复”循环实现——给定初始文本，预测下一个词的概率分布，采样一个词追加，基于新文本再次预测，如此反复。
- **预训练（Pre-trained）**：模型在海量无标注文本数据（如网页、书籍、维基百科）上通过自监督学习预先训练，学习通用语言规律、语法结构和世界知识。
- **Transformer**：其核心神经网络架构，2017年由Google在《Attention Is All You Need》论文中提出，完全基于注意力机制，摒弃了传统的循环和卷积操作。

#### 二、Transformer架构核心组件

##### 1. 词嵌入（Word Embedding）

- 将文本中的每个词（token）转换为高维向量（如GPT-3使用12288维向量）。
- 在几何嵌入空间中，语义相似的词向量位置相近（如“猫”和“狗”比“猫”和“汽车”更接近）。
- 词向量不仅编码词语本身含义，还能通过后续处理吸收上下文信息。

##### 2. 位置编码（Positional Encoding）

- 由于Transformer并行处理整个序列，本身没有顺序概念，需要显式加入位置信息。
- 常用正弦余弦编码公式：`PE(pos,2i)=sin(pos/10000^(2i/d_model))`，`PE(pos,2i+1)=cos(pos/10000^(2i/d_model))`
- 位置编码与词嵌入向量相加，使模型感知词语在序列中的位置。

##### 3. 自注意力机制（Self-Attention）——Transformer的灵魂

###### QKV三矩阵机制：

- **Query（查询）**：当前词的“查询向量”，代表“我想找什么信息”。
- **Key（键）**：其他词的“键向量”，代表“我能提供什么线索”。
- **Value（值）**：其他词的“值向量”，代表“我的具体语义内容”。
- 通过三个独立的可学习权重矩阵W_Q、W_K、W_V从输入向量生成Q、K、V。

###### 注意力计算六步骤：

1. **生成Q、K、V**：输入矩阵X分别乘以W_Q、W_K、W_V
2. **计算注意力得分**：`attention_score = Q × K^T`（矩阵乘法）
3. **缩放**：`attention_score/√(d_k)`，防止d_k过大导致softmax梯度消失
4. **掩码（可选）**：解码器中使用因果掩码，屏蔽未来位置信息
5. **Softmax归一化**：将得分转换为注意力权重（每行和为1）（指数化后平均加权）
6. **加权求和**：`输出 = 注意力权重 × V`

数学公式：

```
Attention(Q,K,V) = softmax(QK^T/√(d_k))V
```

##### 4.多头注意力（Multi-Head Attention）

- 并行计算多组（如8、12、96个头）独立的注意力，每个头将输入投影到不同的子空间。
- 每个头关注不同方面的语义关系：有的头关注语法结构，有的关注指代关系，有的关注情感修饰。
- 最终将所有头的输出拼接，通过线性层W_O映射回原维度。
- 公式：`MultiHead(Q,K,V) = Concat(head_1,...,head_h)W_O`

##### 5. 掩码机制（Masking）

- **填充掩码（Padding Mask）**：屏蔽填充token（如[PAD]）的注意力。
- **因果掩码（Causal Mask）**：解码器专用，确保生成当前词时只能看到前面已生成的词，防止“偷看未来信息”。

##### 6. 前馈神经网络（Feed-Forward Network）

- 每个Transformer层中，注意力输出后接一个两层全连接网络。
- 公式：`FFN(x) = max(0, xW_1 + b_1)W_2 + b_2`
- 对每个位置向量独立处理，提取更高级特征。

##### 7. 残差连接与层归一化

- **残差连接**：将子层输入直接加到输出上，缓解梯度消失，公式：`输出 = LayerNorm(x + Sublayer(x))`
- **层归一化**：对每层所有神经元输出标准化（均值为0，方差为1），加速训练收敛。

#### 三、GPT模型架构特点

- **Decoder-only架构**：GPT只使用Transformer的解码器部分，没有编码器-解码器注意力。
- **单向注意力**：使用因果掩码，只能关注当前位置及之前的词，适合自回归文本生成。
- **堆叠层数**：GPT-3有96层Transformer，每层包含掩码多头注意力和前馈网络。

#### 四、GPT训练三阶段

##### 1. 预训练（Pre-training）

- **目标**：学习通用语言表示能力。
- **方法**：自监督学习，从海量无标注文本中构造任务——给定前k个词，预测第k+1个词。
- **数据规模**：GPT-3使用约3000亿token，来自Common Crawl、WebText2、Books1/2、Wikipedia等。
- **损失函数**：交叉熵损失，最大化下一个词预测概率。

##### 2.监督微调（Supervised Fine-Tuning）

- **目标**：让模型理解并遵循人类指令。
- **方法**：使用高质量人工标注的指令-输出对（如Alpaca的52K数据）继续训练。
- **格式**：`{“instruction”: “任务描述”, “input”: “输入文本”, “output”: “理想输出”}`
- **效果**：激活模型潜在推理能力，适应问答、摘要、代码生成等特定任务。

##### 3. 对齐（Alignment）——RLHF

- **目标**：使模型输出符合人类价值观（Helpful、Honest、Harmless）。

- **方法**：基于人类反馈的强化学习（RLHF）：

  a. **奖励建模**：训练奖励模型RM，人工标注百万级样本，对同一指令的多个输出按质量排序。

  b. **强化学习**：用RM为SFT模型输出打分，通过PPO算法调整参数，使模型更倾向生成高分内容。

- **替代方案**：直接偏好优化（DPO），更高效稳定。

#### 五、GPT-3关键参数与能力

- **参数量**：1750亿（175B）参数
- **架构细节**：96层Transformer，隐藏层维度12288，注意力头数96
- **上下文窗口**：2048个token（早期版本）
- **训练计算**：约100万-1000万美元计算成本

#### 六、上下文学习（In-Context Learning）

- **Zero-shot（零样本）**：只给任务描述，无示例。如：“将英语翻译成法语：cheese =>”
- **One-shot（单样本）**：给一个示例。如：“sea otter => loutre de mer\n cheese =>”
- **Few-shot（少样本）**：给10-100个示例，GPT-3表现最惊人的地方
- **关键特点**：不需要梯度下降更新权重，仅通过注意力机制“读取”prompt中的模式。

#### 七、注意力机制直观理解

以句子“一个毛茸茸的蓝色生物漫步于葱郁的森林”为例：

- 当处理“生物”时，其Query向量会询问：“我的前面有形容词吗？”
- “蓝色”的Key向量回答：“我是形容词。”
- 通过Q·K点积计算相似度，“生物”与“蓝色”获得高注意力权重。
- “蓝色”的Value向量包含“颜色修饰”语义信息，加权融合到“生物”的新表示中。
- 同理，“毛茸茸”的纹理信息也被融合，使“生物”向量同时包含颜色和纹理特征。

#### 八、参数高效微调（PEFT）技术

由于全参数微调1750亿参数成本过高，发展出：

- **LoRA（低秩适应）**：冻结原权重，添加低秩矩阵A、B，只训练ΔW=BA，更新0.1%参数。
- **Adapter模块**：在Transformer层内插入小型瓶颈网络，仅训练这些微型适配器。
- **Prefix Tuning**：学习可训练的“软提示词”向量，拼接在输入前。

# minimind复现

## minimind架构图

![image-20260310104109272](C:\Users\86198\AppData\Roaming\Typora\typora-user-images\image-20260310104109272.png)



### Tokenizer Encoding：

* **做什么**：将输入的自然语言文本（如“hello world”）切割成模型能理解的离散单元（词元或子词），并转换为对应的数字ID序列。
* **原理**：通常使用如Byte-Pair Encoding (BPE) 等子词分词算法。它通过统计词频，将常见单词作为一个整体，生僻词拆分为有意义的子词（如“playing”->“play”+“ing”），在词汇表大小和语义表示间取得平衡。
* **为什么**：计算机无法直接理解文字。分词是连接人类语言与数学模型的**第一道桥梁**。好的分词能有效控制词汇表规模，缓解未登录词（OOV）问题，并保留一定的语义和形态学信息。

### Input Embedding：

* **做什么**：将上一步得到的数字ID序列，通过一个可学习的查找表，映射为**稠密、连续**的向量序列。图中的“Decoder”入口其实就隐含了这一步。
* **原理**：每个唯一的词元ID对应一个高维向量（例如768维）。这个向量在训练过程中会不断更新，最终使得语义相近的词（如“king”和“queen”）在向量空间中的位置也接近。
* **为什么**：将离散符号转换为连续向量，是应用后续所有基于连续数学的神经网络层（如线性变换、注意力）的**必要前提**。它为模型提供了最基础的、可学习的语义表示。

### Transformer Layer：

* 这是模型的核心，由 **k** 个相同的 `Transformer Layer`堆叠而成。每层主要包含两个子模块：**GQA（分组查询注意力）** 和 **FFN（前馈神经网络）**，并辅以 **RMSNorm（均方根归一化）** 和**残差连接**（图中以环绕箭头表示）。

  #### **1. 分组查询注意力 (GQA - Grouped-Query Attention)**

  - **做什么**：让序列中的每个词元，能够根据其与序列中所有其他词元（包括自身）的“相关性”或“重要性”，动态地聚合全局信息。图中的 **Mask** 确保了在生成下一个词时，只能看到它之前的词（自回归特性）。

  - **原理**：

    1. **线性变换**：对当前层的输入（记为 `H`）分别进行三次线性变换，得到**查询(Q)、键(K)、值(V)** 三组向量。
    2. **RoPE旋转位置编码**：考虑到词顺序，我们需要位置编码（PE），将绝对位置信息巧妙地编码到Q和K中。原理是对Q和K向量的每一维进行复数空间旋转，旋转角度与词元的位置序号相关。这样，计算注意力时，内积 `<Q_m, K_n>`会自然地包含位置m和n的相对距离信息。
    3. **注意力计算**：计算 `Scaled Dot-Product Attention`。`注意力分数 = Softmax( Mask( (Q * K^T) / sqrt(d_k) ) )`，然后用这个分数对 `V`进行加权求和，得到当前词元融合了全局上下文的新表示。

  - **为什么**：

    - **注意力机制**：解决了传统RNN的长程依赖和并行计算难题，是Transformer理解上下文关系的**核心动力**。
    - **RoPE**：相比绝对位置编码，RoPE能更好地外推到训练时未见过的更长序列，并且理论上能保持注意力分数的相对性不变。RoPE使用旋转位置编码，用非常巧妙的数学方法，将两个词的计算只和相对位置联系起来，而不与绝对位置有关

    ​     ![img](https://mirage-thought-d06.notion.site/image/attachment%3Adf1963f4-254f-4aa5-b166-e1934457f603%3Aimage.png?table=block&id=2a074782-5dae-8029-936c-e000586c4e99&spaceId=1ca89c23-3537-48e4-89af-137e2bfbe91e&width=1250&userId=&cache=v2)

    但RoPE也有问题，在长度外推的时候会出现错误。在此基础上的YARN，对低维和高维使用不同的旋转频率和速度

    ```python
    def precompute_freqs(
        dim: int,
        end: int = int(32 * 1024),
        rope_base: float = 1e6,
        rope_scaling: Optional[dict] = None,
    ):
        # 1. 初始化标准 RoPE 频率。
        # torch.arange(0, dim, 2) 生成 [0, 2, 4, ... dim-2]
        # 计算出的 freqs 就是标准的 1 / (base ** (2i / d))
        freqs, attn_factor = (
            1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)),
            1.0,
        )
    
        if rope_scaling is not None:
            # 2. 从配置字典中提取 YaRN 的超参数
            # orig_max: 模型预训练时的原始最大长度（例如 Llama-2 是 2048 或 4096）
            # factor: 要扩展的倍数 s (比如从 2k 扩展到 32k，factor 就是 16)
            # beta_fast (对应论文中的 α): 高频边界，波长比例大于此值的维度不缩放
            # beta_slow (对应论文中的 β): 低频边界，波长比例小于此值的维度全量缩放
            # attn_factor: 注意力温度补偿，由于距离拉长导致注意力分布发散（变平缓），需要乘上一个系数让注意力重新“聚焦”
            orig_max, factor, beta_fast, beta_slow, attn_factor = (
                rope_scaling.get("original_max_position_embeddings", 2048),
                rope_scaling.get("factor", 16),
                rope_scaling.get("beta_fast", 32.0),
                rope_scaling.get("beta_slow", 1.0),
                rope_scaling.get("attention_factor", 1.0),
            )
    
            # 只有当要推断的长度大于原始训练长度时，才应用缩放
            if end / orig_max > 1.0:
                # 3. 使用前文推导的公式，定义波长比例 b 到维度索引 i 的映射函数
                inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (
                    2 * math.log(rope_base)
                )
    
                # 4. 计算高频区和低频区的维度切分点
                # low: 不需要缩放的高频部分的最高索引
                # high: 需要完全缩放的低频部分的最低索引
                low, high = (
                    max(math.floor(inv_dim(beta_fast)), 0),
                    min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1),
                )
    
                # 5. 计算混合因子 γ (Ramp)
                # 在 low 之前，ramp 为 0；在 high 之后，ramp 为 1；在 low 和 high 之间，线性过渡。
                # clamp 函数限制了数值只能在 [0, 1] 之间。
                ramp = torch.clamp(
                    (torch.arange(dim // 2, device=freqs.device).float() - low)
                    / max(high - low, 0.001),
                    0,
                    1,
                )
    
                # 6. 频率融合公式：f'(i) = f(i) * ((1-γ) + γ/s)
                # 当 ramp=0 时（高频）：系数为 1，保持原频率不变。
                # 当 ramp=1 时（低频）：系数为 1/factor，即对频率进行线性插值缩放。
                # ramp在0-1之间时：平滑过渡。
                freqs = freqs * (1 - ramp + ramp / factor)
    
        # 7. 根据目标长度 end，生成位置索引向量 t
        t = torch.arange(end, device=freqs.device)
    
        # 8. 计算外积：将位置 t 与处理好的频率 freqs 相乘，得到每个位置的旋转角度 θ
        freqs = torch.outer(t, freqs).float()
    
        # 9. 计算 Cos 和 Sin，并应用注意力补偿系数 (attn_factor)
        freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
        freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    
        return freqs_cos, freqs_sin
    
    
    def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        def rotate_half(x):
            return torch.cat(
                (-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1
            )
    
        q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (
            rotate_half(q) * sin.unsqueeze(unsqueeze_dim)
        )
        k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (
            rotate_half(k) * sin.unsqueeze(unsqueeze_dim)
        )
        return q_embed, k_embed
    ```

    

    - **GQA设计**：这是对标准多头注意力(MHA)的优化。在MHA中，Q、K、V的头数相同；在GQA中，**K和V的头数被分组共享**（例如8个Q头共享2组K/V）。这大幅减少了推理时需要缓存和读取的K/V缓存大小，**在不显著损失效果的前提下，极大地提升了推理效率和内存利用率**，是小型和高效模型的常用技术。

  #### **2. 前馈神经网络 (FFN - Feed-Forward Network)**

  - **做什么**：对经过注意力机制聚合信息后的每个位置（词元）的表示，进行独立、复杂的非线性变换，增强模型的表示能力。
  - **原理**：它是一个两层全连接网络，中间夹着一个激活函数（图中为**SiLU**）。公式通常为：`FFN(x) = (Linear( SiLU( Linear(x) ) ))`。第一个 `Linear`会将维度扩大数倍（如从768到3072，称为“中间维度”），第二个 `Linear`再投影回原始维度。
  - **为什么**：注意力机制主要进行“信息筛选和聚合”，而FFN则充当**每个位置上的“深度处理器”**。高维的中间层让模型能够学习非常复杂的特征交互和模式，是提升模型容量的关键。SiLU激活函数通常比传统的ReLU更平滑，有时能带来更好的性能。

  #### **3. RMSNorm (均方根层归一化)**

  - **做什么**：对某一层的输入或输出数据进行标准化，使其均值为0，方差为1（近似）。

  - **原理**：RMSNorm相比传统的LayerNorm少了均值计算和减去均值两步操作，在训练中有不小的开销节省

    ![img](https://mirage-thought-d06.notion.site/image/attachment%3A499dd160-ee89-44dd-98f1-29223534f616%3Aimage.png?table=block&id=29d74782-5dae-80a6-8410-ddc515df4380&spaceId=1ca89c23-3537-48e4-89af-137e2bfbe91e&width=920&userId=&cache=v2)

    ```python
    class RMSNorm(torch.nn.Module):
        """
        RMS归一化 (Root Mean Square Normalization)
        相比LayerNorm，RMSNorm去掉了均值中心化，只保留方差缩放
        计算更简单，效果相当，在大模型中广泛使用
        """
        def __init__(self, dim: int, eps: float = 1e-5):
            """
            Args:
                dim: 归一化的维度大小
                eps: 防止除零的小常数
            """
            super().__init__()  # 调用父类nn.Module的构造函数
            self.eps = eps      # 存储epsilon值
            # nn.Parameter: 将tensor注册为可学习参数，会自动加入optimizer
            # torch.ones(dim): 创建全1的tensor作为缩放参数
            self.weight = nn.Parameter(torch.ones(dim))
    
        def _norm(self, x):
            """
            RMSNorm的核心计算：x / sqrt(mean(x^2) + eps)
            """
            # x.pow(2): 对x每个元素平方
            # .mean(-1, keepdim=True): 在最后一维求均值，保持维度
            # torch.rsqrt(): 计算平方根的倒数，即 1/sqrt(x)
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
        def forward(self, x):
            """
            前向传播
            Args:
                x: 输入tensor，shape为[batch, seq_len, dim]
            Returns:
                归一化后的tensor
            """
            # .float(): 转换为float32进行计算，提高数值稳定性
            # .type_as(x): 将结果转换回x的原始数据类型
            # self.weight *: 可学习的缩放参数
            return self.weight * self._norm(x.float()).type_as(x)
    ```

  - **为什么**：

    1. **稳定训练**：深度网络训练中，数据分布会随着层数加深而发生漂移（内部协变量偏移），归一化可以缓解此问题，使训练更稳定，允许使用更大的学习率。
    2. **RMSNorm的优势**：计算量比LayerNorm稍小，且在一些实验中表现出同等甚至更好的效果，成为许多新架构（如LLaMA）的选择。

  #### **4. 残差连接 (Residual Connection)**

  - **做什么**：将子层（如GQA或FFN）的输入，直接加到其输出上，作为该子层的最终输出。
  - **原理**：即 `输出 = 输入 + Sublayer(输入)`。
  - **为什么**：
    1. **缓解梯度消失**：为梯度提供了直接回传的捷径，使得训练极深层网络成为可能。
    2. **保护信息**：即使网络学到了一个恒等映射，也能保证信息不损失，让模型更专注于学习“增量变化”。

### MoE 混合专家模型

[(6 封私信 / 80 条消息) 一文带你详细了解：大模型MoE架构（含DeepSeek MoE详解） - 知乎](https://zhuanlan.zhihu.com/p/31145348325)

​	旨在为由多个单独网络组成的系统建立一个监管机制。在这种系统中，每个网络 (被称为“专家”) 处理训练样本的不同子集，专注于输入空间的特定区域。

​	**MoE 的一个显著优势是它们能够在远少于稠密模型所需的计算资源下进行有效的预训练**。这意味着在相同的计算预算条件下，您可以显著扩大模型或数据集的规模。特别是在预训练阶段，与稠密模型相比，混合专家模型通常能够更快地达到相同的质量水平。

​	作为一种基于 Transformer 架构的模型，混合专家模型主要由两个关键部分组成。

- **稀疏 [MoE 层](https://zhida.zhihu.com/search?content_id=255251579&content_type=Article&match_order=1&q=MoE+层&zhida_source=entity)**: 这些层代替了传统 Transformer 模型中的前馈网络 (FFN) 层。MoE 层包含若干“专家”(例如 8 个)，每个专家本身是一个独立的神经网络。在实际应用中，这些专家通常是前馈网络 (FFN)，但它们也可以是更复杂的网络结构，甚至可以是 MoE 层本身，从而形成层级式的 MoE 结构。（**稀疏性允许我们仅针对整个系统的某些特定部分执行计算**）
- **门控网络或路由**: 这个部分用于决定哪些Token (token) 被发送到哪个专家。例如，在下图中，“More”这个Token可能被发送到第二个专家，而“Parameters”这个Token被发送到第一个专家。有时，一个Token甚至可以被发送到多个专家。Token的路由方式是 MoE 使用中的一个关键点，因为路由器由学习的参数组成，并且与网络的其他部分一同进行预训练。

![img](https://pica.zhimg.com/v2-5bd5606dd339a2318e3661925cdeb132_1440w.jpg)

上图来自Switch Transformers论文的 MoE layer图示。总结来说，在混合专家模型 (MoE) 中，是将传统 Transformer 模型中的每个前馈网络 (FFN) 层替换为 MoE 层，其中 MoE 层由两个核心部分组成: 一个门控网络和若干数量的专家。

​	![img](https://picx.zhimg.com/v2-84ad6e0bcf1c37b296e38b06452a8833_r.jpg)

## 数据收集与训练

[minimind_dataset · 数据集](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files)

![image-20260314221707866](C:\Users\86198\AppData\Roaming\Typora\typora-user-images\image-20260314221707866.png)

**数据格式**：采用 **JSONL**（JSON Lines）格式。这是一种每行一个独立JSON对象的格式，具有内存友好、易于分割和并行处理的优点，非常适合处理海量文本数据。

**关键组件**：

- **特殊标识符**：图中列出了常见的控制标记，如代表序列开始的 `<BOS>`、代表序列结束的 `<EOS>`和用于填充长度的 `<PAD>`。这些是模型理解文本边界和进行批处理所必需的。
- **代码结构**：介绍了使用 `Dataset`类（需实现 `__len__`和 `__getitem__`方法）来定义如何读取和访问数据，并使用 `Dataloader`来高效地加载和分批数据。这是PyTorch等深度学习框架中的标准实践。

**数据示例**：右下角展示了一条具体的训练数据样本。其内容是多轮对话，使用特定的标记 `<|im_start|>`和 `<|im_end|>`来清晰地区分**角色（如用户、助手）和每轮的边界**。这显示了该数据集很可能用于训练对话模型或指令遵循模型。

### 预训练模型

![image-20260315212058176](C:\Users\86198\AppData\Roaming\Typora\typora-user-images\image-20260315212058176.png)

1.动态学习率公式是 **余弦退火（Cosine Annealing）** 的一种变体，结合了**热启动（Warmup）**：

* **η_t**: 当前时间步`t`的学习率。**η_min, η_max**: 学习率的下界与上界，是需精心调节的超参数。**T_cur**: 当前已进行的周期内迭代次数。**T_max**: 当前周期的总迭代次数（通常等于一个epoch或指定的步数）。**cos(π \* T_cur / T_max)**: 核心调度项，其值从 `1`平滑衰减至 `-1`。
* 训练初期，T_cur 很小，公式右侧余弦项的值较大，学习率接近 η_max，这就是 “先大步走” ，有利于快速逃离初始的局部最优点。_
* 随着 T_cur接近 T_max，余弦项趋近于 -1，学习率平滑下降至 η_min，这就是 “再精细调整” ，有助于模型在后期收敛到更优的局部最优点。

2.**深度学习训练闭环**：

1. **数据获取**：从 `DataLoader`中获取 `input_ids`（输入文本）、`labels`（目标文本，通常与 `input_ids`一样，用于预测下一个词），以及 `attention_mask`（用于区分真实内容与填充部分）。
2. **前向传播**：数据通过模型，计算预测结果。
3. **损失计算**：比较预测结果和 `labels`，计算交叉熵损失。
4. **反向传播**：计算损失相对于模型参数的梯度。
5. **梯度下降**：优化器（如AdamW）根据学习率和梯度更新模型参数。

3.梯度累计是一个**解决显存限制的经典技巧**

* **原理**：当GPU显存不足以支持大的 `batch_size`时，可以将一个大的“逻辑批次”拆分成多个小的“物理批次”。
* **操作**：连续进行几次（如4次）前向传播和反向传播，但**不立即更新参数**，而是将这小几步的梯度**累加**起来。在累积到预设步数后，用累积的总梯度**一次性**更新参数。
* **效果**：这等效于用更大的有效批次进行训练，有助于提高训练稳定性，并使大批量训练成为可能。

### loRA

​	**LoRA（Low-Rank Adaptation，低秩适配）**是一种**参数高效微调（PEFT）** 方法，允许在**不修改原始预训练模型权重**的情况下，通过添加少量可训练参数来适应新任务。传统微调需要更新整个模型（数十亿参数），而LoRA只在关键层注入可训练的低秩矩阵，只训练**1-5%的参数量**，实现**高效适配**。（由于r << min(d, k)，所以参数量d * r + r * k << d * k，r为两个的低秩矩阵的秩）

![image-20260316141049496](C:\Users\86198\AppData\Roaming\Typora\typora-user-images\image-20260316141049496.png)

## Transformer layer具体代码和原理解读

### RoPE、YaRN

#### 函数1：`precompute_freqs_cis`- 预计算旋转位置编码频率

##### 函数定义

```python
def precompute_freqs_cis(dim: int, end: int = int(32 * 1924), rope_base: float = 1e6, 
                         rope_scaling: Optional[dict] = None):
```

**数学原理**：定义函数，用于预计算旋转位置编码的频率矩阵。其中：

- `dim`：特征维度 d
- `end`：序列长度 L′
- `rope_base`：基础频率基数 base
- `rope_scaling`：YaRN扩展的配置参数

##### 标准RoPE频率计算

```
freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
```

**数学原理**：计算标准RoPE的基础频率 θi。

![image-20260312202128721](C:\Users\86198\AppData\Roaming\Typora\typora-user-images\image-20260312202128721.png)

##### YaRN扩展判断

```python
if rope_scaling is not None:
```

**数学原理**：如果提供了YaRN配置，则进行上下文长度扩展。

##### 提取YaRN超参数

```python
orig_max, factor, beta_fast, beta_slow, attn_factor = (
    rope_scaling.get("original_max_position_embeddings", 2048), 
    rope_scaling.get("factor", 16), 
    rope_scaling.get("beta_fast", 32.0), 
    rope_scaling.get("beta_slow", 1.0), 
    rope_scaling.get("attention_factor", 1.0)
)
```

**数学原理**：

- `orig_max`：原始预训练长度 L
- `factor`：扩展倍数 s=L′/L
- `beta_fast`：高频边界 βfast
- `beta_slow`：低频边界 βslow
- `attn_factor`：注意力温度补偿因子 t

##### 检查是否需要扩展

```python
if end / orig_max > 1.0:
```

**数学原理**：只有当目标长度 L′大于原始长度 L时，才需要应用YaRN扩展。

##### 波长比例到维度索引的映射函数

```python
inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
```

**数学原理**：计算波长比例 b对应的维度索引 i。

**推导**：

![image-20260312202440458](C:\Users\86198\AppData\Roaming\Typora\typora-user-images\image-20260312202440458.png)

这里 b是波长比例，与上述推导的 b相差因子 2π，本质相同。

##### 计算高低频分界点

```python
low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
```

**数学原理**：

- `low`：对应 βfast的维度索引，高频区的上界
- `high`：对应 βslow的维度索引，低频区的下界

**公式**：

![image-20260312202703229](C:\Users\86198\AppData\Roaming\Typora\typora-user-images\image-20260312202703229.png)

##### 计算混合因子 γi(Ramp)

```python
ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
```

**数学原理**：计算每个维度 i的混合系数 γi，实现NTK-by-parts插值。

**公式**：

![image-20260312202855753](C:\Users\86198\AppData\Roaming\Typora\typora-user-images\image-20260312202855753.png)

##### 应用频率融合公式

```python
freqs = freqs * (1 - ramp + ramp / factor)
```

**数学原理**：应用YaRN的频率缩放公式。

**公式**：

![image-20260312203035636](C:\Users\86198\AppData\Roaming\Typora\typora-user-images\image-20260312203035636.png)

##### 生成位置序列

```python
t = torch.arange(end, device=freqs.device).float()
```

**数学原理**：生成位置索引向量 m=0,1,...,L′−1。

##### 计算外积得到旋转角度矩阵

```python
freqs = torch.outer(t, freqs).float()
```

**数学原理**：计算每个位置 m在每个维度 i的旋转角度。

**公式**：

![image-20260312203146252](C:\Users\86198\AppData\Roaming\Typora\typora-user-images\image-20260312203146252.png)

##### 计算cos和sin并应用注意力补偿

```python
freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
```

**数学原理**：

1. 计算旋转角度的余弦和正弦值
2. 在最后一维复制一份，因为每个二维子空间的两个维度使用相同的 cos和 sin值
3. 乘以注意力补偿因子 t

**公式**：

![image-20260312203319400](C:\Users\86198\AppData\Roaming\Typora\typora-user-images\image-20260312203319400.png)

#### 函数2：`apply_rotary_pos_emb`- 应用旋转位置编码

##### 函数定义

```python
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
```

**数学原理**：将预计算的旋转位置编码应用到查询 Q和键 K向量上。

##### 定义半旋转函数

```python
def rotate_half(x):
    return torch.cat(
        (-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1
    )
```

**数学原理**：实现二维旋转矩阵的"半旋转"操作，对应于复数乘法中的乘以虚数单位 j。

**公式**：

![image-20260312203442454](C:\Users\86198\AppData\Roaming\Typora\typora-user-images\image-20260312203442454.png)

##### 应用旋转位置编码

```python
q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
```

**数学原理**：实现旋转位置编码的实数形式。

**公式**：

对于每个二维子空间 [x2i,x2i+1]T：

![image-20260312203620460](C:\Users\86198\AppData\Roaming\Typora\typora-user-images\image-20260312203620460.png)

![image-20260312203643942](C:\Users\86198\AppData\Roaming\Typora\typora-user-images\image-20260312203643942.png)

#### 总结

![image-20260312203721778](C:\Users\86198\AppData\Roaming\Typora\typora-user-images\image-20260312203721778.png)

### GQA

![image-20260312212224262](C:\Users\86198\AppData\Roaming\Typora\typora-user-images\image-20260312212224262.png)

![image-20260314094132170](C:\Users\86198\AppData\Roaming\Typora\typora-user-images\image-20260314094132170.png)

## 相关torch的api

### torch.arrange()：用于创建一维tensor

其值是一个等差数列 

```python
# 从 0 开始，到 10 结束（不含），步长为 2
>>> torch.arange(0, 10, 2)
tensor([0, 2, 4, 6, 8])

# 也可以使用浮点数
>>> torch.arange(0.0, 1.0, 0.25)
tensor([0.0000, 0.2500, 0.5000, 0.7500])

# 步长可以是负数（递减）
>>> torch.arange(5, 0, -1)
tensor([5, 4, 3, 2, 1])
```

### torch.outer()：计算两个一维tensor的外积

- **输入：** 两个一维张量 `input` (长度 $N$) 和 `vec2` (长度 $M$)。
- **输出：** 一个二维张量（矩阵），形状为 $(N, M)$。
- **计算：** 输出矩阵的 `[i, j]` 元素等于 `input[i] * vec2[j]`。

```python
>>> v1 = torch.arange(1, 4)  # tensor([1, 2, 3])
>>> v2 = torch.arange(1, 3)  # tensor([1, 2])

# v1 的每个元素 乘以 v2 的每个元素
>>> torch.outer(v1, v2)
tensor([[1, 2],  # 1 * [1, 2]
        [2, 4],  # 2 * [1, 2]
        [3, 6]]) # 3 * [1, 2]
```

### torch.where()：

- **功能：** 这是一个三元运算符，类似 `if/else`，但是是元素级别的。
- **签名：** `torch.where(condition, x, y)`
- 输入：
  - `condition`：一个布尔型张量（包含 `True` 和 `False`）。
  - `x`：当条件为 `True` 时，从中取值的张量。
  - `y`：当条件为 `False` 时，从中取值的张量。
- **输出：** 一个新的张量，其元素根据 `condition` 从 `x` 或 `y` 中选择。

```python
>>> x = torch.tensor([1, 2, 3, 4, 5])
>>> y = torch.tensor([10, 20, 30, 40, 50])

# 条件：元素是否大于 3
>>> condition = (x > 3)  # tensor([False, False, False,  True,  True])

# 大于 3 的用 x (即 [4, 5])，否则用 y (即 [10, 20, 30])
>>> torch.where(condition, x, y)
tensor([10, 20, 30,  4,  5])
```

### torch.cat()：沿一个指定的维度dim将多个tensor拼接在一起

- **签名：** `torch.cat(tensors, dim=0)`
- 输入：
  - `tensors`：一个包含多个张量的列表或元组。
  - `dim`：要拼接的维度（轴）。
- **要求：** 除了 `dim` 维度之外，所有其他维度的大小必须完全相同。

```python
>>> t1 = torch.tensor([[1, 2], 
...                    [3, 4]])
>>> t2 = torch.tensor([[5, 6], 
...                    [7, 8]])

# dim=0 (沿着行拼接，即垂直堆叠)
>>> torch.cat([t1, t2], dim=0)
tensor([[1, 2],
        [3, 4],
        [5, 6],
        [7, 8]])
# 形状: (2, 2) 和 (2, 2) -> (4, 2)

# dim=1 (沿着列拼接，即水平堆叠)
>>> torch.cat([t1, t2], dim=1)
tensor([[1, 2, 5, 6],
        [3, 4, 7, 8]])
# 形状: (2, 2) 和 (2, 2) -> (2, 4)
```

### tensor.unsqueeze()：用于增加维度

它的功能非常简单：**在张量的指定位置插入一个大小为 1 的新维度**。

```
output_tensor = tensor.unsqueeze(dim)
```

- `tensor`：你的输入张量。
- `dim` (int)：你想要**插入**新维度的位置（索引）。这个索引的范围可以在 `[-input.dim() - 1, input.dim()]` 之间。

### torch.clamp()：

* 声明：torch.clamp(input, min=None, max=None, *, out=None) → Tensor

* 功能：将 input 中的所有元素限制在范围 [ min, max ] 内。

  * 令 min_value 和 max_value 分别为 min 和 max，则返回yi=min⁡(max⁡(xi,min_valuei),max_valuei)yi=min(max(xi,min_valuei),max_valuei)

  * 如果 min 为 None，则没有下界。或者，如果 max 为 None，则没有上界。

    --注意--：如果 min 大于 max，则 torch.clamp(..., min, max) 将 input 中的所有元素设置为 max 的值。

* 参数:
  * input (Tensor) – 输入张量。
  * min (Number 或 Tensor, 可选) – 要限制到的范围的下界
  * max (Number 或 Tensor, 可选) – 要限制到的范围的上界
* 关键字参数:
  * out (Tensor, optional) – 输出张量。

```python
a = torch.randn(4)
a
torch.clamp(a, min=-0.5, max=0.5)

min = torch.linspace(-1, 1, steps=4)
torch.clamp(a, min=min)
```

### tensor.view()

**在不改变数据内容的前提下，改变张量的“视图”或“维度”**。

**1. 核心功能：重塑形状**

`view()` 允许你指定一个新的形状（维度），PyTorch 会将原始张量的数据“重新填充”到这个新形状中。

**关键要求：** 新形状的总元素数量必须与原始形状的总元素数量**完全相同**。

```python
# 原始张量：2x6，总共 12 个元素
t = torch.tensor([[ 1,  2,  3,  4,  5,  6],
                  [ 7,  8,  9, 10, 11, 12]])
# shape: (2, 6)

# 1. 重塑为 (3, 4) (总共 12 个元素)
t_view1 = t.view(3, 4)
print(t_view1)
# tensor([[ 1,  2,  3,  4],
#         [ 5,  6,  7,  8],
#         [ 9, 10, 11, 12]])

# 2. 重塑为 (4, 3) (总共 12 个元素)
t_view2 = t.view(4, 3)
print(t_view2)
# tensor([[ 1,  2,  3],
#         [ 4,  5,  6],
#         [ 7,  8,  9],
#         [10, 11, 12]])

# 3. 错误：重塑为 (3, 5) (总共 15 个元素)
# t.view(3, 5) 
# # 会报错: RuntimeError: shape '[3, 5]' is invalid for input of size 12
```

### tensor.tranpose()

用于**交换 (swap)** 张量的**两个**指定维度。

```python
output_tensor = tensor.transpose(dim0, dim1)
```

- `tensor`：你的输入张量。
- `dim0` (int)：你想要交换的第一个维度的索引。
- `dim1` (int)：你想要交换的第二个维度的索引。

### nn.Parameter

它本质上是一个 `Tensor`，但它被 `nn.Parameter` 包装（wrap）后，就有了特殊的含义：

> "我是一个模型的可训练参数（权重或偏置）。"

当你将一个 `nn.Parameter` 赋值给一个 `nn.Module` 的属性时（例如 `self.weight = nn.Parameter(...)`），`nn.Module` 就会自动“注册”它。

`nn.Parameter` 与普通 `Tensor` 的关键区别：

1. **自动注册：** `nn.Parameter` 会被 `nn.Module` 自动添加到 `.parameters()` 列表中。普通的 `Tensor` 不会。
2. **`requires_grad` 默认为 `True`：** 因为参数天生就是需要被训练的，所以它们默认需要计算梯度。

### nn.module

`nn.Module` 是 PyTorch 中所有神经网络层和模型的**基类（Base Class）**。

当你构建一个模型（例如 `MyModel`）或一个层（例如 `nn.Linear`, `nn.Conv2d`，或者你自己写的 `FeedForward` 层）时，你**必须**继承 `nn.Module`。

`nn.Module` 扮演着一个**容器**的角色，它主要帮你做两件大事：

1. 自动追踪参数 (Parameters)：

   - 它会自动“注册”所有在 `__init__` 中被定义为 `nn.Parameter` 或**子模块**（其他 `nn.Module`）的属性。
   - 这就是为什么你可以简单地调用 `model.parameters()` 就能获取到模型**所有**的可训练参数（包括所有嵌套子模块的参数），并把它们交给优化器 `torch.optim.Adam(model.parameters())`

2. 提供核心功能：

   - 它提供了一系列标准方法，如 `.to(device)`（将模型所有参数移动到 GPU/CPU）、`.train()`（设置为训练模式，激活 Dropout/BN）和 `.eval()`（设置为评估模式，关闭 Dropout/BN）。

   - 它要求你必须实现 `forward()` 方法，这定义了数据（`x`）如何流经你的模块。

### nn.Linear

唯一作用就是对输入数据 $x$ 执行一次线性变换，即应用一个“权重矩阵” (A) 和一个 “偏置” (b)

```python
import torch
import torch.nn as nn

# 创建一个 Linear 层
# in_features: 输入特征的数量
# out_features: 输出特征的数量
layer = nn.Linear(in_features=10, out_features=5, bias=True)

# -----------------
# 参数说明：
#
# in_features (int): 
#   输入张量最后一个维度的大小。
#   例如，如果你的输入数据是 (BatchSize, 10)，这里就是 10。
#
# out_features (int): 
#   你希望输出张量最后一个维度的大小。
#   例如，你希望把 10 个特征压缩成 5 个，这里就是 5。
#
# bias (bool, 可选): 
#   默认为 True。
#   如果为 True，该层会自动创建并学习一个偏置向量 b。
#   如果为 False，公式就变为 y = x * A^T，没有 b。
```

### nn.Dropout

`nn.Dropout` 层在训练时，会**随机地**将输入张量中的一部分元素**置为零**，同时对剩余的元素进行**放大（scale up）**。

想象一下你有一个张量（比如一层的激活值）： `input = [1, 2, 3, 4, 5]`

当你创建一个 `nn.Dropout(p=0.5)`（即 50% 的丢弃率）并将其应用到输入上时：

**在 `model.train()` 模式下 (训练时):** 它可能会产生这样的输出（每次都不同）： `output = [0, 4, 0, 8, 0]`

- `1`, `3`, `5` 被随机选中并置为 `0`。
- `2`, `4` 被保留，但它们被**放大**了。
- 为什么放大？nn.Dropout会将剩余元素乘以 1 / (1 - p)
  - 在这里 `p=0.5`，所以 `1 / (1 - 0.5) = 2`。
  - `2 * 2 = 4`，`4 * 2 = 8`。
  - 这样做的目的是**保持层激活的总和（期望值）不变**，使得训练和测试时的激活尺度大致相同。

```python
import torch.nn as nn

# p: 丢弃概率（一个元素被置为 0 的概率）
# 推荐值通常在 0.2 到 0.5 之间
# p=0.2 意味着随机丢弃 20% 的神经元
dropout_layer = nn.Dropout(p=0.5)
```

