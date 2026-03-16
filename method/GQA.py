import torch
import torch.nn as nn

#Dropout是为防止过拟合而引入的一种正则化技术，
# 在训练过程中随机将一部分神经元的输出设置为0，
# 以减少模型对特定神经元的依赖，从而提高模型的泛化能力。
# dropout_layer = nn.Dropout(p=0.5)

# t1 = torch.Tensor([1,2,3])
# t2 = dropout_layer(t1)
# print(t2)

#函数Linear的作用是实现一个线性变换，即对输入数据进行加权求和并添加偏置项
# layer = nn.Linear(in_features = 3, out_features = 5, bias = True)
# t1 = torch.Tensor([1, 2, 3])
# print(t1.shape)
# t2 = torch.Tensor([[1, 2, 3]])
# print(t2.shape)
# output = layer(t2)
# print(output)

# t = torch.Tensor([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])
#函数view的作用是改变张量的形状，但不改变其数据内容。它返回一个新的张量，具有指定的形状。
#参数shape指定了新的形状，可以是一个整数或一个整数的元组。例如，t.view(3, 4)将张量t重新形状为3行4列。
# t1_view = t.view(3, 4)
# print(t1_view, t1_view.shape)
# t2_view = t.view(4, 3)
# print(t2_view, t2_view.shape)

# t = torch.Tensor([[1, 2, 3], [4, 5, 6]])
#函数transpose的作用是交换张量的维度。它返回一个新的张量，其维度按照指定的顺序进行重新排列。
#参数dim0和dim1指定了要交换的两个维度的索引。例如，t.transpose(0, 1)将交换第0维和第1维。
# print(t.transpose(0, 1))

# t = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# #函数triu的作用是返回一个上三角矩阵，即将输入张量中位于主对角线以下的元素设置为0。
# print(torch.triu(t))
# print(torch.triu(t, diagonal=1))  # diagonal=1表示主对角线以上的元素设置为0


x = torch.arange(1, 7)
print(torch.reshape(x, (2, 3)))
#参数-1表示自动推断该维度的大小，使得总元素数量保持不变。
print(torch.reshape(x, (3, -1)))
