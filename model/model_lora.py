import torch
from torch import optim, nn


"""
📚 LoRA 核心知识点：
- 什么是LoRA：一种参数高效微调方法，只训练少量新增参数
- 原理：在预训练模型的权重矩阵旁边添加低秩分解矩阵 ΔW = BA
  - 原始权重 W 保持冻结（requires_grad=False）
  - 新增两个小矩阵 A(d×r) 和 B(r×d)，其中 r<<d（秩远小于维度）
  - 前向计算：output = Wx + BAx
- 优势对比：
  - Full SFT：更新所有参数，效果好但需要大显存和长时间
  - LoRA：只更新1-5%的参数，显存需求小，训练快，适合资源受限场景
  - 多任务切换：可以保存多组LoRA权重，快速切换不同任务能力

📚 适用场景：
- 个性化定制：医疗、法律、金融等垂直领域适配
- 快速实验：尝试不同数据/超参时，LoRA训练速度快
- 资源受限：单卡或小显存环境
"""


# 定义Lora网络结构
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank  # LoRA的秩（rank），控制低秩矩阵的大小
        self.A = nn.Linear(in_features, rank, bias=False)  # 低秩矩阵A，将输入从in_features维降到rank维
        self.B = nn.Linear(rank, out_features, bias=False)  # 低秩矩阵B，将中间表示从rank维升到out_features维
        # 矩阵A高斯初始化：使用小标准差的高斯初始化（std=0.02），类似Transformer的参数初始化
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # 矩阵B全0初始化：关键设计！确保训练开始时ΔW = BA = 0，不影响原始模型
        self.B.weight.data.zero_()

    """ 输入 x ∈ ℝ^{batch×d}
            ↓ A矩阵（降维）
        中间表示 z = A(x) ∈ ℝ^{batch×r}
            ↓ B矩阵（升维）
        输出 y = B(z) ∈ ℝ^{batch×d'}

        等价于：y = BAx = ΔWx"""
    def forward(self, x):
        return self.B(self.A(x))

def apply_lora(model, rank=8):
    for name, module in model.named_modules():
        #isinstance(module, nn.Linear)：只对线性层应用LoRA
        #module.weight.shape[0] == module.weight.shape[1]：只对方阵线性层应用(保持ΔW与W形状相同，便于加法操作)
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank=rank).to(model.device)
            setattr(module, "lora", lora) #动态属性添加：将LoRA层作为属性附加到原模块
            original_forward = module.forward

            # 显式绑定(默认参数绑定)：默认参数在函数定义时求值并绑定，每个函数有自己的副本
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)

            module.forward = forward_with_lora
        """ 原始：output = Wx
            LoRA后：output = Wx + BAx
                        = (W + BA)x
                        = (W + ΔW)x"""
#加载LoRA权重
def load_lora(model, path):
    state_dict = torch.load(path, map_location=model.device)#自动处理设备不匹配问题
    """处理DataParallel包装：
    module.前缀：当使用nn.DataParallel或DistributedDataParallel包装时，参数名会添加module.前缀
    去掉前缀：k[7:]去掉前7个字符（module.）
    保持一致性：确保加载时能与模型参数名匹配"""
    state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}

    #权重分配：
    for name, module in model.named_modules():
        #查找所有带有lora属性的模块
        if hasattr(module, 'lora'):
            #从state_dict中提取该LoRA层的权重，键名模式：{模块名}.lora.{参数名}
            #将完整键名转换为LoRA层内部参数名
            lora_state = {k.replace(f'{name}.lora.', ''): v for k, v in state_dict.items() if f'{name}.lora.' in k}
            #load_state_dict加载到对应LoRA层
            module.lora.load_state_dict(lora_state)

#保存LoRA权重
def save_lora(model, path):
    raw_model = getattr(model, '_orig_mod', model) #获取原始模型
    state_dict = {}
    for name, module in raw_model.named_modules():
        if hasattr(module, 'lora'):
            #去掉可能的module.前缀
            clean_name = name[7:] if name.startswith("module.") else name
            #添加lora.前缀，形成完整参数名
            lora_state = {f'{clean_name}.lora.{k}': v for k, v in module.lora.state_dict().items()}
            #更新到总state_dict
            state_dict.update(lora_state)
    torch.save(state_dict, path)