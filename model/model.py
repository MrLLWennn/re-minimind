from transformers import PretrainedConfig


class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"

    def __init__(
            self,
            dropout: float = 0.0,
            bos_token_id: int = 1,
            eos_token_id: int = 2,
            hidden_act: str = 'silu',
            hidden_size: int = 512,
            intermediate_size: int = None,
            max_position_embeddings: int = 32768,
            num_attention_heads: int = 8,
            num_hidden_layers: int = 8,
            num_key_value_heads: int = 2,
            vocab_size: int = 6400,
            rms_norm_eps: float = 1e-05,
            rope_theta: int = 1000000.0,
            inference_rope_scaling: bool = False,
            flash_attn: bool = True,
            ####################################################
            # Here are the specific configurations of MOE
            # When use_moe is false, the following is invalid
            ####################################################
            use_moe: bool = False,
            num_experts_per_tok: int = 2,
            n_routed_experts: int = 4,
            n_shared_experts: int = 1,
            scoring_func: str = 'softmax',
            aux_loss_alpha: float = 0.01,
            seq_aux: bool = True,
            norm_topk_prob: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        # 外推长度 = factor * original_max_position_embeddings = 32768
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        self.flash_attn = flash_attn
        ####################################################
        # Here are the specific configurations of MOE
        # When use_moe is false, the following is invalid
        ####################################################
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok  # 每个token选择的专家数量
        self.n_routed_experts = n_routed_experts  # 总的专家数量
        self.n_shared_experts = n_shared_experts  # 共享专家
        self.scoring_func = scoring_func  # 评分函数，默认为'softmax'
        self.aux_loss_alpha = aux_loss_alpha  # 辅助损失的alpha参数
        self.seq_aux = seq_aux  # 是否在序列级别上计算辅助损失
        self.norm_topk_prob = norm_topk_prob  # 是否标准化top-k概率


# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Model
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

import math
import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from typing import Optional, Tuple, List, Union
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

#继承pytorch.nn.Module类
class RMSNorm(torch.nn.Module):
    #__init__初始化
    def __init__(self, dim:int, eps:float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    #_norm函数
    def _norm(self, x):
        return torch.rsqrt(x.pow(2).mean(-1, keepdim = True) + self.eps)
    
    #forward函数
    def forward(self,x):
        return x * self._norm(x.float()).type_as(x)
    

#函数：预计算旋转位置编码（RoPE），并实现了YaRN（NTK-aware插值）
#核心作用：其核心作用是为Transformer模型的注意力机制生成位置编码所需的余弦（cos）和正弦（sin）值，
#         以支持远超模型原始预训练长度的序列。
def precompute_freqs_cis(dim : int, end : int = int(32 * 1924), rope_base : float = 1e6, 
                         rope_scaling : Optional[dict] = None):
    # 1. 初始化标准 RoPE 频率。
    # torch.arange(0, dim, 2) 生成 [0, 2, 4, ... dim-2]
    # 计算出的 freqs 就是标准的 1 / (base ** (2i / d))
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
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
            rope_scaling.get("attention_factor", 1.0)
        )

        # 只有当要推断的长度大于原始训练长度时，才应用缩放YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
        if end / orig_max > 1.0:
            
            # 3. 使用前文推导的公式，定义波长比例 b 到维度索引 i 的映射函数
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            # 4. 计算高频区和低频区的维度切分点
            # low: 不需要缩放的高频部分的最高索引
            # high: 需要完全缩放的低频部分的最低索引
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            
            # 5. 计算混合因子 γ (Ramp)
            # 在 low 之前，ramp 为 0；在 high 之后，ramp 为 1；在 low 和 high 之间，线性过渡。
            # clamp 函数限制了数值只能在 [0, 1] 之间。
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            
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


#函数：应用旋转位置编码（RoPE）
def apply_rotary_pos_emb(q, k, cos, sin, position_ids = None, unsqueeze_dim = 1):
    #[a, b] -> [-b, a]
    def rotate_half(x):
        # x.shape[-1]取最后一个维度的大小，即特征维度d
        # x[..., : x.shape[-1] // 2]取前半部分特征，x[..., x.shape[-1] // 2:]取后半部分特征
        return torch.cat(
            (-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1
        )
    
    # x_rotated = x * cos + rotate_half(x) * sin
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed


#函数repeat_kv的作用是重复key-value张量以匹配query头数，适用于分组查询注意力（GQA）机制。
#参数说明：x是输入的key-value张量，形状为[batch, seq_len, num_kv_heads, head_dim]；n_rep是重复次数，即每个kv头需要重复多少次来匹配query头数。
def repeat_kv(x : torch.Tensor, n_rep : int) -> torch.Tensor:
     
    """
    重复key-value张量以匹配query头数 (用于分组查询注意力GQA)
    等价于torch.repeat_interleave(x, dim=2, repeats=n_rep)，但更高效
    
    在GQA中，key和value的头数少于query，需要重复来匹配
    例如：8个query头，2个kv头，则需要每个kv头重复4次
    
    Args:
        x: kv张量 [batch, seq_len, num_kv_heads, head_dim]
        n_rep: 重复次数
    
    Returns:
        重复后的张量 [batch, seq_len, num_kv_heads * n_rep, head_dim]
    """
    bs, slen, num_key_value_heads, head_dim = x.shape#解包获取各维度
    if n_rep == 1:
        return x #无需重复直接返回
     
    # 高效的重复实现：
    # 1. x[:, :, :, None, :]: 在第4维插入新维度 -> [bs, slen, num_kv_heads, 1, head_dim]
    # 2. .expand(...): 扩展第4维到n_rep -> [bs, slen, num_kv_heads, n_rep, head_dim]
    # 3. .reshape(...): 合并第3、4维 -> [bs, slen, num_kv_heads * n_rep, head_dim]

    return (
        x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: MiniMindConfig):
        super().__init__()

        # 处理GQA：如果没有指定kv头数，则使用与query相同的头数
        # 三元运算符：condition ? value1 : value2
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads

        # assert语句：断言检查，如果条件为False则抛出AssertionError
        # 确保query头数能被kv头数整除（GQA的基本要求）
        assert args.num_attention_heads % self.num_key_value_heads == 0, "num_attention_heads must be divisible by num_key_value_heads for GQA"

        # 设置注意力头的配置
        self.n_loacl_heads = args.num_attention_heads   #query头数
        self.n_local_kv_heads = self.num_key_value_heads      #kv头数
        self.n_rep = self.n_loacl_heads // self.n_local_kv_heads  #每个kv头需要重复的次数
        self.head_dim = args.hidden_size // args.num_attention_heads  #每个头的维度

        # 定义线性投影层（无偏置，节省参数）
        # nn.Linear(in_features, out_features, bias = False)
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias = False)  #Query投影
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias = False)  #Key投影
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias = False)  #Value投影
        self.out_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias = False)  #输出投影

        #Dropout层用于正则化，防止过拟合
        self.attn_dropout = nn.Dropout(args.dropout)    # 注意力权重  dropout
        self.resid_dropout = nn.Dropout(args.dropout)   # 残差连接dropout
        self.dropout = args.dropout                     # 保存dropout概率以备后续使用

        # 检查是否支持Flash Attention
        # hasattr(obj, 'attr'): 检查对象是否有指定属性
        # Flash Attention需要PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn
        # 如果不支持可以打印警告: print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def forward(self,
                x : torch.Tensor,
                position_embeddings : Tuple[torch.Tensor, torch.Tensor], #修改为接收cos和sin
                past_key_value : Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache = False,
                attention_mask : Optional[torch.Tensor] = None):
        # x: [batch_size, seq_len, hidden]
        #投影，计算QKV
        #把输入拆分为多个头，并调整维度以适应注意力计算
        bsz, seq_len, _ = x.shape
        # 线性投影为Q,K,V
        # q_proj: hidden -> num_heads * head_dim
        # k_proj/v_proj: hidden -> num_kv_heads * head_dim (GQA情形)
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)  #投影得到QKV
        #用view
        xq = xq.view(bsz, seq_len, self.n_loacl_heads, self.head_dim)  #调整为[bsz, seq_len, num_heads, head_dim]
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)  #调整为[bsz, seq_len, num_kv_heads, head_dim]
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)  #调整为[bsz, seq_len, num_kv_heads, head_dim]
        #QK使用RoPE位置编码
        # position_embeddings是预计算的(cos, sin)，按序列位置切片并应用RoPE
        cos, sin = position_embeddings
        # 只取当前序列长度的前缀（用于inference时从start_pos开始）
        xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])

        # -------------------- KV cache 处理 --------------------
        # past_key_value: (past_k, past_v) 或 None
        # 当存在past时，将past拼接到当前k,v的时间维度上，便于自回归推理
        if past_key_value is not None:
            # past_key_value[0] 的shape为 [bsz, past_seq_len, n_local_kv_heads, head_dim]
            xk = torch.cat([past_key_value[0], xk], dim=1)  #在时间维度上拼接 -> [bsz, past_seq_len + seq_len, n_local_kv_heads, head_dim]
            xv = torch.cat([past_key_value[1], xv], dim=1)  #同上 -> [bsz, past_seq_len + seq_len, n_local_kv_heads, head_dim]
        # 如果需要缓存，返回拼接后的(k,v)，否则past_kv置为None
        past_kv = (xk, xv) if use_cache else None

        # -------------------- GQA: 对KV重复以匹配Q头 --------------------
        # transpose到形状 [bsz, n_heads, seq_len, head_dim] 以便矩阵乘法
        xq = xq.transpose(1, 2)  # [bsz, num_heads, seq_len, head_dim]
        #KV使用repeat_kv函数处理以适应GQA
        xk = repeat_kv(xk, self.n_rep).transpose(1, 2)  # [bsz, num_heads, seq_len, head_dim]
        xv = repeat_kv(xv, self.n_rep).transpose(1, 2)  # [bsz, num_heads, seq_len, head_dim]
        
        # -------------------- Attention计算 --------------------
        # 优先使用PyTorch 2.0+的scaled_dot_product_attention（Flash Attention实现）
        if self.flash and (seq_len > 1) and (attention_mask is None or torch.all(attention_mask == 1)):
            attn_mask = (
                None
                if attention_mask is None
                else attention_mask.view(bsz, 1, 1, -1).expand(bsz, self.n_loacl_heads, seq_len, -1).bool()
            )
            output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask = attn_mask, 
                                                    dropout_p = self.dropout if self.training else 0.0, is_causal = True)
        else:
            #注意力score计算：QK^T / sqrt(d_k)
            scores = (xq @ xk.transpose(-2, -1)) // math.sqrt(self.head_dim)
            #每次训练时需要遮住后面的token，防止信息泄露（causal mask），在矩阵中将下半部分去掉，设置为-inf，使得softmax后为0
            scores = scores + torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device = scores.device), diagonal = 1 #上三角
            ).unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]

            # 如果有attention_mask(0/1)，将其扩展后转为 -1e9 的加性mask（掩掉pad位置）
            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [bsz, 1, 1, seq_len]
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9  # 0 -> 0, 1 -> -1e9
                scores = scores + extended_attention_mask  # 加到score上，掩掉pad位置

            # softmax得到注意力权重
            scores = F.softmax(scores.float(), dim = -1).type_as(scores) #先通过float()提升精度，softmax后再转换回原数据类型
            scores = self.attn_dropout(scores)  # 在minimind Transformer layer的GQA部分没有体现的操作，是为了进一步防止过拟合，在训练时随机丢弃部分注意力连接，增强模型的泛化能力

            output = scores @ xv  # 注意力权重与V相乘得到输出 [bsz, num_heads, seq_len, head_dim]
    
        #这一步的目的是将多头注意力的输出重新拼接回原来的维度，以便后续的线性变换和残差连接。
        #由于之前的transpose是[bsz, num_heads, seq_len, head_dim],现在的维度变回[bsz, seq_len, num_heads * head_dim]
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        #resid_dropout原理是ResNet中的残差连接后面加一个dropout层，防止过拟合，增强模型的泛化能力
        output = self.resid_dropout(self.out_proj(output))  #输出投影并应用残差dropout
        #最后将输出和新的past_kv返回，供下一步Transformer层使用
        return output, past_kv
    
#Transformer layer经过GQA后下一步是前馈网络（FeedForward Network，FFN），也称为MLP层。它通常由两层线性变换和一个非线性激活函数组成，负责对每个位置的表示进行独立的变换和增强。
class FeedForward(nn.Module):
    #初始化
    def __init__(self, config : MiniMindConfig):
        super().__init__()
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3) #实践证明该倍数在性能和效率之间有较好平衡
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)

        # SwiGLU类似于Gated Linear Unit变体：act(gate(x)) * up(x)
        # gate_proj: hidden -> intermediate (用于计算gate部分)
        # up_proj: hidden -> intermediate (用于被gate的部分)
        # down_proj: intermediate -> hidden (用于投影回hidden维度)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias = False)   #gate线性层:gate层负责生成一个门控信号，控制信息流动
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias = False)  #降维线性层
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias = False)    #升维线性层
        
        self.dropout = nn.Dropout(config.dropout)  #前馈网络的dropout
        self.act_fn = ACT2FN[config.hidden_act]  #激活函数

    #前馈网络的计算过程：首先通过gate_proj和up_proj分别计算gate部分和被gate部分，然后将gate部分通过激活函数处理后与被gate部分逐元素相乘，
    # 最后通过down_proj投影回hidden维度，并应用dropout。
    # SwiGLU的核心思想是通过gate部分动态调整被gate部分的信息流动，使得模型能够更灵活地捕捉输入特征之间的复杂关系，从而提升模型的表达能力。
    # gate操作使得模型能够根据输入的特征动态调整信息流动，增强了模型的非线性表达能力和适应性，从而提升了模型在各种任务上的性能。
    def forward(self, x):
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))  #SwiGLU计算：act(gate(x)) * up(x)，再通过down_proj投影回hidden维度，并应用dropout

#
class MoEGate(nn.Module):  #门控矩阵Wg∈R(N×d)，N=专家数，d=隐藏维度
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok #带噪声的 Top-K 门控，引入了一些可调整的噪声，然后保留前 k 个值
        self.n_routed_experts = config.n_routed_experts #路由专家总数

        self.scoring_func = config.scoring_func # 评分函数类型
        self.alpha = config.aux_loss_alpha  # 评分函数类型
        self.seq_aux = config.seq_aux   # 是否序列级辅助损失

        self.norm_topk_prob = config.norm_topk_prob # 是否归一化top-k概率
        self.gating_dim = config.hidden_size    # 门控维度=隐藏层维度
        # 门控权重矩阵: [n_routed_experts, hidden_size]
        # 每个专家有一个对应的权重向量
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None: #使用Kaiming均匀初始化，适合ReLU类激活函数，避免梯度消失/爆炸。
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)   ## 展平: [bsz*seq_len, h]
        # 计算专家分数: [bsz*seq_len, n_routed_experts]
        # 线性变换: score = x * W_g^T
        logits = F.linear(hidden_states, self.weight, None)
        # Softmax归一化得到专家概率分布
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')
        
        # 选择top-k个专家: [bsz*seq_len, top_k]
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        # 可选: 归一化top-k权重，使其和为1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        """辅助损失计算（负载均衡）
        * 核心问题：如果某些专家总是被选中，某些从不被选中，会导致：
        * 热点专家过拟合
        * 冷点专家不更新
        * 计算资源浪费
        辅助损失目标：鼓励均匀使用所有专家。"""
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            #序列级辅助损失（seq_aux=True）
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                    seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = scores.new_zeros(1).squeeze()
        return topk_idx, topk_weight, aux_loss

#
class MOEFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        # 路由专家: 每个是独立的FeedForward网络
        self.experts = nn.ModuleList([
            FeedForward(config)
            for _ in range(config.n_routed_experts)
        ])
        # 门控网络
        self.gate = MoEGate(config)
        # 共享专家: 所有token都经过，不稀疏
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                FeedForward(config)
                for _ in range(config.n_shared_experts)
            ])

    def forward(self, x):
        identity = x 
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape
        # 使用门控机制选择专家
        topk_idx, topk_weight, aux_loss = self.gate(x)
        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            y = torch.empty_like(x, dtype=x.dtype)
            for i, expert in enumerate(self.experts):
                expert_out = expert(x[flat_topk_idx == i])
                if expert_out.shape[0] > 0: y[flat_topk_idx == i] = expert_out.to(y.dtype)
                else: y[flat_topk_idx == i] = expert_out.to(y.dtype) + 0 * sum(p.sum() for p in expert.parameters())
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
        else:
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)
        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.config.num_experts_per_tok
        # 当tokens_per_expert = [6, 15, 20, 26]，tokens_per_expert.shape[0]即为专家数量（此时为4）
        # 且token_idxs = [3, 7, 19, 21, 24, 25,  4,  5,  6, 10, 11, 12...] 时
        # 意味token_idxs[:6] -> [3, 7, 19, 21, 24, 25]这6个位置属于专家0处理的token（每个token有可能被多个专家处理，这取决于num_experts_per_tok）
        # 接下来9个位置token_idxs[6:15] -> [4,  5,  6, 10, 11, 12...]属于专家1处理的token...依此类推
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache

#接下来是将之前已经实现的 Transformer 的GQA和FFN模块组合成一个完整的Transformer层（TransformerLayer）。
# 每个Transformer层包含一个注意力子层和一个前馈网络子层，并且在它们之间有残差连接和层归一化。
class MiniMindBlock(nn.Module):
    def __init__(self, layer_id : int, config : MiniMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads #注意力头数
        self.hidden_size = config.hidden_size #隐藏层维度
        self.head_dim = config.hidden_size // config.num_attention_heads #每个头的维度
        self.self_attn = Attention(config) #注意力子层

        self.layer_id = layer_id #层ID，用于区分不同层
        self.input_layernorm = RMSNorm(config.hidden_size, eps = config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps = config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value = None, use_cache = False, attention_mask = None):
        residual = hidden_states #保存输入作为残差连接
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        ) #将Attention forward所需要的参数都传入，经过Attention计算得到新的hidden_states和present_key_value
        hidden_states = hidden_states + residual #注意力子层的残差连接
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states)) #前馈网络子层，包含残差连接
    
        return hidden_states, present_key_value
    
# 最后是将多个Transformer层堆叠起来形成完整的MiniMind模型（MiniMindModel）。这个模型包含一个输入嵌入层、多个Transformer层以及一个输出投影层。
class MiniMindModel(nn.Module):
    def __init__(self, config : MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers #Tokenizer Encoding词表大小和Transformer层数
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size) #Input Embedding初始化输入嵌入层
        self.dropout = nn.Dropout(config.dropout) #输入嵌入的dropout
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)]) #Transformer layer  # noqa: E741
        self.norm = RMSNorm(config.hidden_size, eps = config.rms_norm_eps) #RMSNorm

        #RoPE位置编码的预计算：根据配置计算出足够长度的cos和sin位置编码，以支持最长32768的序列输入
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim = config.hidden_size // config.num_attention_heads,
            end = config.max_position_embeddings, rope_base = config.rope_theta,
            rope_scaling = config.rope_scaling
        )
        #register_buffer: 将预计算的频率张量注册为模型的buffer，这样它们就会随着模型一起保存和加载，但不会被优化器更新
        self.register_buffer("freqs_cos", freqs_cos, persistent = False)
        self.register_buffer("freqs_sin", freqs_sin, persistent = False)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **kwargs): #kwargs是其他可能用到参数
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers') : past_key_values = None #兼容输入past_key_values为BaseModelOutputWithPast的情况  # noqa: E701
        past_key_values = past_key_values or [None] * len(self.layers) #如果没有提供past_key_values，则初始化为None列表，长度与Transformer层数相同
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0 #如果有past_key_values，计算当前输入的起始位置（即已经处理过的序列长度）

        # 预计算位置编码的切片：根据当前输入的起始位置和序列长度，从预计算的cos和sin中切出对应长度的部分，供Transformer层使用
        hidden_states = self.dropout(self.embed_tokens(input_ids))

        # 将切片后的位置编码作为参数传递给每个Transformer层，确保它们在计算注意力时能够正确地应用RoPE位置编码
        position_embeddings = (
            self.freqs_cos[start_pos : start_pos + seq_length],
            self.freqs_sin[start_pos : start_pos + seq_length]
        )

        # 逐层处理：对于每个Transformer层，传入当前的hidden_states、位置编码、对应的past_key_value以及其他参数，得到新的hidden_states和present_key_value
        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)): #zip函数将Transformer层和对应的past_key_value打包在一起，方便在循环中同时访问
            hidden_states, present = layer(     #解包
                hidden_states,
                position_embeddings,
                past_key_value = past_key_value,
                use_cache = use_cache,
                attention_mask = attention_mask
            )
            presents.append(present) #将每层的present_key_value保存到presents列表中，供后续使用（如生成时的缓存）

        hidden_states = self.norm(hidden_states) #最后一层的输出经过RMSNorm归一化

        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        return hidden_states, presents, aux_loss


# PreTrainedModel是Hugging Face Transformers库中一个基类，提供了模型权重加载、保存和其他实用功能。
# GenerationMixin是一个混入类，提供了文本生成相关的方法，如generate()，使得模型能够方便地进行文本生成任务。
# MiniMindForCasualLM的作用是将MiniMindModel封装成一个适用于语言模型训练和生成的接口，包含输入嵌入、Transformer层和输出投影，并实现了前向传播逻辑，包括损失计算和生成时的缓存机制。
class MiniMindForCasualLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig

    def __init__(self, config : MiniMindConfig = None):
        self.config = config or MiniMindConfig() #如果没有提供配置，则使用默认配置
        super().__init__(self.config)
        self.model = MiniMindModel(self.config) #实例化MiniMindModel作为核心模型
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias = False) #输出投影层，将hidden状态映射到词表大小的维度，生成每个token的预测分数
        self.model.embed_tokens.weight = self.lm_head.weight #权重共享：输入嵌入层和输出投影层共享权重矩阵，这样可以减少模型参数量，并且在某些情况下可以提升模型性能

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                **args):
        hidden_states, past_key_values, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args
        )
        # slice_indices的作用是根据logits_to_keep参数确定在计算损失时应该保留哪些token的logits。
        # logits_to_keep是整数，就保留最后n个token的logits
        # 生成的时候只需要保留最后一个token的logits来预测下一个token，因此默认值是0，表示只保留最后一个token的logits。
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep 
        #这一步计算的logits是模型对每个token的预测分数
        logits = self.lm_head(hidden_states[:, slice_indices, :])


        loss = None 
        #以下代码是计算交叉熵损失的标准做法，适用于语言模型的训练。它通过将模型输出的logits与标签进行对齐，计算每个位置的预测分数与真实标签之间的差异，从而得到整体的损失值。
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous() #将logits的最后一个时间步去掉，因为它没有对应的标签
            shift_labels = labels[..., 1:].contiguous() #将标签的第一个时间步去掉，因为它没有对应的logits
            #计算交叉熵损失，ignore_index=-100表示标签中值为-100的位置会被忽略，不参与损失计算
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100) 

        #最后将计算得到的loss、logits、past_key_values和hidden_states封装到CasualLMOutputWithPast对象中返回
        #CausalLMOutputWithPast是Hugging Face Transformers库中一个数据类，用于存储语言模型的输出结果，包括损失、预测分数、缓存的键值对和隐藏状态等信息，方便后续处理和分析。
        ouput = CausalLMOutputWithPast(loss = loss, logits = logits, past_key_values = past_key_values, hidden_states = hidden_states)
        ouput.aux_loss = aux_loss #如果有MOE的辅助损失，也返回

        return ouput