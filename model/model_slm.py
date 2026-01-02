import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from transformers import PreTrainedModel, GenerationMixin, Cache, DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast

from model.configuration_slm import SLMConfig


# Mean Root Squre
class RSMNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.var_eps = eps
    
    def forward(self, x: torch.tensor):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.var_eps) # rsqrt: 平方根倒数
        return self.weight * x.to(input_dtype)


class RoPE(nn.Module):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: int = 10000,
        device: torch.device = None
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # 预先生成并缓存 后续不够自动扩容
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=device, dtype=torch.float32)
    
    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=dtype)
        
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but consistent with transformers Llama implementation
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # 注册为 buffer，形状为 [max_len, dim]
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)
    
    @torch.no_grad()
    def forward(self, x: torch.tensor, seq_len: int = None):
        # x.shape = (b, h, s, h_d)
        
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
            
        return (
            self.cos_cached[:self.max_seq_len_cached].to(dtype=x.dtype),
            self.sin_cached[:self.max_seq_len_cached].to(dtype=x.dtype)
        )

def rotate_half(x: torch.tensor) -> torch.tensor:
    # h1, h2 --> -h2, h1
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.tensor, k: torch.tensor, cos: torch.tensor, sin: torch.tensor, position_ids):
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# Attn
class Attention(nn.Module):
    def __init__(self, config: SLMConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
    
    def forward(self, x, attention_mask: torch.tensor=None, position_ids=None, past_key_value=None, rotary_emb=None):
        b, seq_len, _ = x.size()
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # (b, seq_len, num_heads, head_dim) -> (b, num_heads, seq_len, head_dim)
        q = q.view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # RoPE
        cos, sin = rotary_emb(v, seq_len=seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        current_k_for_cache = k
        current_v_for_cache = v
        
        # KV cache
        if past_key_value is not None and past_key_value[0] is not None and past_key_value[1] is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
        
        past_key_value = (current_k_for_cache, current_v_for_cache)

        # GQA: 如果KV头数小于Q头数 则需要将KV复制多次对齐
        if self.num_key_value_groups > 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)
        
        # PyTorch 2.0 版本以上才有 Flash Attention 优化
        q, k = q.to(v.dtype), k.to(v.dtype)
        if attention_mask is not None:
            attention_mask = attention_mask.to(v.dtype)
        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(b, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output, past_key_value

# SwiGLU
class MLP(nn.Module):
    def __init__(self, config: SLMConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()
    
    def forward(self, x: torch.tensor):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class MoE(nn.Module):
    def __init__(self, config: SLMConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        # Router
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        # 真实场景中会专门写一个算子
        self.experts = nn.ModuleList([MLP(config) for _ in range(self.num_experts)])
    
    def forward(self, x):
        b, s, hidden_dim = x.shape
        # -> (b * s, hidden_dim)
        x = x.view(-1, hidden_dim)
        router_logits = self.gate(x)

        # load balancing loss: 防止坍缩
        routing_probs = F.softmax(router_logits, dim=-1)
        
        # top-k: 每个token对选中专家的权重, 选中专家的索引
        routing_weights, selected_experts = torch.topk(routing_probs, self.num_experts_per_tok, dim=-1)
        # routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float32).to(x.dtype)

        # aux loss = alpha * N * sum(f_i * p_i), f_i: 专家i被选中的频率 p_i: 专家i被选中的平均概率
        # one-hot 统计有多少token选择了专家i
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).float()
        # (totle_tokens, top_k, num_experts) -> (num_experts)
        tokens_per_expert = expert_mask.sum(dim=0).sum(dim=0)
        fraction_of_tokens = tokens_per_expert / x.shape[0]

        # router给专家i的平均打分
        prob_per_expert = routing_probs.mean(dim=0)

        # 计算相关性: 均匀分布 就要最小化点积
        aux_loss = (fraction_of_tokens * prob_per_expert).sum() * self.num_experts

        results = torch.zeros_like(x)
        for k in range(self.num_experts_per_tok):
            expert_ids = selected_experts[:, k]
            weights = routing_weights[:, k].unsqueeze(-1)
            for expert_idx in range(self.num_experts):
                # 找到属于该expert的padding mask
                idx_mask = (expert_ids == expert_idx)
                if idx_mask.any():
                    input_chunk = x[idx_mask]
                    expert_output = self.experts[expert_idx](input_chunk)
                    results[idx_mask] += expert_output * weights[idx_mask]
        return results.view(b, s, hidden_dim), aux_loss # 记得把后面的一层一层传递加上


# Decoder Block
class SLMBlock(nn.Module):
    def __init__(self, config: SLMConfig):
        super().__init__()
        self.self_attn = Attention(config)

        if config.use_moe and config.num_experts > 1:
            self.mlp = MoE(config)
        else:
            self.mlp = MLP(config)
        
        self.pre_layernorm = RSMNorm(config.hidden_size, config.rms_norm_eps)
        self.post_layernorm = RSMNorm(config.hidden_size, config.rms_norm_eps)
    
    def forward(self, x: torch.tensor, attn_mask=None, position_ids=None, past_key_value=None, rotary_emb=None):
        # pre-norm -> attn -> residual
        residual = x
        x = self.pre_layernorm(x)
        
        attn_out, present_kv = self.self_attn(
            x, attention_mask=attn_mask, position_ids=position_ids, past_key_value=past_key_value, rotary_emb=rotary_emb
        )
        x = residual + attn_out

        # pre-norm -> mlp -> res
        residual = x
        x = self.post_layernorm(x)
        mlp_outputs = self.mlp(x)
        if isinstance(mlp_outputs, tuple):
            mlp_out = mlp_outputs[0]
            aux_loss = mlp_outputs[1]
        else:
            mlp_out = mlp_outputs
            aux_loss = 0.0
        x = residual + mlp_out

        return x, present_kv, aux_loss

class SLMPretrainedModel(PreTrainedModel):
    config_class = SLMConfig
    base_model_prefix = "model"
    _no_split_modules = ["SLMBlock"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

class SLMModel(SLMPretrainedModel):
    def __init__(self, config: SLMConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([SLMBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = RSMNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = RoPE(config.hidden_size // config.num_attention_heads, config.max_position_embeddings)

        self.post_init()
    
    def forward(self, input_ids, attn_mask=None, past_key_values=None, token_type_ids=None, **kwargs):
        b, seq_len = input_ids.shape
        device = input_ids.device

        # 如果用户传入的是元组 直接转换为DynamicCache
        use_legacy_cache = False
        if past_key_values is not None and not isinstance(past_key_values, Cache):
            use_legacy_cache = True

        # # 假设连续 实际上要考虑past_key_values的长度
        # past_length = 0 if past_key_values is None else past_key_values[0][0].shape[2]
        # position_ids = torch.arange(past_length, seq_len + past_length, dtype=torch.long, device=input_ids.device)

        # 缓存过去的长度 past_length = 0 if past_key_values is None else past_key_values[0][0].shape[2]
        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                past_length = past_key_values.get_seq_length()
            else:
                past_length = past_key_values[0][0].shape[2]
        
        if "position_ids" in kwargs and kwargs["position_ids"] is not None:
            position_ids = kwargs["position_ids"]
        else:
            position_ids = torch.arange(past_length, seq_len + past_length, dtype=torch.long, device=device)
            # 虽然会自动广播, 但保持形状一致更安全
            position_ids = position_ids.unsqueeze(0).view(-1, seq_len)

        # Embedding
        hidden_states = self.embed_tokens(input_ids)

        # DynamicCach通常在generate时自动处理mask?
        # 训练阶段 等价于 past_length == 0
        if seq_len > 1:
            attn_mask = self._prepare_decoder_attention_mask(
                attn_mask, (b, seq_len), hidden_states, past_length
            )

        # Layers
        next_decoder_cache = []
        all_router_losses = []
        for i, layer in enumerate(self.layers):
            # 新版本的transformers
            layer_past = None
            if past_key_values is not None:
                if isinstance(past_key_values, Cache):
                    if past_key_values.get_seq_length() > 0:
                        if i < len(past_key_values.layers):
                            layer_past = (past_key_values.layers[i].keys, past_key_values.layers[i].values)
                else:
                    layer_past = past_key_values[i]

            # past_kv = past_key_values[i] if past_key_values is not None else None
            layer_outputs = layer(
                hidden_states, attn_mask=attn_mask, position_ids=position_ids, past_key_value=layer_past, rotary_emb=self.rotary_emb
            )
            hidden_states, present_kv = layer_outputs[0], layer_outputs[1]

            if past_key_values is not None:
                if isinstance(past_key_values, Cache):
                    k_new, v_new = present_kv
                    past_key_values.update(k_new, v_new, i)
                else:
                    next_decoder_cache.append(present_kv)
            
            if len(layer_outputs) > 2:
                all_router_losses.append(layer_outputs[2])
            # next_decoder_cache.append(present_kv)
        
        hidden_states = self.norm(hidden_states)

        if past_key_values is not None and isinstance(past_key_values, Cache):
            next_decoder_cache = past_key_values
        
        return hidden_states, next_decoder_cache, all_router_losses
    
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_ksy_values_length):
        '''padding 和 上三角掩码'''
        # input_shape: (b, seq_len)
        combined_attention_mask = None
        # (1, 1, seq_len, seq_len)
        seq_len = input_shape[1]
        causal_mask = torch.full((seq_len, seq_len), torch.finfo(inputs_embeds.dtype).min, device=inputs_embeds.device)
        causal_mask = torch.triu(causal_mask, diagonal=1).unsqueeze(0).unsqueeze(0)
        # causal_mask = torch.triu(causal_mask, diagonal=1)[None, None, :, :]

        if attention_mask is not None:
            # (b, seq_len) -> (b, 1, 1, seq_len)
            expanded_attn_mask = attention_mask[:, None, None, :].expand(input_shape[0], 1, seq_len, seq_len)
            # TODO padding
            expanded_attn_mask = (1.0 - expanded_attn_mask) * torch.finfo(inputs_embeds.dtype).min
            combined_attention_mask = expanded_attn_mask + causal_mask
        else:
            combined_attention_mask = causal_mask
        
        return combined_attention_mask
    
# ForCausalLM
class SLMForCausalLM(SLMPretrainedModel, GenerationMixin):

    def __init__(self, config: SLMConfig):
        super().__init__(config)
        self.model = SLMModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Weight Tying: safetensors不支持参数共享
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        
        self.post_init()
    
    def forward(self, input_ids, attn_mask=None, labels=None, past_key_values=None, token_type_ids=None, **kwargs):
        hidden_states, past_kv, router_losses = self.model(input_ids, attn_mask, past_key_values)

        logits = self.lm_head(hidden_states)

        ce_loss = torch.tensor(0.0, device=logits.device)
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            ce_loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
        
        aux_loss = torch.tensor(0.0, device=logits.device)
        if len(router_losses) > 0:
            # aux_loss = torch.stack(router_losses).sum()
            aux_loss = torch.sum(torch.tensor(router_losses))
        
        moe_loss_weight = 0.01
        loss = ce_loss + moe_loss_weight * aux_loss
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_kv
        )
    
    def prepare_inputs_for_generation(
        self, 
        input_ids, 
        past_key_values=None, 
        attention_mask=None, 
        inputs_embeds=None, 
        **kwargs
    ):
        past_length = 0
        
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                # 新版Cache对象获取长度的方法
                past_length = past_key_values.get_seq_length()
            else:
                # 旧版元组获取长度的方法
                past_length = past_key_values[0][0].shape[2]
            
            # 只保留最新的token 因为generate循环中，input_ids包含了所有历史
            if input_ids.shape[1] > past_length:
                 input_ids = input_ids[:, past_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]

        model_inputs = {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            **kwargs 
        }
        
        return model_inputs