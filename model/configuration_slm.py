from transformers import PretrainedConfig

class SLMConfig(PretrainedConfig):
    model_type = 'slm'
    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 512,
        intermediate_size: int = 1408, # 8 / 3 * hidden_size
        num_hidden_layers: int = 8,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 4,
        hidden_act: str = 'silu',
        max_position_embeddings: int = 2048,
        initializer_range: float = 0.02, # 权重初始化方差
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        tie_word_embeddings = False, # 是否共享输入输出权重
        use_moe: bool = False,
        num_experts: int = 4,
        num_experts_per_tok: int = 2, # 每次激活几个
        pad_token_id: int = None,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.pad_token_id = pad_token_id

        if num_key_value_heads is None:
            # 标准MHA
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings

        if self.hidden_size % self.num_attention_heads != 0 or self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) % num_attention_heads ({self.num_attention_heads}) != 0 or"
                f"num_attention_heads ({self.num_attention_heads}) % num_key_value_heads ({self.num_key_value_heads}) != 0"
                f"\nPlease CHECK THEM!"
            )

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs
        )