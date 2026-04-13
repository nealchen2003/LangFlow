"""HuggingFace configuration class for LangFlow."""

import transformers


class LangFlowConfig(transformers.PretrainedConfig):
    """HuggingFace configuration class for LangFlow.
    
    LangFlow is a continuous diffusion language model that operates in embedding space.
    It uses a DiT (Diffusion Transformer) backbone with adaptive layer normalization.
    
    Key features:
        - Continuous diffusion in embedding space
        - Self-conditioning: uses previous predictions as additional input
        - Bias (preconditioning): skip connection for improved training
        - Normalized embeddings: layernorm on embedding vectors
        - Learnable Gumbel proposal for gamma (log-SNR) sampling
    """
    model_type = "LangFlow"

    def __init__(
        self,
        vocab_size: int = 50257,
        hidden_size: int = 768,
        cond_dim: int = 128,
        n_blocks: int = 12,
        n_heads: int = 12,
        dropout: float = 0.1,
        model_length: int = 1024,
        # Embedding normalization
        use_normalized_embedding: bool = True,
        embedding_norm_method: str = "layernorm",
        # Self-conditioning
        self_conditioning: bool = True,
        # Bias (preconditioning) - always enabled for inference
        use_bias: bool = True,
        # Gumbel proposal parameters (learnable)
        gumbel_loc: float = 4.723,
        gumbel_scale: float = 0.852,
        gumbel_cutoff: float = 1e-5,
        gumbel_entropy: float = 7.02,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.cond_dim = cond_dim
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.dropout = dropout
        self.model_length = model_length
        # Embedding normalization
        self.use_normalized_embedding = use_normalized_embedding
        self.embedding_norm_method = embedding_norm_method
        # Self-conditioning
        self.self_conditioning = self_conditioning
        # Bias (preconditioning)
        self.use_bias = use_bias
        # Gumbel proposal parameters
        self.gumbel_loc = gumbel_loc
        self.gumbel_scale = gumbel_scale
        self.gumbel_cutoff = gumbel_cutoff
        self.gumbel_entropy = gumbel_entropy
