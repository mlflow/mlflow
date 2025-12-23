"""Configuration for MemAlign optimizer."""

from pydantic import BaseModel, Field


class MemAlignConfig(BaseModel):
    """Configuration for MemAlign optimizer with dual memory system."""

    model: str = Field(..., description="Model for guideline distillation and evaluation")
    disable_semantic_memory: bool = Field(
        default=False, description="If True, disable guideline distillation"
    )
    disable_episodic_memory: bool = Field(
        default=False, description="If True, disable example retrieval"
    )
    distill_prompt_template_name: str = Field(
        default="memalign_distill_guidelines.txt", description="Template for guideline extraction"
    )
    dedup_guidelines: bool = Field(default=True, description="If True, deduplicate guidelines")
    retrieval_k: int = Field(
        default=5, description="Number of examples to retrieve from episodic memory"
    )
    embedder_name: str = Field(
        default="openai/text-embedding-3-small",
        description="Embedding model name in LiteLLM format",
    )
    embed_dim: int = Field(default=512, description="Embedding dimension")
    temperature: float = Field(default=0.0, description="Temperature for LLM calls")
    max_output_tokens: int = Field(
        default=4096, description="Max tokens for guideline distillation"
    )
    max_input_tokens: int = Field(default=128000, description="Max input tokens for LLM calls")
    num_retries: int = Field(default=3, description="Number of retries on LLM failures")
    timeout: int = Field(default=60, description="Timeout in seconds for LLM queries")

    class Config:
        frozen = True
