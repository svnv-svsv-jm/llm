import pydantic
import pydantic_settings
from loguru import logger


class Settings(pydantic_settings.BaseSettings):
    """App's settings."""

    # Settings configuration
    model_config = pydantic_settings.SettingsConfigDict(
        env_prefix="SVSV_",
        case_sensitive=False,  # from the environment
        env_file_encoding="utf-8",
        env_file=".env",
        extra="ignore",
        env_nested_delimiter="_",
        validate_assignment=True,
        revalidate_instances="always",
        validate_default=True,
    )

    debug_mode: bool = pydantic.Field(False, description="Whether to run in debug mode or not.")
    log_level: str = pydantic.Field("INFO", description="Logging level.")
    default_response: str = pydantic.Field(
        "",
        description="Default response for the assistant.",
    )
    default_llm_id: str = pydantic.Field(
        "TinyLlama/TinyLlama_v1.1",
        description="Default LLM.",
    )
    default_llm_mlx_id: str = pydantic.Field(
        "mlx-community/Mistral-7B-Instruct-v0.3",
        description="Default LLM MLX.",
        examples=[
            "mlx-community/Mistral-7B-Instruct-v0.3",
            "mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit",
            "mlx-community/Mistral-Small-24B-Instruct-2501-4bit",
            "mlx-community/DeepSeek-R1-Distill-Llama-8B-4bit",
            "mlx-community/gemma-3-4b-pt-4bit",
        ],
    )

    @pydantic.computed_field  # type: ignore
    @property
    @logger.catch(ImportError, default=False)
    def has_mlx(self) -> bool:
        """Whether to use `mlx` or not."""
        # pylint: disable=import-error
        import mlx_lm

        logger.trace(f"Found MLX: {mlx_lm}")
        return True  # pragma: no cover

    @pydantic.computed_field  # type: ignore
    @property
    def default_llm(self) -> str:
        """Chooses the global general default LLM based on system."""
        return self.default_llm_mlx_id if self.has_mlx else self.default_llm_id


settings = Settings()
