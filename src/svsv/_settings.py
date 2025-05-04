import pydantic_settings
import pydantic


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


settings = Settings()
