__all__ = ["Settings", "settings"]

import typing as ty
from pathlib import Path
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .const import (
    ENV_PREFIX,
    DEFAULT_UPLOADED_FILES_DIR,
    Q_SYSTEM_PROMPT,
    ZEPHYR_CHAT_TEMPLATE as CHAT_TEMPLATE,
)


class Settings(BaseSettings):
    """App settings.

    By default, the environment variable name is the same as the field name.

    You can change the prefix for all environment variables by setting the env_prefix config setting, or via the `_env_prefix` keyword argument on instantiation:

    ```python
    from pydantic_settings import BaseSettings, SettingsConfigDict

    class Settings(BaseSettings):
        model_config = SettingsConfigDict(env_prefix='my_prefix_')

        auth_key: str = 'xxx'  # will be read from `my_prefix_auth_key`
    ```

    If you want to change the environment variable name for a single field, you can use an alias.

    There are two ways to do this:

    * Using `Field(alias=...)` (see `api_key` above)
    * Using `Field(validation_alias=...)` (see `auth_key` above)

    `env_prefix` does not apply to fields with alias. It means the environment variable name is the same as field alias:

    ```python
    from pydantic import Field
    from pydantic_settings import BaseSettings, SettingsConfigDict

    class Settings(BaseSettings):
        model_config = SettingsConfigDict(env_prefix='my_prefix_')

        foo: str = Field('xxx', alias='FooAlias')
    ```
    """

    # Settings configuration
    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        case_sensitive=False,  # from the environment
        env_file_encoding="utf-8",
        env_file=".env",
        extra="ignore",
        env_nested_delimiter="_",
        validate_assignment=True,
        revalidate_instances="always",
        validate_default=True,
    )

    # Settings
    test_mode: bool = Field(
        False,
        description="If `True`, verbosity and strictness on errors are higher.",
    )
    uploaded_files_dir: str = Field(
        DEFAULT_UPLOADED_FILES_DIR,
        description="Location (in the file system) of the uploaded files.",
    )
    q_system_prompt: str = Field(
        Q_SYSTEM_PROMPT,
        description="Prompt for history aware retriever.",
    )
    has_chat: bool = Field(
        default=True,
        description="Whether the app has a chatbot or not.",
    )
    has_sidebar: bool = Field(
        default=True,
        description="Whether the app has a sidebar or not.",
    )
    app_name: str = Field(
        default="FiscalAI",
        description="App's name.",
    )
    app_title: str = Field(
        "💬 FiscalAI",
        description="App's title.",
    )
    app_subheader: str = Field(
        "Smart Assistant",
        description="App's subheader.",
    )
    app_caption: str = Field(
        "🚀 Your favorite chatbot, powered by FiscalAI.",
        description="App's subheader.",
    )
    start_message_en: str = Field(
        "How can I help you?",
        description="App's starting message (English).",
    )
    start_message_it: str = Field(
        "Come posso aiutarti?",
        description="App's starting message (Italian).",
    )
    verbose_item_set: bool = Field(
        False,
        description="Whether to log TRACE-level information every time session state's items are written.",
    )
    verbose_log_depth_item_set: int = Field(
        4,
        description="The value for the `dept` argument of the logger's `.opt()` method: `.opt(depth=depth)`. Valid only for logs concerning session state's items being written",
    )
    apply_chat_template: bool = Field(
        True,
        description="Whether to apply chat template to tokenizer. Unless `force_chat_template` is `True`, this is applied only if the tokenizer does not have a template already.",
    )
    chat_template: str | dict[str, ty.Any] | list[dict[str, ty.Any]] = Field(
        CHAT_TEMPLATE,
        description="Chat template to enforce when a default one is not available.",
    )
    force_chat_template: bool = Field(
        False,
        description="If `True`, the provided chat template will be forced on the tokenizer.",
    )
    pipeline_kwargs: dict = Field({}, description="Pipeline kwargs.")

    @field_validator("uploaded_files_dir", mode="before")
    @classmethod
    def _uploaded_files_dir(cls, value: str) -> str:
        """Make sure it's a `str` and create the location."""
        Path(value).mkdir(parents=True, exist_ok=True)
        return f"{value}"


settings = Settings()
