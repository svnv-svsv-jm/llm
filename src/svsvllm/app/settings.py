__all__ = ["Settings", "settings"]

import os
from pathlib import Path
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .const import (
    ENV_PREFIX,
    DEFAULT_UPLOADED_FILES_DIR,
    Q_SYSTEM_PROMPT,
    OPEN_SOURCE_MODELS_SUPPORTED,
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

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        case_sensitive=False,  # from the environment
    )

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
    open_source_models_supported: bool = Field(
        OPEN_SOURCE_MODELS_SUPPORTED,
        description="Whether open source models are supported.",
    )

    @classmethod
    @field_validator("uploaded_files_dir")
    def _uploaded_files_dir(cls, value: str) -> str:
        """Make sure it's a `str` and create the location."""
        value = f"{value}"
        Path(value).mkdir(parents=True, exist_ok=True)
        return value


settings = Settings()
