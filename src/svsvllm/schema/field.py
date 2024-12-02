__all__ = ["FieldExtraOptions"]

from pydantic import BaseModel, Field


class FieldExtraOptions(BaseModel):
    """Extra options to be passed in the `Field` at `json_schema_extra`:
    ```python
    Field(json_schema_extra=FieldExtraOptions().model_dump())
    ```
    """

    is_synced: bool = Field(
        True,
        description="Whether this attribute should be propagated to Streamlit's session state.",
    )
