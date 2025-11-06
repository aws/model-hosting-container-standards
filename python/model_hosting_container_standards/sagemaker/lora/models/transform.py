"""LoRA transform models."""

from typing import Any, Literal, Optional

import jmespath
from pydantic import BaseModel, Field, model_validator

from ....common import BaseTransformRequestOutput


class AppendOperation(BaseModel):
    """Configuration for append operation in inject_adapter_id decorator.

    This model defines how to append an adapter ID from a JMESPath expression
    to an existing value in the request body.
    """

    operation: Literal["append"] = Field(
        default="append", description="Operation type (always 'append')"
    )
    separator: str = Field(
        description="Separator string to use when appending (e.g., ':', '-', or empty string '')"
    )
    expression: str = Field(
        description="JMESPath expression string to extract the adapter ID (e.g., 'headers.\"x-adapter-id\"')"
    )
    compiled_expression: Any = Field(
        default=None,
        description="Compiled JMESPath expression (auto-compiled from expression)",
    )

    @model_validator(mode="after")
    def compile_jmespath_expression(self):
        """Compile the JMESPath expression from the expression field."""
        if self.compiled_expression is None and self.expression:
            self.compiled_expression = jmespath.compile(self.expression)
        return self


class BaseLoRATransformRequestOutput(BaseTransformRequestOutput):
    """Output model for LoRA request transformation."""

    adapter_name: Optional[str] = None
