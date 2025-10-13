"""LoRA transformer models."""

from typing import Any, Optional
from pydantic import BaseModel


class BaseLoRATransformRequestOutput(BaseModel):
    """Output model for LoRA request transformation."""
    request: Optional[Any] = None
    raw_request: Optional[Any] = None  # TODO: fix issue with Request in pydantic model
    adapter_name: Optional[str] = None