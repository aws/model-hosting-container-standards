"""LoRA transform models."""

from typing import Optional

from ....common import BaseTransformRequestOutput


class BaseLoRATransformRequestOutput(BaseTransformRequestOutput):
    """Output model for LoRA request transformation."""

    adapter_name: Optional[str] = None
