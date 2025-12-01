from typing import List

from ....logging_config import logger
from ..constants import LoRAHandlerType
from .inject_to_body import InjectToBodyApiTransform
from .register import RegisterLoRAApiTransform
from .unregister import UnregisterLoRAApiTransform


def resolve_lora_transform(handler_type: str) -> type:
    """Get the appropriate transformer class based on the LoRA handler type.

    Maps handler type strings to their corresponding transformer classes that handle
    request/response transformations for different LoRA operations.

    :param str handler_type: The LoRA handler type (e.g., 'register_adapter', 'unregister_adapter')
    :return type: The transformer class corresponding to the handler type
    :raises ValueError: If the handler_type is not supported
    """
    logger.debug(f"Resolving transformer class for handler type: {handler_type}")
    match handler_type:
        case LoRAHandlerType.REGISTER_ADAPTER:
            logger.debug("Resolved to RegisterLoRAApiTransform")
            return RegisterLoRAApiTransform
        case LoRAHandlerType.UNREGISTER_ADAPTER:
            logger.debug("Resolved to UnregisterLoRAApiTransform")
            return UnregisterLoRAApiTransform
        case LoRAHandlerType.INJECT_ADAPTER_ID:
            logger.debug("Resolved to InjectToBodyApiTransform")
            return InjectToBodyApiTransform
        case _:
            logger.error(f"Unsupported LoRAHandlerType: {handler_type}")
            raise ValueError(f"Unsupported LoRAHandlerType: {handler_type}")


__all__: List[str] = [
    "InjectToBodyApiTransform",
    "RegisterLoRAApiTransform",
    "UnregisterLoRAApiTransform",
    "resolve_lora_transform",
]
