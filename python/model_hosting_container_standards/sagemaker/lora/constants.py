from enum import Enum


# LoRA Handler Types
class LoRAHandlerType(str, Enum):
    REGISTER_ADAPTER = "register_adapter"
    UNREGISTER_ADAPTER = "unregister_adapter"
    INJECT_ADAPTER_ID = "inject_adapter_id"


# SageMaker API Headers for LoRA API
class SageMakerLoRAApiHeader(str, Enum):
    ADAPTER_IDENTIFIER = "X-Amzn-SageMaker-Adapter-Identifier"
    ADAPTER_ALIAS = "X-Amzn-SageMaker-Adapter-Alias"


# Common fields to access in LoRA requests (body, path parameters, etc.)
class RequestField(str, Enum):
    ADAPTER_NAME = "adapter_name"


# Response message formats
class ResponseMessage:
    ADAPTER_REGISTERED = "Adapter {alias} registered"
    ADAPTER_UNREGISTERED = "Adapter {alias} unregistered"
    ADAPTER_UPDATED = "Adapter {alias} updated"

    # Errors
    ADAPTER_NOT_FOUND = "The adapter {alias} was not found"
    ADAPTER_INVALID_WEIGHTS = "doesn't contain tensors"
    ADAPTER_MAX_LORA_RANK = "greater than max_lora_rank"
    # ADAPTER_ALREADY_EXISTS = "The adapter {alias} already exists. If you want to replace it, please unregister before registering a new adapter with the same name."
