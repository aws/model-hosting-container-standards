from enum import Enum

# LoRA Handler Types
class LoRAHandlerType(str, Enum):
    REGISTER_ADAPTER = "register_adapter"
    UNREGISTER_ADAPTER = "unregister_adapter"
    ADAPTER_HEADER_TO_BODY = "adapter_header_to_body"

# Supported Engine Names
class SupportedEngine(str, Enum):
    VLLM = "vllm"

# SageMaker API Headers for LoRA API
class SageMakerLoRAApiHeader(str, Enum):
    ADAPTER_IDENTIFIER = "X-Amzn-SageMaker-Adapter-Identifier"
    ADAPTER_ALIAS = "X-Amzn-SageMaker-Adapter-Alias"

# Common fields to access in LoRA requests (body, path parameters, etc.)
class RequestField(str, Enum):
    MODEL = "model"
    ADAPTER_NAME = "adapter_name"

# Response message formats
class ResponseMessage(str, Enum):
    ADAPTER_REGISTERED = "Adapter {alias} registered"
    ADAPTER_UNREGISTERED = "Adapter {alias} unregistered"