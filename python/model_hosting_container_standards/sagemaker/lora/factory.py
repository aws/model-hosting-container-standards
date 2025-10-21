import logging

from model_hosting_container_standards.common.transforms.base_factory import (
    create_transform_decorator,
)

from .transforms import resolve_lora_transform

logger = logging.getLogger(__name__)


def create_lora_transform_decorator(handler_type: str):
    return create_transform_decorator(handler_type, resolve_lora_transform)
