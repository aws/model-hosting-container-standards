from typing import Dict, Optional

from ...common.fastapi.routing import RouteConfig
from .constants import LoRAHandlerType

# Route registry mapping LoRA handler types to their route configurations
# This dict serves as the single source of truth for LoRA API routes
LORA_ROUTE_REGISTRY: Dict[str, RouteConfig] = {
    # Register/load a new LoRA adapter
    LoRAHandlerType.REGISTER_ADAPTER: RouteConfig(
        path="/adapters",
        method="POST",
        tags=["adapters", "lora"],
        summary="Register a new LoRA adapter",
    ),
    # Unregister/unload an existing LoRA adapter
    LoRAHandlerType.UNREGISTER_ADAPTER: RouteConfig(
        path="/adapters/{adapter_name}",
        method="DELETE",
        tags=["adapters", "lora"],
        summary="Unregister a LoRA adapter",
    ),
    # Note: INJECT_ADAPTER_ID is intentionally omitted
    # It's a request transformer, not a standalone API endpoint
    # It modifies requests in-flight but doesn't expose its own route
}


def get_lora_route_config(handler_type: str) -> Optional[RouteConfig]:
    """Get the route configuration for a LoRA handler type.

    It's designed to be used with the generic routing utilities in
    model_hosting_container_standards.sagemaker.routing.

    Args:
        handler_type: The LoRA handler type identifier (e.g., 'register_adapter',
                     'unregister_adapter'). These identifiers are defined in
                     the LoRAHandlerType enum in constants.py.

    Returns:
        RouteConfig: The route configuration if the handler type has a default route
        None: If the handler type doesn't have a default route (e.g., transform-only
              handlers like INJECT_ADAPTER_ID)
    """
    return LORA_ROUTE_REGISTRY.get(handler_type)


__all__ = [
    "LORA_ROUTE_REGISTRY",
    "get_lora_route_config",
]
