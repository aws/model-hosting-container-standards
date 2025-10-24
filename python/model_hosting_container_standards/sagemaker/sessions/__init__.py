from ...common.transforms.base_factory import create_transform_decorator
from .transform import SessionApiTransform


def resolve_session_transform(handler_type: str) -> type:
    """Resolve the transform class for session management.

    Args:
        handler_type: Handler type (unused - sessions only have one transform type)

    Returns:
        SessionApiTransform class
    """
    # handler_type is unused because sessions only have one transform type,
    # but the parameter is required by the transform resolver interface
    return SessionApiTransform


def create_session_transform_decorator():
    return create_transform_decorator("stateful_session_manager", resolve_session_transform)
