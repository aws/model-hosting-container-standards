from typing import Optional

from ...common.transforms.base_factory import create_transform_decorator
from .models import SageMakerSessionHeader
from .transform import SessionApiTransform
from .transforms import resolve_engine_session_transform
from .transforms.constants import RESPONSE_CONTENT_KEY


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
    return create_transform_decorator(
        "stateful_session_manager", resolve_session_transform
    )


def _create_engine_session_transform_decorator(handler_type: str):
    return create_transform_decorator(handler_type, resolve_engine_session_transform)


def register_engine_session_handler(
    handler_type: str,
    request_shape,
    response_session_id_path: Optional[str] = None,
    content_path: Optional[str] = None,
):
    """Register a handler for engine-specific session management.

    Args:
        handler_type: Type of session handler ('create_session' or 'close_session')
        request_shape: JMESPath expressions for transforming request data
        response_session_id_path: JMESPath expression for extracting session ID FROM
                                  the engine's response (required for 'create_session',
                                  ignored for 'close_session')
        content_path: JMESPath expression for extracting content from response

    Returns:
        Decorator function for registering the session handler

    Raises:
        ValueError: If handler_type is invalid or required parameters are missing
    """
    # Validate handler_type
    if handler_type not in ("create_session", "close_session"):
        raise ValueError(
            f"Invalid handler_type '{handler_type}'. "
            f"Must be 'create_session' or 'close_session'"
        )

    response_shape = {
        RESPONSE_CONTENT_KEY: content_path,
    }

    if handler_type == "create_session":
        if not response_session_id_path:
            raise ValueError("response_session_id_path is required for create_session")
        response_shape[SageMakerSessionHeader.NEW_SESSION_ID] = response_session_id_path

    return _create_engine_session_transform_decorator(handler_type)(
        request_shape, response_shape
    )
