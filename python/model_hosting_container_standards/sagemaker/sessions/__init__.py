from typing import Optional

from ...common.transforms.base_factory import create_transform_decorator
from .models import SageMakerSessionHeader
from .transform import SessionApiTransform
from .transforms.create_session import (
    RESPONSE_CONTENT_KEY,
    resolve_engine_session_transform,
)


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
    session_id_path: Optional[str] = None,
    content_path: Optional[str] = None,
):
    """Register a handler for creating a new session.

    Args:
        session_id_path: JMESPath expression for session ID
        content_path: JMESPath expression for session content
    """
    response_shape = {
        RESPONSE_CONTENT_KEY: content_path,
    }
    if handler_type == "create_session":
        if not session_id_path:
            raise ValueError("session_id_path is required for create_session")
        response_shape[SageMakerSessionHeader.NEW_SESSION_ID] = session_id_path
    return _create_engine_session_transform_decorator(handler_type)(
        request_shape, response_shape
    )
