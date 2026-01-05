from typing import Dict, Optional

from ...common.transforms.base_factory import create_transform_decorator
from ...logging_config import logger
from .models import SageMakerSessionHeader
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
    return create_transform_decorator(
        "stateful_session_manager", resolve_session_transform
    )


def build_session_request_shape(
    session_id_path: Optional[str],
    additional_shape: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Build the request shape for session handlers with proper session ID injection.

    This helper consolidates the logic for constructing request shapes, ensuring
    the session ID is always properly mapped and warning about any conflicts.

    Args:
        session_id_path: Optional target path for the session ID in the request.
                        If None, session ID is not injected into the request.
        additional_shape: Optional additional transformations to merge.

    Returns:
        A complete request shape dict with session ID and any additional mappings.
    """
    request_shape: Dict[str, str] = {}

    if additional_shape:
        request_shape.update(additional_shape)

    # Only inject session ID if a path is specified
    if session_id_path:
        # Warn if session_id_path would be overwritten
        if session_id_path in request_shape:
            existing_value = request_shape[session_id_path]
            logger.warning(
                f"Session ID path '{session_id_path}' found in additional_request_shape "
                f"with value '{existing_value}'. This will be overwritten with the "
                f"SageMaker session header value."
            )

        request_shape[session_id_path] = (
            f'headers."{SageMakerSessionHeader.SESSION_ID}"'
        )

    return request_shape
