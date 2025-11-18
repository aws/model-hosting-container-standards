from http import HTTPStatus

from fastapi import Request
from fastapi.exceptions import HTTPException

from ...logging_config import logger
from .manager import SessionManager
from .models import SageMakerSessionHeader


def get_session_id_from_request(raw_request: Request):
    """Extract the session ID from the request headers.

    Args:
        raw_request: FastAPI Request object

    Returns:
        Session ID string if present in headers, None otherwise
    """
    if not raw_request.headers:
        return None
    return raw_request.headers.get(SageMakerSessionHeader.SESSION_ID)


def get_session(session_manager: SessionManager, raw_request: Request):
    """Retrieve the session associated with the request.

    Args:
        session_manager: SessionManager instance, or None if sessions disabled
        raw_request: FastAPI Request object containing session ID in headers

    Returns:
        Session instance if found, None if no session ID in request

    Raises:
        HTTPException: If session header is present but sessions are not enabled (400 BAD_REQUEST)
        ValueError: If session_id is not found in the session manager (propagated from get_session)
    """
    session_id = get_session_id_from_request(raw_request)

    # If sessions are not enabled but session header is present, reject the request
    if session_manager is None:
        if session_id is not None:
            logger.error(
                f"Invalid payload. stateful sessions not enabled, {SageMakerSessionHeader.SESSION_ID} header not supported"
            )
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail=f"Invalid payload. stateful sessions not enabled, {SageMakerSessionHeader.SESSION_ID} header not supported",
            )
        return None

    session = session_manager.get_session(session_id)
    return session
