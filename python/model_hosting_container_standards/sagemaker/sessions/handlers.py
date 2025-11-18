from datetime import datetime, timezone
from http import HTTPStatus

from fastapi import Request, Response
from fastapi.exceptions import HTTPException

from ...logging_config import logger
from .manager import SessionManager
from .models import SageMakerSessionHeader, SessionRequestType
from .utils import get_session_id_from_request


def get_handler_for_request_type(request_type: SessionRequestType):
    """Map session request type to the appropriate handler function.

    Args:
        request_type: The type of session request

    Returns:
        Handler function for the request type, or None if no handler
    """
    if request_type == SessionRequestType.NEW_SESSION:
        return create_session
    elif request_type == SessionRequestType.CLOSE:
        return close_session
    else:
        return None


async def close_session(session_manager: SessionManager, raw_request: Request):
    """Close an existing session and clean up its resources.

    Args:
        session_manager: SessionManager instance to manage the session lifecycle
        raw_request: FastAPI Request object containing session ID in headers

    Returns:
        Response with 200 status and closed session ID in headers

    Raises:
        HTTPException: If session closure fails with 424 FAILED_DEPENDENCY status
    """
    session_id = get_session_id_from_request(raw_request)
    try:
        session_manager.close_session(session_id)
        logger.info(f"Session {session_id} closed")
        return Response(
            status_code=HTTPStatus.OK.value,
            content=f"Session {session_id} closed",
            headers={SageMakerSessionHeader.CLOSED_SESSION_ID: f"{session_id}"},
        )
    except Exception as e:
        logger.exception(f"Failed to close session: {str(e)}")
        raise HTTPException(
            status_code=HTTPStatus.FAILED_DEPENDENCY.value,
            detail=f"Failed to close session: {str(e)}",
        )


async def create_session(session_manager: SessionManager, raw_request: Request):
    """Create a new stateful session with expiration tracking.

    Args:
        session_manager: SessionManager instance to manage the session lifecycle
        raw_request: FastAPI Request object (unused but part of handler signature)

    Returns:
        Response with 200 status, session ID and expiration in headers

    Raises:
        HTTPException: If session creation fails with 424 FAILED_DEPENDENCY status
    """
    try:
        session = session_manager.create_session()
        # expiration_ts is guaranteed to be set for newly created sessions
        assert session.expiration_ts is not None
        expiration_ts = datetime.fromtimestamp(
            session.expiration_ts, tz=timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
        logger.info(f"Session {session.session_id} created")
        return Response(
            status_code=HTTPStatus.OK.value,
            content=f"Session {session.session_id} created",
            headers={
                SageMakerSessionHeader.NEW_SESSION_ID: f"{session.session_id}; Expires={expiration_ts}"
            },
        )
    except Exception as e:
        logger.exception(f"Failed to create session: {str(e)}")
        raise HTTPException(
            status_code=HTTPStatus.FAILED_DEPENDENCY.value,
            detail=f"Failed to create session: {str(e)}",
        )
