from enum import Enum

from pydantic import BaseModel, ConfigDict


class SessionRequestType(str, Enum):
    """Types of session management requests.

    NEW_SESSION: Request to create a new stateful session
    CLOSE: Request to close an existing session
    """

    NEW_SESSION = "NEW_SESSION"
    CLOSE = "CLOSE"


class SessionRequest(BaseModel):
    """Request model for session management operations.

    Validates that the request contains a valid requestType field
    and rejects any extra fields.
    """

    model_config = ConfigDict(extra="forbid")

    requestType: SessionRequestType


class SageMakerSessionHeader:
    """SageMaker API header constants for stateful session management.

    SESSION_ID: Header to pass existing session ID to the server
    NEW_SESSION_ID: Header returned by server when creating a new session
    CLOSED_SESSION_ID: Header returned by server when closing a session
    """

    SESSION_ID = "X-Amzn-SageMaker-Session-Id"
    NEW_SESSION_ID = "X-Amzn-SageMaker-New-Session-Id"
    CLOSED_SESSION_ID = "X-Amzn-SageMaker-Closed-Session-Id"


# Error messages for session management
SESSION_DISABLED_ERROR_DETAIL = "Invalid payload. stateful sessions not enabled"
SESSION_DISABLED_LOG_MESSAGE = (
    f"Invalid payload. stateful sessions not enabled, "
    f"{SageMakerSessionHeader.SESSION_ID} header not supported"
)
