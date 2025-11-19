from .close_session import CloseSessionApiTransform
from .create_session import CreateSessionApiTransform


def resolve_engine_session_transform(handler_type: str):
    """Resolve the appropriate transform class for engine session handlers.

    :param str handler_type: Type of session handler ('create_session' or 'close_session')
    :return: Transform class or None if handler type is not recognized
    """
    if handler_type == "create_session":
        return CreateSessionApiTransform
    elif handler_type == "close_session":
        return CloseSessionApiTransform
    return None
