from typing import Any, Dict, Optional, Union

from fastapi import Request
from pydantic import BaseModel


def serialize_request(
    request: Optional[Union[BaseModel, Dict[str, Any]]], raw_request: Request
) -> Dict[str, Any]:
    """Create a structured data dictionary for JMESPath transformations.

    Extracts and organizes request data into a standardized format that can be used
    with JMESPath expressions to transform and extract specific data elements.

    :param Optional[Union[BaseModel, Dict[str, Any]]] request: Request body data - can be:
        - Pydantic BaseModel instance (will be converted to dict via model_dump())
        - Dictionary containing request data
        - None if no request body
    :param Request raw_request: The raw FastAPI request object
    :return Dict[str, Any]: Structured data with body, headers, query_params, and path_params
    """
    # Process request body based on type
    body = None
    if isinstance(request, BaseModel):
        body = request.model_dump()
    elif isinstance(request, dict):
        body = request
    # If request is None or any other type, body remains None

    return {
        "body": body,
        "headers": raw_request.headers,
        "query_params": raw_request.query_params,
        "path_params": raw_request.path_params,
    }
