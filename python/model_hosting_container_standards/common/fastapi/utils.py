import json
from typing import Any, Dict, Optional, Union

from fastapi import Request, Response
from fastapi.responses import JSONResponse
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


def serialize_response(response: Union[Response, JSONResponse]):
    """Create a structured data dictionary for JMESPath transformations.

    Extracts and organizes response data into a standardized format that can be used
    with JMESPath expressions to transform and extract specific data elements.

    :param Union[Response, JSONResponse] response: Response body data - can be:
        - FastAPI Response object
        - JSONResponse object
    :return Dict[str, Any]: Structured data with body, headers, status_code, and media_type
    """
    # Process response body based on type
    body = response.body.decode(response.charset)
    try:
        body = json.loads(body)
    except json.JSONDecodeError:
        # If body is not JSON, keep it as a string
        # logger.warning(f"Response body is not JSON, keeping as string: {e}")
        pass

    return {
        "body": body,
        "headers": response.headers,
    }
