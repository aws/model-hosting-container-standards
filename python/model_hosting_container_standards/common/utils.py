from typing import Any, Dict

import jmespath

from ..logging_config import logger


def _compile_jmespath_expressions(shape: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively compile JMESPath expressions in the shape dictionary.

    :param Dict[str, Any] shape: Dictionary containing JMESPath expressions to compile
    :return Dict[str, Any]: Dictionary with compiled JMESPath expressions
    """
    compiled_shape = {}
    for key, value in shape.items():
        if isinstance(value, str):
            # Compile the JMESPath expression
            compiled_shape[key] = jmespath.compile(value)
        elif isinstance(value, dict):
            # Recursively compile nested dictionaries
            compiled_shape[key] = _compile_jmespath_expressions(value)
        else:
            logger.warning(
                f"Request/response mapping must be a dictionary of strings (nested allowed), not {type(value)}. This value will be ignored."
            )
    return compiled_shape


def set_value(obj: dict, path: str, value: Any) -> dict:
    """Set value in a nested dict using JMESPath for the parent path."""
    # Split "parent.child" into ('parent', 'child')
    if "." not in path:
        obj[path] = value
        return obj

    *parent_parts, child = path.split(".")
    parent_expr = ".".join(parent_parts)

    # Use JMESPath to find the parent node
    parent = jmespath.search(parent_expr, obj)
    if parent is None:
        logger.exception(f"Parent path '{parent_expr}' not found in {obj}")
        raise KeyError(f"Parent path '{parent_expr}' not found in {obj}")

    # Assign directly (since parent is a dict)
    parent[child] = value
    return obj
