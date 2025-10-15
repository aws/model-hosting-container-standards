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
