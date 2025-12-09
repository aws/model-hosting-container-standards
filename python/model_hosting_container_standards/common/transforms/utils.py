from typing import Any, Dict, Optional

import jmespath

from ...logging_config import logger

DEFAULT_MAX_DEPTH_TO_CREATE = 2


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


def set_value(
    obj: dict,
    path: str,
    value: Any,
    create_parent: bool = False,
    max_create_depth: int = DEFAULT_MAX_DEPTH_TO_CREATE,
) -> dict:
    """Set value in a nested dict using JMESPath for the parent path.

    Args:
        obj: The dictionary to modify
        path: Dot-separated path to the value (e.g., "parent.child.key")
        value: The value to set
        create_parent: If True, create missing parent structures. If False, raise KeyError if parent doesn't exist.
        max_create_depth: Maximum nesting depth when creating parents (None = unlimited). Only applies if create_parent=True. Defaults to DEFAULT_MAX_DEPTH_TO_CREATE.

    Returns:
        The modified obj dictionary

    Raises:
        KeyError: If parent path doesn't exist and create_parent=False, or if max_create_depth is exceeded
    """
    # Split "parent.child" into ('parent', 'child')
    if "." not in path:
        obj[path] = value
        return obj

    *parent_parts, child = path.split(".")

    if create_parent:
        return _set_value_with_parent_creation(
            obj, parent_parts, child, value, max_create_depth
        )

    parent_expr = ".".join(parent_parts)

    # Use JMESPath to find the parent node
    parent = jmespath.search(parent_expr, obj)
    if parent is None:
        logger.exception(f"Parent path '{parent_expr}' not found in {obj}")
        raise KeyError(f"Parent path '{parent_expr}' not found in {obj}")

    # Assign directly (since parent is a dict)
    parent[child] = value
    return obj


def _set_value_with_parent_creation(
    obj: dict,
    parent_parts: list[str],
    child,
    value,
    max_create_depth: int = DEFAULT_MAX_DEPTH_TO_CREATE,
    _full_depth: Optional[int] = None,
):
    """Set a value in a nested dict, creating parent structures as needed.

    Args:
        obj: The dictionary to modify
        parent_parts: List of parent keys to traverse/create
        child: The final child key to set
        value: The value to set at the child key
        max_create_depth: Maximum nesting depth of the full path when creating parents (None = unlimited). Defaults to DEFAULT_MAX_DEPTH_TO_CREATE.
        _full_depth: Internal parameter tracking the total depth of the full path

    Returns:
        The modified obj dictionary

    Raises:
        KeyError: If max_create_depth is exceeded when creating parents
    """
    if len(parent_parts) == 0:
        obj[child] = value
        return obj

    parent_expr = ".".join(parent_parts)
    parent = jmespath.search(parent_expr, obj)

    if parent is None and len(parent_parts) > 0:
        # Parent doesn't exist, we'll need to create it
        # On first call where we need to create, check the depth limit
        if _full_depth is None:
            _full_depth = len(parent_parts) + 1  # +1 for the child key

            if max_create_depth is not None and _full_depth > max_create_depth:
                logger.exception(
                    f"Path depth of {_full_depth} exceeds maximum allowed depth of {max_create_depth}."
                )
                raise KeyError(
                    f"Path '{'.'.join(parent_parts + [str(child)])}' has depth {_full_depth}, "
                    f"which exceeds max depth of {max_create_depth}."
                )

        return _set_value_with_parent_creation(
            obj=obj,
            parent_parts=parent_parts[:-1],
            child=parent_parts[-1],
            value={child: value},
            max_create_depth=max_create_depth,
            _full_depth=_full_depth,
        )

    parent[child] = value

    return obj
