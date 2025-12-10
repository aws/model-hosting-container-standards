from typing import Any, Dict

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
    """Set value in a nested dict using dot-separated path traversal.

    Note: This function assumes JMESPath-style dot notation but only supports simple
    period-separated dictionary traversal. It does not use JMESPath directly and
    does not support complex JMESPath expressions (filters, functions, etc.).

    Limitations:
        - Only supports dictionary containers. Lists and other container types are unsupported.
        - When create_parent=True, only dictionary structures will be created for missing parents.

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
    if len(parent_parts) == 0:
        obj[child] = value
        return obj

    # Find the deepest existing parent by manually traversing the dict
    current = obj
    existing_parent = {}
    existing_depth = 0

    for i, part in enumerate(parent_parts):
        if isinstance(current, dict) and part in current:
            current = current[part]
            existing_parent = current
            existing_depth = i + 1
        else:
            break

    # If we found the complete parent path, just set the value
    if existing_depth == len(parent_parts):
        existing_parent[child] = value
        return obj

    # Parent doesn't exist completely, we need to create missing parts
    if not create_parent:
        parent_expr = ".".join(parent_parts)
        logger.error(f"Parent path '{parent_expr}' not found in {obj}")
        raise KeyError(f"Parent path '{parent_expr}' not found in {obj}")

    # Check depth limit only when we need to create parents
    if max_create_depth is not None:
        full_depth = len(parent_parts) + 1  # +1 for the child key
        if full_depth > max_create_depth:
            full_path = ".".join(parent_parts + [str(child)])
            logger.exception(
                f"Path depth of {full_depth} exceeds maximum allowed depth of {max_create_depth}."
            )
            raise KeyError(
                f"Path '{full_path}' has depth {full_depth}, "
                f"which exceeds max depth of {max_create_depth}."
            )

    # Build the nested structure from the deepest level up
    current_value = {child: value}

    # Work backwards from the missing parts
    for i in range(len(parent_parts) - 1, existing_depth - 1, -1):
        current_value = {parent_parts[i]: current_value}

    # Set the constructed structure at the appropriate location
    if existing_depth == 0:
        # No existing parent found, set at root
        obj[parent_parts[0]] = current_value[parent_parts[0]]
    else:
        # Set at the existing parent level
        existing_parent[parent_parts[existing_depth]] = current_value[
            parent_parts[existing_depth]
        ]

    return obj
