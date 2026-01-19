from typing import Any, Dict, Literal, Optional, Union

import jmespath

from ...logging_config import logger

DEFAULT_MAX_DEPTH_TO_CREATE = 2


SAGEMAKER_HEADER_PREFIX = "X-Amzn-SageMaker-"

SageMakerInjectMode = Literal["append", "prepend", "replace"]


def to_hyphens(field_name: str) -> str:
    return field_name.replace("_", "-")


def to_sagemaker_headers(field_name: str) -> str:
    field_name = to_hyphens(field_name)
    field_name = SAGEMAKER_HEADER_PREFIX + field_name.title()
    return field_name


def _compile_jmespath_expressions(shape: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively compile JMESPath expressions in the shape dictionary.

    :param Dict[str, Any] shape: Dictionary containing JMESPath expressions to compile
    :return Dict[str, Any]: Dictionary with compiled JMESPath expressions
    """
    compiled_shape: Dict[str, Union[jmespath.parser.ParsedResult, Dict[str, Any]]] = {}
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


def _set(new_value, original_value, mode, separator):
    if mode == "replace":
        return new_value
    elif original_value is None:
        return new_value
    elif mode == "append":
        return f"{original_value}{separator}{new_value}"
    elif mode == "prepend":
        return f"{new_value}{separator}{original_value}"


def set_value(
    obj: Dict[str, Any],
    path: str,
    value: Any,
    create_parent: bool = False,
    max_create_depth: Optional[int] = DEFAULT_MAX_DEPTH_TO_CREATE,
    mode: SageMakerInjectMode = "replace",
    separator: Optional[str] = None,
) -> Dict:
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
        obj[path] = _set(value, obj.get(path), mode, separator)
        return obj

    *parent_parts, child = path.split(".")
    if len(parent_parts) == 0:
        obj[child] = _set(value, obj.get(child), mode, separator)
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
        existing_parent[child] = _set(
            value, existing_parent.get(child), mode, separator
        )
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
            logger.exception(
                f"Path depth of {path} exceeds maximum allowed depth of {max_create_depth}."
            )
            raise KeyError(
                f"Path '{path}' has depth {full_depth}, "
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


ValidPrefix = Literal["body.", "headers.", "query_params.", "path_params."]


def validate_engine_path(
    engine_path: str, default_prefix: Optional[ValidPrefix] = "body."
) -> str:
    if not isinstance(engine_path, str):
        raise ValueError(
            f"Engine path must be a string, got {type(engine_path)}: {engine_path}"
        )
    if engine_path == "body":
        return engine_path
    if not engine_path.startswith(
        ("body.", "headers.", "query_params.", "path_params.")
    ):
        if default_prefix:
            return f"{default_prefix}{engine_path}"
        else:
            raise ValueError(f"Invalid path, missing valid prefix: {engine_path}")
    return engine_path
