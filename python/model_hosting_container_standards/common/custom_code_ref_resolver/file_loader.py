"""File-based module loader for SageMaker model hosting containers.

This module provides the FileLoader class, which dynamically loads Python functions
and classes from files using specifications like "model.py:predict_fn" or "handler.py:MyClass.process".

Use Cases:
- Load inference handler: "model.py:model_fn" → loads model_fn() from model.py
- Load preprocessing: "preprocess.py:transform_input" → loads transform_input() from preprocess.py
- Load class methods: "handler.py:ModelHandler.predict" → loads ModelHandler.predict method
- Load from subdirs: "utils/helper.py:cleanup" → loads cleanup() from utils/helper.py
- Load with absolute path: "/opt/ml/model/custom.py:handler" → loads from absolute path
- Load with relative path: "../shared/common.py:utility" → loads using parent directory

Path Resolution:
- Absolute paths: Used directly (e.g., "/opt/ml/model/handler.py")
- Relative paths: Searched in configured search paths (default: current directory)
- Search path priority: Earlier paths in the list take precedence
- Parent directory references: Supported (e.g., "../parent/file.py")

Key Features:
- Dynamic loading from Python files by filename and attribute name
- Support for both absolute and relative file paths
- Multiple search paths with configurable priority
- Special handling for customer scripts with caching optimization
- Class method loading support (Class.method syntax)
- Integration with SageMaker's customer module loading system
"""

import importlib.util
from pathlib import Path
from typing import Any, List, Optional

from model_hosting_container_standards.exceptions import (
    HandlerFileNotFoundError,
    HandlerNotFoundError,
    ModuleLoadError,
)
from model_hosting_container_standards.logging_config import logger


class FileLoader:
    """Loader for functions from Python files."""

    def __init__(self, search_paths: Optional[List[str]] = None):
        """Initialize FileLoader.

        Args:
            search_paths: List of paths to search for files
        """
        self.search_paths = search_paths or ["."]

    def load_function(self, filename: str, attr_name: str) -> Any:
        """Load function or class from a Python file."""
        file_path = self._find_file(filename)
        if not file_path or not file_path.is_file():
            logger.debug(
                f"File '{filename}' not found in search paths: {self.search_paths}"
            )
            raise HandlerFileNotFoundError(filename, self.search_paths)

        try:
            module = self._load_regular_module(file_path)

            if not hasattr(module, attr_name):
                logger.debug(
                    f"Attribute '{attr_name}' not found in module loaded from {file_path}"
                )
                raise HandlerNotFoundError(f"{filename}:{attr_name}", str(file_path))

            return getattr(module, attr_name)
        except Exception as e:
            logger.debug(
                f"Failed to load module from {file_path}: {type(e).__name__}: {e}"
            )
            raise ModuleLoadError(str(file_path), str(e))

    def _load_regular_module(self, file_path: Path) -> Any:
        """Load a regular Python module from file path."""
        module_name = f"file_module_{str(hash(str(file_path)))[:8]}"
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if not spec or not spec.loader:
            logger.debug(f"Could not create module spec for {file_path}")
            raise ModuleLoadError(str(file_path), "Could not create module spec")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _find_file(self, filename: str) -> Optional[Path]:
        """Find file in search paths or as absolute path.

        Path resolution order:
        1. If filename is absolute path, use it directly
        2. Otherwise, search in configured search paths
        3. Relative paths are resolved relative to each search path
        """
        file_path = Path(filename)

        # If it's an absolute path, use it directly
        if file_path.is_absolute():
            if file_path.is_file():
                return file_path
            return None

        # For relative paths, search in all search paths
        for search_path in self.search_paths:
            candidate_path = Path(search_path) / filename
            if candidate_path.is_file():
                return candidate_path
        return None
