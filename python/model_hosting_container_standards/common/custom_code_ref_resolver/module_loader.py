"""Module-based loader for installed Python packages.

This loader handles module-based function specifications (no .py extension):

Examples:
- "mypackage:handler_fn" → loads handler_fn() from installed mypackage module
- "sklearn.preprocessing:StandardScaler" → loads StandardScaler from sklearn
- "numpy:array" → loads array function from numpy
- "model:predict" → loads predict() from model module (special customer script handling)
- "utils.helpers:format_output" → loads format_output() from utils.helpers submodule
"""

import importlib
from typing import Any, Optional

from model_hosting_container_standards.exceptions import (
    HandlerNotFoundError,
    ModuleLoadError,
)
from model_hosting_container_standards.logging_config import logger


class ModuleLoader:
    """Loader for functions from installed Python modules."""

    def load_function(self, module_path: str, attr_name: str) -> Optional[Any]:
        """Load function or class from an installed Python module."""
        module = self._load_module(module_path)
        return self._extract_attribute(module, module_path, attr_name)

    def _load_module(self, module_path: str) -> Any:
        """Load a Python module by path."""
        try:
            return importlib.import_module(module_path)
        except Exception as e:
            logger.debug(
                f"Failed to import module {module_path}: {type(e).__name__}: {e}"
            )
            raise ModuleLoadError(module_path, f"Failed to import module: {e}")

    def _extract_attribute(self, module: Any, module_path: str, attr_name: str) -> Any:
        """Extract an attribute from a loaded module."""
        if not hasattr(module, attr_name):
            logger.debug(f"Attribute '{attr_name}' not found in module '{module_path}'")
            raise HandlerNotFoundError(f"{module_path}:{attr_name}", module_path)

        return getattr(module, attr_name)
