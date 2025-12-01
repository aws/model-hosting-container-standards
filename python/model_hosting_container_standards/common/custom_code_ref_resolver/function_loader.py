from typing import Any, Callable, Dict, List, Optional

from model_hosting_container_standards.exceptions import (
    HandlerFileNotFoundError,
    HandlerNotCallableError,
    HandlerNotFoundError,
)
from model_hosting_container_standards.logging_config import logger

from ..handler.spec import HandlerSpec
from .file_loader import FileLoader
from .module_loader import ModuleLoader


class FunctionLoader:
    """Unified function loader supporting multiple formats.

    Supports both file-based and module-based function specifications:

    File-based examples:
    - "model.py:predict_fn" → loads predict_fn() from model.py file
    - "handler.py:ModelHandler.process" → loads ModelHandler.process method from handler.py
    - "utils/preprocess.py:clean_data" → loads clean_data() from utils/preprocess.py

    Module-based examples:
    - "mypackage:handler_function" → loads handler_function() from installed mypackage module
    - "sklearn.preprocessing:StandardScaler" → loads StandardScaler class from sklearn
    - "model:inference_fn" → loads inference_fn() from model module (via alias configuration)

    Module aliases allow mapping module names to file paths:
    - module_aliases={"model": "/opt/ml/model/model.py"} enables "model:predict_fn" → loads from file
    """

    def __init__(
        self,
        search_paths: Optional[List[str]] = None,
        module_aliases: Optional[Dict[str, str]] = None,
    ):
        """Initialize FunctionLoader.

        Args:
            search_paths: List of paths to search for files
            module_aliases: Optional dict mapping module names to file paths (e.g., {"model": "model.py"})
        """
        self.file_loader = FileLoader(search_paths)
        self.module_loader = ModuleLoader()
        self.module_aliases = module_aliases or {}
        self._module_cache: Dict[str, Any] = {}

    def load_function(self, spec: str) -> Optional[Callable]:
        """Load function from specification string."""
        try:
            # Parse and validate the specification
            handler_spec = HandlerSpec(spec)

            # Only handle function specs, not router paths
            if handler_spec.is_router_path:
                return None

            # Validate the function specification
            module_path, func_name = handler_spec.validate_function_spec()

            # Load the module (with caching)
            module = self._load_module(module_path, handler_spec.is_file_function)

            # Extract and validate the attribute
            result = self._extract_attribute_from_module(module, module_path, func_name)
            return self._validate_callable(result, spec)

        except Exception as e:
            logger.warning(
                f"Failed to load function from spec '{spec}': {type(e).__name__}: {e}"
            )
            raise

    def _load_module(self, module_path: str, is_file: bool) -> Any:
        """Load module with caching support."""
        if is_file:
            return self.load_module_from_file(module_path)
        else:
            return self._load_module_from_import(module_path)

    def _get_cached_module(self, cache_key: str) -> Optional[Any]:
        """Get module from cache if it exists."""
        if cache_key in self._module_cache:
            logger.debug(f"Using cached module for {cache_key}")
            return self._module_cache[cache_key]
        return None

    def load_module_from_file(self, file_path: str) -> Any:
        """Load module from file with caching.

        Public method to preload and cache a module from a file path.
        Useful for warming up the module cache.

        Args:
            file_path: Path to the Python file to load

        Returns:
            The loaded module

        Raises:
            HandlerFileNotFoundError: If the file doesn't exist
            ModuleLoadError: If the module fails to load
        """
        # Resolve the file path first to get the actual file location
        file_path_obj = self.file_loader._find_file(file_path)
        if not file_path_obj or not file_path_obj.is_file():
            raise HandlerFileNotFoundError(file_path, self.file_loader.search_paths)

        # Use the resolved absolute path for caching to avoid duplicate loads
        cache_key = f"file:{file_path_obj.resolve()}"

        # Check cache first
        cached_module = self._get_cached_module(cache_key)
        if cached_module is not None:
            return cached_module

        # Load and cache the module
        module = self.file_loader._load_regular_module(file_path_obj)
        self._module_cache[cache_key] = module
        return module

    def _load_module_from_import(self, module_path: str) -> Any:
        """Load module from import with caching and alias support."""
        # Check for module aliases first
        if module_path in self.module_aliases:
            # Resolve alias to file path
            file_path = self.module_aliases[module_path]
            return self.load_module_from_file(file_path)

        cache_key = f"module:{module_path}"

        # Check cache first
        cached_module = self._get_cached_module(cache_key)
        if cached_module is not None:
            return cached_module

        # Load and cache the module
        module = self.module_loader._load_module(module_path)
        self._module_cache[cache_key] = module
        return module

    def _extract_attribute_from_module(
        self, module: Any, source: str, attr_path: str
    ) -> Optional[Callable[..., Any]]:
        """Extract attribute from loaded module, handling nested attributes."""
        if "." in attr_path:
            # Format: Class.method
            # Examples:
            # - source="model.py", attr_path="ModelHandler.process" → loads ModelHandler class, then gets process method
            # - source="sklearn.preprocessing", attr_path="StandardScaler.fit" → loads StandardScaler, then gets fit method
            # - source="utils.py", attr_path="MyClass.inner_class.method" → navigates through nested attributes
            parts = attr_path.split(".")
            obj = getattr(module, parts[0])  # Load class

            # Navigate through nested attributes
            for i, part in enumerate(parts[1:], 1):
                if not hasattr(obj, part):
                    current_path = ".".join(parts[: i + 1])
                    logger.debug(f"Attribute '{current_path}' not found in {source}")
                    raise HandlerNotFoundError(f"{source}:{current_path}", source)
                obj = getattr(obj, part)

            result = obj
        else:
            # Simple function
            # Examples:
            # - source="model.py", attr_path="predict_fn" → loads predict_fn function directly
            # - source="mypackage", attr_path="handler_function" → loads handler_function from module
            # - source="utils/preprocess.py", attr_path="clean_data" → loads clean_data function from file
            if not hasattr(module, attr_path):
                logger.debug(f"Attribute '{attr_path}' not found in {source}")
                raise HandlerNotFoundError(f"{source}:{attr_path}", source)
            result = getattr(module, attr_path)

        return self._validate_callable(result, f"{source}:{attr_path}")

    def _validate_callable(self, result: Any, spec: str) -> Callable:
        """Validate that the loaded result is callable."""
        if result is None:
            logger.debug(f"Loaded handler from spec '{spec}' is None")
            raise HandlerNotCallableError(spec, "NoneType")
        elif not callable(result):
            logger.debug(
                f"Loaded handler from spec '{spec}' is not callable, got {type(result).__name__}"
            )
            raise HandlerNotCallableError(spec, type(result).__name__)
        return result
