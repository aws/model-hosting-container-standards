"""SageMaker function loader for loading handlers from environment variables.

Usage:
    # Load handlers from environment variables
    ping_handler = SageMakerFunctionLoader.get_ping_handler_from_env()
    invoke_handler = SageMakerFunctionLoader.get_invocation_handler_from_env()

    # Load specific functions
    handler = SageMakerFunctionLoader.load_function_from_spec("model:predict_fn")
"""

import os
from typing import Any, Callable, Optional, Union

from ..common.fastapi.config import FastAPIEnvVars
from ..common.handler.spec import HandlerSpec, parse_handler_spec
from .config import SageMakerDefaults, SageMakerEnvVars


class SageMakerFunctionLoader:
    """Utility class for SageMaker function loading from environment variables.

    This class provides class methods for loading handlers and functions
    in SageMaker environments. All methods are class methods - no instantiation needed.
    """

    # Class-level cached function loader to avoid recreating instances
    _default_function_loader: Optional[Any] = None

    @classmethod
    def get_function_loader(cls):
        """Get or create the default SageMaker function loader (cached)."""
        if cls._default_function_loader is None:
            script_path = os.getenv(
                SageMakerEnvVars.SAGEMAKER_MODEL_PATH, SageMakerDefaults.SCRIPT_PATH
            )
            cls._default_function_loader = cls._create_function_loader(script_path)

        return cls._default_function_loader

    @classmethod
    def get_custom_script_filename(cls, default_script: Optional[str] = None) -> str:
        """Get custom script filename from environment or default."""
        default = default_script or SageMakerDefaults.SCRIPT_FILENAME
        return os.getenv(SageMakerEnvVars.CUSTOM_SCRIPT_FILENAME, default)

    @classmethod
    def _create_function_loader(cls, script_path: str) -> Any:
        """Create a function loader for the given script path."""
        script_filename = os.getenv(
            SageMakerEnvVars.CUSTOM_SCRIPT_FILENAME, SageMakerDefaults.SCRIPT_FILENAME
        )

        search_paths = [script_path]
        model_file_path = os.path.join(script_path, script_filename)

        from ..common.custom_code_ref_resolver.function_loader import FunctionLoader

        module_aliases = {"model": model_file_path}
        function_loader = FunctionLoader(search_paths, module_aliases)

        # Preload the model file if it exists to trigger any decorators
        if os.path.isfile(model_file_path):
            function_loader.load_module_from_file(model_file_path)

        return function_loader

    @classmethod
    def load_function_from_spec(
        cls, spec: str, custom_script_path: Optional[str] = None
    ) -> Optional[Callable]:
        """Load function from specification string."""
        # Use HandlerSpec for validation and parsing
        handler_spec = HandlerSpec(spec)

        # Only attempt to load if it's a valid function specification
        if handler_spec.is_router_path or not handler_spec.is_valid_function_spec():
            return None

        # Check if we can use the cached loader (when using default SageMaker paths)
        default_script_path = os.getenv(
            SageMakerEnvVars.SAGEMAKER_MODEL_PATH, SageMakerDefaults.SCRIPT_PATH
        )

        # Use cached loader if no custom path specified OR if custom path matches default
        if custom_script_path is None or custom_script_path == default_script_path:
            return cls.get_function_loader().load_function(spec)

        # Create a new loader for the custom path
        function_loader = cls._create_function_loader(custom_script_path)
        return function_loader.load_function(spec)

    @classmethod
    def _get_handler_from_env(
        cls, env_var: str, custom_script_path: Optional[str] = None
    ) -> Union[Callable[..., Any], str, None]:
        """Generic method to get handler from environment variable.

        Returns:
        - Callable: When spec is a function/method specification
        - str: When spec is a router URL (starts with "/")
        - None: When environment variable is not set
        """
        spec_string = os.getenv(env_var)
        spec = parse_handler_spec(spec_string)

        if spec:
            if spec.is_router_path:
                return spec.router_path
            elif spec.is_function:
                return cls.load_function_from_spec(spec.spec, custom_script_path)
        return None

    @classmethod
    def get_ping_handler_from_env(
        cls, custom_script_path: Optional[str] = None
    ) -> Union[Callable[..., Any], str, None]:
        """Get custom ping handler from CUSTOM_FASTAPI_PING_HANDLER."""
        return cls._get_handler_from_env(
            FastAPIEnvVars.CUSTOM_FASTAPI_PING_HANDLER, custom_script_path
        )

    @classmethod
    def get_invocation_handler_from_env(
        cls, custom_script_path: Optional[str] = None
    ) -> Union[Callable[..., Any], str, None]:
        """Get custom invocation handler from CUSTOM_FASTAPI_INVOCATION_HANDLER."""
        return cls._get_handler_from_env(
            FastAPIEnvVars.CUSTOM_FASTAPI_INVOCATION_HANDLER, custom_script_path
        )

    @classmethod
    def _get_handler_spec(cls, env_var: str) -> Optional[HandlerSpec]:
        """Generic method to get handler specification from environment variable."""
        spec_string = os.getenv(env_var)
        return parse_handler_spec(spec_string)

    @classmethod
    def get_ping_handler_spec(cls) -> Optional[HandlerSpec]:
        """Get ping handler specification object."""
        return cls._get_handler_spec(FastAPIEnvVars.CUSTOM_FASTAPI_PING_HANDLER)

    @classmethod
    def get_invocation_handler_spec(cls) -> Optional[HandlerSpec]:
        """Get invocation handler specification object."""
        return cls._get_handler_spec(FastAPIEnvVars.CUSTOM_FASTAPI_INVOCATION_HANDLER)
