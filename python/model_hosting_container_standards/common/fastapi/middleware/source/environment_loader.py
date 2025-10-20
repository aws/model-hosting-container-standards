"""Environment variable middleware loader."""

import os

from .....logging_config import logger
from ...config import FastAPIEnvVars
from .base import BaseMiddlewareLoader


class MiddlewareEnvironmentLoader(BaseMiddlewareLoader):
    """Loads middleware from environment variables using function loader."""

    def __init__(self) -> None:
        super().__init__()
        # Single mapping: name -> (env_var, property_name)
        self.middleware_mapping = {
            "throttle": (
                FastAPIEnvVars.CUSTOM_FASTAPI_MIDDLEWARE_THROTTLE,
                "throttle_middleware",
            ),
            "pre_post_process": (
                FastAPIEnvVars.CUSTOM_FASTAPI_MIDDLEWARE_PRE_POST_PROCESS,
                "pre_post_middleware",
            ),
            "pre_process": (FastAPIEnvVars.CUSTOM_PRE_PROCESS, "pre_fn"),
            "post_process": (FastAPIEnvVars.CUSTOM_POST_PROCESS, "post_fn"),
        }

    def load(self, function_loader) -> None:
        """Load all middlewares from environment variables.

        Args:
            function_loader: Function loader to use for loading middleware functions.
        """
        # Load each middleware type individually
        for name in self.middleware_mapping:
            self.load_middleware(name, function_loader)

        # Handle combination logic after loading individual pieces
        self._handle_pre_post_combination()

    def load_middleware(self, name: str, function_loader) -> None:
        """Load a specific middleware by name from environment variables.

        Args:
            name: Middleware name ("throttle", "pre_post_process", "pre_process", "post_process")
            function_loader: Function loader to use for loading middleware functions
        """
        if name not in self.middleware_mapping:
            return

        env_var, property_name = self.middleware_mapping[name]
        spec_string = os.getenv(env_var)

        if not spec_string:
            return

        try:
            middleware_func = function_loader.load_function(spec_string)
            if middleware_func:
                setattr(self, property_name, middleware_func)
                func_name = getattr(middleware_func, "__name__", str(middleware_func))
                logger.info(
                    f"[MIDDLEWARE_ENV] Loaded {name} from {env_var}: {func_name}"
                )
        except Exception as e:
            logger.error(
                f"[MIDDLEWARE_ENV] Failed to load {name} from {env_var} '{spec_string}': {e}"
            )

    def _handle_pre_post_combination(self) -> None:
        """Handle combination of pre_fn + post_fn into pre_post_middleware if needed."""
        # Only combine if we don't already have a direct pre_post_middleware
        if not self.pre_post_middleware and (self.pre_fn or self.post_fn):
            self.pre_post_middleware = self._combine_pre_post_middleware(
                "MIDDLEWARE_ENV"
            )
            logger.info(
                "[MIDDLEWARE_ENV] Created combined pre_post_process middleware from pre_process and/or post_process"
            )
