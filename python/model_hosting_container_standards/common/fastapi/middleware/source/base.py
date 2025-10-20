"""Base middleware loader for common middleware loading functionality."""

from typing import Any, Callable, Optional

from .....logging_config import logger


class BaseMiddlewareLoader:
    """Base class for middleware loaders that handle common middleware types."""

    def __init__(self) -> None:
        # Core middleware types that all loaders handle
        self.pre_fn: Optional[Callable] = None
        self.post_fn: Optional[Callable] = None
        self.pre_post_middleware: Optional[Callable] = None
        self.throttle_middleware: Optional[Callable] = None

    def _create_pre_post_middleware(
        self,
        pre_process_func: Optional[Callable],
        post_process_func: Optional[Callable],
        middleware_name: str,
        log_prefix: str,
    ) -> Callable:
        """Create a middleware that applies pre and post processing functions.

        Args:
            pre_process_func: Function to call before the main handler
            post_process_func: Function to call after the main handler
            middleware_name: Name for the middleware function
            log_prefix: Prefix for log messages

        Returns:
            Async middleware function
        """

        async def middleware(request: Any, call_next: Callable) -> Any:
            """Generic pre/post process middleware."""
            try:
                # Apply pre-process if exists
                if pre_process_func is not None:
                    logger.debug(f"[{log_prefix}] Applying pre-process function")
                    request = await pre_process_func(request) or request

                # Call the next middleware/handler
                response = await call_next(request)

                # Apply post-process if exists
                if post_process_func is not None:
                    logger.debug(f"[{log_prefix}] Applying post-process function")
                    response = await post_process_func(response) or response

                return response

            except Exception as e:
                logger.error(
                    f"[{log_prefix}] Error in {middleware_name} middleware: {e}"
                )
                # Return 500 error
                from fastapi.responses import JSONResponse

                return JSONResponse(
                    status_code=500,
                    content={
                        "error": f"Internal server error in {middleware_name} middleware",
                        "message": str(e),
                    },
                )

        # Set a meaningful name
        middleware.__name__ = middleware_name
        return middleware

    def _combine_pre_post_middleware(self, log_prefix: str) -> Optional[Callable]:
        """Combine pre_fn and post_fn into pre_post_middleware if needed."""
        if self.pre_fn or self.post_fn:
            return self._create_pre_post_middleware(
                self.pre_fn, self.post_fn, "combined_pre_post_process", log_prefix
            )
        return None

    def get_middleware(self, middleware_name: str) -> Optional[Callable]:
        """Get specific middleware by name."""
        if middleware_name == "throttle":
            return self.throttle_middleware
        elif middleware_name == "pre_post_process":
            return self.pre_post_middleware
        return None

    def has_middlewares(self) -> bool:
        """Check if any middlewares were loaded."""
        return any(
            [
                self.pre_fn,
                self.post_fn,
                self.pre_post_middleware,
                self.throttle_middleware,
            ]
        )
