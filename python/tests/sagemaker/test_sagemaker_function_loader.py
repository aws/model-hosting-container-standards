"""Unit tests for SageMaker loader functionality."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from model_hosting_container_standards.common.fastapi import EnvVars
from model_hosting_container_standards.sagemaker.config import SageMakerEnvVars
from model_hosting_container_standards.sagemaker.sagemaker_loader import (
    SageMakerFunctionLoader,
)


class TestSageMakerLoaders:
    """Test SageMaker-specific loader functionality."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = Path(self.temp_dir) / "model.py"

        # Create test Python file
        test_code = """
def predict_fn(input_data):
    return f"prediction for {input_data}"

def transform_fn(model, input_data, content_type, accept):
    return f"transformed {input_data}"

def ping_handler():
    return "pong"

def invocation_handler(data):
    return f"processed: {data}"

class ModelHandler:
    def __init__(self):
        self.model = "test_model"

    def predict(self, data):
        return f"handler prediction for {data}"
"""
        self.test_file.write_text(test_code)

        yield

        # Cleanup
        import shutil

        shutil.rmtree(self.temp_dir)

        # Clear the class-level cache to avoid test interference
        SageMakerFunctionLoader._default_function_loader = None

    def test_create_loader_with_default_env_vars(self):
        """Test loader creation with default environment variables."""
        # Clear cache first
        SageMakerFunctionLoader._default_function_loader = None

        with patch.dict(
            os.environ,
            {
                "SAGEMAKER_MODEL_PATH": str(self.temp_dir),
                "CUSTOM_SCRIPT_FILENAME": "model.py",
            },
            clear=False,
        ):
            loader = SageMakerFunctionLoader.get_function_loader()

            # Test that the loader was created successfully
            assert loader is not None

            # Test that the module alias is set up correctly
            assert "model" in loader.module_aliases
            expected_path = os.path.join(str(self.temp_dir), "model.py")
            assert loader.module_aliases["model"] == expected_path

    def test_create_loader_uses_sagemaker_model_path(self):
        """Test loader creation uses SAGEMAKER_MODEL_PATH as search path."""
        # Clear cache first
        SageMakerFunctionLoader._default_function_loader = None

        with patch.dict(
            os.environ,
            {
                "SAGEMAKER_MODEL_PATH": str(self.temp_dir),
                "CUSTOM_SCRIPT_FILENAME": "model.py",
            },
            clear=False,
        ):
            loader = SageMakerFunctionLoader.get_function_loader()

            # Test that SAGEMAKER_MODEL_PATH is used as search path
            assert str(self.temp_dir) in loader.file_loader.search_paths

    def test_preloading_existing_file(self):
        """Test that existing model file is preloaded and cached."""
        # Clear cache first
        SageMakerFunctionLoader._default_function_loader = None

        with patch.dict(
            os.environ,
            {
                "SAGEMAKER_MODEL_PATH": str(self.temp_dir),
                "CUSTOM_SCRIPT_FILENAME": "model.py",
            },
            clear=False,
        ):
            loader = SageMakerFunctionLoader.get_function_loader()

            # The module should be preloaded when the loader is created
            # Check that the module is already cached
            from pathlib import Path

            resolved_path = Path(self.test_file).resolve()
            expected_cache_key = f"file:{resolved_path}"
            assert expected_cache_key in loader._module_cache

            # Test that we can load functions from the preloaded module
            func = loader.load_function("model:predict_fn")
            assert func is not None
            assert func("test_data") == "prediction for test_data"

            # Module should still be cached after function loading
            assert expected_cache_key in loader._module_cache

    def test_preloading_nonexistent_file_no_error(self):
        """Test that nonexistent model file doesn't cause errors during loader creation."""
        with patch.dict(
            os.environ,
            {
                "SAGEMAKER_MODEL_PATH": str(self.temp_dir),
                "CUSTOM_SCRIPT_FILENAME": "nonexistent.py",
            },
            clear=False,
        ):
            # This should not raise an exception
            loader = SageMakerFunctionLoader.get_function_loader()
            assert loader is not None

            # The module alias should still be set up
            assert "model" in loader.module_aliases

    def test_preloading_file_with_syntax_error_raises_exception(self):
        """Test that syntax errors in model file raise exceptions during loader creation."""
        # Create a file with syntax error
        bad_file = Path(self.temp_dir) / "bad_model.py"
        bad_file.write_text("def invalid_syntax(\n  # Missing closing parenthesis")

        # Clear cache first
        SageMakerFunctionLoader._default_function_loader = None

        with patch.dict(
            os.environ,
            {
                "SAGEMAKER_MODEL_PATH": str(self.temp_dir),
                "CUSTOM_SCRIPT_FILENAME": "bad_model.py",
            },
            clear=False,
        ):
            # Now syntax errors should be raised immediately during preloading
            with pytest.raises(SyntaxError):
                SageMakerFunctionLoader.get_function_loader()

    def test_load_function_with_model_alias(self):
        """Test loading functions using the 'model' alias."""
        with patch.dict(
            os.environ,
            {
                "SAGEMAKER_MODEL_PATH": str(self.temp_dir),
                "CUSTOM_SCRIPT_FILENAME": "model.py",
            },
            clear=False,
        ):
            loader = SageMakerFunctionLoader.get_function_loader()

            # Test loading different functions
            predict_fn = loader.load_function("model:predict_fn")
            assert predict_fn("input") == "prediction for input"

            transform_fn = loader.load_function("model:transform_fn")
            assert transform_fn(None, "data", "json", "json") == "transformed data"

            # Test loading class method
            handler_predict = loader.load_function("model:ModelHandler.predict")
            assert handler_predict is not None

    def test_environment_variable_defaults(self):
        """Test that default values are used when environment variables are not set."""
        # Clear cache first
        SageMakerFunctionLoader._default_function_loader = None

        # Clear relevant environment variables
        env_vars_to_clear = ["SAGEMAKER_MODEL_PATH", "CUSTOM_SCRIPT_FILENAME"]
        with patch.dict(os.environ, {}, clear=False):
            for var in env_vars_to_clear:
                os.environ.pop(var, None)

            loader = SageMakerFunctionLoader.get_function_loader()

            # Should use default values
            assert "model" in loader.module_aliases
            # Default path should be /opt/ml/model/model.py
            expected_default = "/opt/ml/model/model.py"
            assert loader.module_aliases["model"] == expected_default

    def test_caching_behavior(self):
        """Test that module caching works correctly."""
        # Clear cache first
        SageMakerFunctionLoader._default_function_loader = None

        with patch.dict(
            os.environ,
            {
                "SAGEMAKER_MODEL_PATH": str(self.temp_dir),
                "CUSTOM_SCRIPT_FILENAME": "model.py",
            },
            clear=False,
        ):
            loader = SageMakerFunctionLoader.get_function_loader()

            # Load the same function twice
            func1 = loader.load_function("model:predict_fn")
            func2 = loader.load_function("model:predict_fn")

            # Should be the same function object (from cached module)
            assert func1 is func2

            # Module should be in cache (using resolved path for consistency)
            from pathlib import Path

            resolved_path = Path(self.test_file).resolve()
            expected_cache_key = f"file:{resolved_path}"
            assert expected_cache_key in loader._module_cache

    def test_public_load_module_from_file_method(self):
        """Test the public load_module_from_file method works with SageMaker loader."""
        with patch.dict(
            os.environ,
            {
                "SAGEMAKER_MODEL_PATH": str(self.temp_dir),
                "CUSTOM_SCRIPT_FILENAME": "model.py",
            },
            clear=False,
        ):
            loader = SageMakerFunctionLoader.get_function_loader()

            # Test loading module directly
            module = loader.load_module_from_file(str(self.test_file))
            assert module is not None
            assert hasattr(module, "predict_fn")
            assert hasattr(module, "ModelHandler")

            # Test that it's cached
            module2 = loader.load_module_from_file(str(self.test_file))
            assert module is module2

    def test_get_ping_handler_function(self):
        """Test getting ping handler as function."""
        with patch.dict(
            os.environ,
            {
                "SAGEMAKER_MODEL_PATH": str(self.temp_dir),
                "CUSTOM_SCRIPT_FILENAME": "model.py",
                EnvVars.CUSTOM_FASTAPI_PING_HANDLER: f"{self.test_file.name}:ping_handler",
            },
            clear=False,
        ):
            ping_func = SageMakerFunctionLoader.get_ping_handler_from_env(self.temp_dir)
            assert ping_func is not None
            assert ping_func() == "pong"

    def test_get_ping_handler_router(self):
        """Test getting ping handler as router URL."""
        with patch.dict(
            os.environ,
            {EnvVars.CUSTOM_FASTAPI_PING_HANDLER: "/health"},
            clear=False,
        ):
            result = SageMakerFunctionLoader.get_ping_handler_from_env()
            assert result == "/health"

    def test_get_ping_handler_none(self):
        """Test getting ping handler when not set."""
        result = SageMakerFunctionLoader.get_ping_handler_from_env()
        assert result is None

    def test_get_invocation_handler_function(self):
        """Test getting invocation handler as function."""
        with patch.dict(
            os.environ,
            {
                "SAGEMAKER_MODEL_PATH": str(self.temp_dir),
                "CUSTOM_SCRIPT_FILENAME": "model.py",
                EnvVars.CUSTOM_FASTAPI_INVOCATION_HANDLER: f"{self.test_file.name}:invocation_handler",
            },
            clear=False,
        ):
            inv_func = SageMakerFunctionLoader.get_invocation_handler_from_env(
                self.temp_dir
            )
            assert inv_func is not None
            assert inv_func("test") == "processed: test"

    def test_get_invocation_handler_router(self):
        """Test getting invocation handler as router URL."""
        with patch.dict(
            os.environ,
            {EnvVars.CUSTOM_FASTAPI_INVOCATION_HANDLER: "/v1/chat/completions"},
            clear=False,
        ):
            result = SageMakerFunctionLoader.get_invocation_handler_from_env()
            assert result == "/v1/chat/completions"

    def test_get_invocation_handler_none(self):
        """Test getting invocation handler when not set."""
        result = SageMakerFunctionLoader.get_invocation_handler_from_env()
        assert result is None

    def test_get_custom_script_filename(self):
        """Test getting custom script filename."""
        with patch.dict(
            os.environ,
            {SageMakerEnvVars.CUSTOM_SCRIPT_FILENAME: "custom_model.py"},
            clear=False,
        ):
            result = SageMakerFunctionLoader.get_custom_script_filename()
            assert result == "custom_model.py"

    def test_get_custom_script_filename_default(self):
        """Test getting default script filename."""
        result = SageMakerFunctionLoader.get_custom_script_filename()
        assert result == "model.py"

    def test_load_function_from_spec(self):
        """Test loading function from specification."""
        with patch.dict(
            os.environ,
            {
                "SAGEMAKER_MODEL_PATH": str(self.temp_dir),
                "CUSTOM_SCRIPT_FILENAME": "model.py",
            },
            clear=False,
        ):
            # Clear cache to pick up new environment
            SageMakerFunctionLoader._default_function_loader = None

            func = SageMakerFunctionLoader.load_function_from_spec("model:ping_handler")
            assert func is not None
            assert func() == "pong"

    def test_load_function_from_spec_router_path(self):
        """Test that router paths return None."""
        result = SageMakerFunctionLoader.load_function_from_spec("/health")
        assert result is None

    def test_handler_specs(self):
        """Test getting handler specification objects."""
        with patch.dict(
            os.environ,
            {
                EnvVars.CUSTOM_FASTAPI_PING_HANDLER: "/health",
                EnvVars.CUSTOM_FASTAPI_INVOCATION_HANDLER: "model:invoke",
            },
            clear=False,
        ):
            ping_spec = SageMakerFunctionLoader.get_ping_handler_spec()
            invoke_spec = SageMakerFunctionLoader.get_invocation_handler_spec()

            assert ping_spec is not None
            assert invoke_spec is not None
            assert ping_spec.is_router_path
            assert invoke_spec.is_function

    def test_load_function_from_spec_with_custom_path(self):
        """Test loading function from specification with custom script path.

        This test verifies that when a custom_script_path is provided, the loader
        creates a new FunctionLoader instance that looks in the custom directory
        rather than using the cached default loader.
        """
        # Create a separate directory with a custom script file
        custom_temp_dir = tempfile.mkdtemp()
        custom_script_name = "custom_handlers.py"
        custom_test_file = Path(custom_temp_dir) / custom_script_name

        # Create custom script with distinctly different function behavior
        custom_test_code = """
def predict_fn(input_data):
    return f"CUSTOM prediction for {input_data}"

def custom_ping_handler():
    return "custom pong from different directory"
"""
        custom_test_file.write_text(custom_test_code)

        try:
            # Set up environment with default path pointing to self.temp_dir
            with patch.dict(
                os.environ,
                {
                    "SAGEMAKER_MODEL_PATH": str(self.temp_dir),
                    "CUSTOM_SCRIPT_FILENAME": custom_script_name,
                },
                clear=False,
            ):
                # Clear cache to pick up new environment
                SageMakerFunctionLoader._default_function_loader = None

                # Load from default path - should use the cached loader with self.temp_dir/model.py
                # (Note: self.temp_dir has model.py from setup_method)
                with patch.dict(
                    os.environ,
                    {"CUSTOM_SCRIPT_FILENAME": "model.py"},
                    clear=False,
                ):
                    SageMakerFunctionLoader._default_function_loader = None
                    default_func = SageMakerFunctionLoader.load_function_from_spec(
                        "model:predict_fn"
                    )
                    assert default_func is not None
                    assert default_func("test") == "prediction for test"

                # Load from custom path - should create new loader pointing to custom_temp_dir/custom_handlers.py
                custom_func = SageMakerFunctionLoader.load_function_from_spec(
                    "model:predict_fn", custom_script_path=custom_temp_dir
                )
                assert custom_func is not None
                assert custom_func("test") == "CUSTOM prediction for test"

        finally:
            import shutil

            shutil.rmtree(custom_temp_dir)

    def test_load_function_from_spec_with_different_filename(self):
        """Test loading function with custom path and different script filename."""
        # Create a separate directory with a descriptively named script file
        custom_temp_dir = tempfile.mkdtemp()
        script_filename = "my_inference_handlers.py"
        custom_test_file = Path(custom_temp_dir) / script_filename

        # Create custom script with different filename
        custom_test_code = """
def predict_fn(input_data):
    return f"INFERENCE HANDLERS prediction for {input_data}"

def health_check():
    return "healthy from inference handlers"
"""
        custom_test_file.write_text(custom_test_code)

        try:
            # Set up environment to use the descriptive filename
            with patch.dict(
                os.environ,
                {
                    "SAGEMAKER_MODEL_PATH": str(self.temp_dir),
                    "CUSTOM_SCRIPT_FILENAME": script_filename,
                },
                clear=False,
            ):
                # Clear cache to pick up new environment
                SageMakerFunctionLoader._default_function_loader = None

                # Load from custom path with descriptive filename
                custom_func = SageMakerFunctionLoader.load_function_from_spec(
                    "model:predict_fn", custom_script_path=custom_temp_dir
                )
                assert custom_func is not None
                assert custom_func("test") == "INFERENCE HANDLERS prediction for test"

                # Also test loading the health_check function
                health_func = SageMakerFunctionLoader.load_function_from_spec(
                    "model:health_check", custom_script_path=custom_temp_dir
                )
                assert health_func is not None
                assert health_func() == "healthy from inference handlers"

        finally:
            import shutil

            shutil.rmtree(custom_temp_dir)

    def test_load_function_from_spec_with_descriptive_filename(self):
        """Test loading function with a descriptive, non-standard filename."""
        # Create a directory with a very descriptive filename
        custom_temp_dir = tempfile.mkdtemp()
        script_filename = "llm_chat_completion_handlers.py"
        custom_test_file = Path(custom_temp_dir) / script_filename

        # Create script with chat completion specific functions
        custom_test_code = """
def predict_fn(input_data):
    return f"LLM CHAT prediction for {input_data}"

def chat_completion_handler(messages):
    return {"response": f"Chat completion for {len(messages)} messages"}

def token_counter(text):
    return len(text.split())
"""
        custom_test_file.write_text(custom_test_code)

        try:
            # Set up environment to use the descriptive filename
            with patch.dict(
                os.environ,
                {
                    "SAGEMAKER_MODEL_PATH": str(self.temp_dir),
                    "CUSTOM_SCRIPT_FILENAME": script_filename,
                },
                clear=False,
            ):
                # Clear cache to pick up new environment
                SageMakerFunctionLoader._default_function_loader = None

                # Load functions from the descriptively named file
                predict_func = SageMakerFunctionLoader.load_function_from_spec(
                    "model:predict_fn", custom_script_path=custom_temp_dir
                )
                assert predict_func is not None
                assert predict_func("test") == "LLM CHAT prediction for test"

                # Load chat-specific function
                chat_func = SageMakerFunctionLoader.load_function_from_spec(
                    "model:chat_completion_handler", custom_script_path=custom_temp_dir
                )
                assert chat_func is not None
                result = chat_func(["hello", "world"])
                assert result["response"] == "Chat completion for 2 messages"

                # Load utility function
                counter_func = SageMakerFunctionLoader.load_function_from_spec(
                    "model:token_counter", custom_script_path=custom_temp_dir
                )
                assert counter_func is not None
                assert counter_func("hello world test") == 3

        finally:
            import shutil

            shutil.rmtree(custom_temp_dir)

    def test_load_function_from_spec_custom_path_same_as_default(self):
        """Test that custom path matching default path uses cached loader."""
        with patch.dict(
            os.environ,
            {
                "SAGEMAKER_MODEL_PATH": str(self.temp_dir),
                "CUSTOM_SCRIPT_FILENAME": "model.py",
            },
            clear=False,
        ):
            # Clear cache to pick up new environment
            SageMakerFunctionLoader._default_function_loader = None

            # Load function with custom path that matches default
            func1 = SageMakerFunctionLoader.load_function_from_spec(
                "model:predict_fn", custom_script_path=str(self.temp_dir)
            )

            # Load function without custom path (should use same cached loader)
            func2 = SageMakerFunctionLoader.load_function_from_spec("model:predict_fn")

            # Both should work and return the same function
            assert func1 is not None
            assert func2 is not None
            assert func1("test") == "prediction for test"
            assert func2("test") == "prediction for test"
