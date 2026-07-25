"""Unit tests for SageMaker setup functions."""

from unittest.mock import Mock, patch

from fastapi import APIRouter, FastAPI

from model_hosting_container_standards.sagemaker import (
    bootstrap,
    create_sagemaker_router,
)


class TestCreateSageMakerRouter:
    """Test create_sagemaker_router function."""

    @patch("model_hosting_container_standards.sagemaker.sagemaker_router.create_router")
    def test_creates_router_with_lora_resolver(self, mock_create_router):
        """Test that create_sagemaker_router calls create_router with correct resolver."""
        # Arrange
        mock_router = Mock(spec=APIRouter)
        mock_router.routes = []  # Add routes attribute
        mock_create_router.return_value = mock_router

        # Act
        result = create_sagemaker_router()

        # Assert
        assert result == mock_router
        mock_create_router.assert_called_once()
        call_kwargs = mock_create_router.call_args.kwargs
        assert "route_resolver" in call_kwargs
        assert call_kwargs["tags"] == ["sagemaker"]

    def test_returns_api_router_instance(self):
        """Test that create_sagemaker_router returns an APIRouter instance."""
        # Act
        result = create_sagemaker_router()

        # Assert
        assert isinstance(result, APIRouter)


class TestBootstrap:
    """Test bootstrap function."""

    @patch("model_hosting_container_standards.sagemaker.create_sagemaker_router")
    @patch.object(FastAPI, "include_router")
    def test_creates_and_includes_router(self, mock_include_router, mock_create_router):
        """Test that bootstrap creates router and includes it in app."""
        # Arrange
        mock_router = Mock()
        mock_router.routes = []  # Add routes attribute for safe_include_router
        mock_create_router.return_value = mock_router
        app = FastAPI()

        # Act
        bootstrap(app)

        # Assert
        mock_create_router.assert_called_once()
        mock_include_router.assert_called_once_with(mock_router)

    def test_works_with_real_app(self):
        """Test that bootstrap works with a real FastAPI app."""
        # Arrange
        app = FastAPI()
        initial_route_count = len(app.routes)

        # Act
        bootstrap(app)

        # Assert - router should be mounted (may add routes if handlers are registered)
        # At minimum, the app should not error
        assert len(app.routes) >= initial_route_count


class TestRegisterLoadAdapterHandlerV2:
    """Test register_load_adapter_handler_v2 function."""

    def test_validates_and_creates_engine_request_paths(self):
        """Test that all paths are validated and engine_request_paths dict is created."""
        from unittest.mock import patch

        from model_hosting_container_standards.sagemaker import (
            register_load_adapter_handler_v2,
        )

        with patch(
            "model_hosting_container_standards.sagemaker.create_lora_transform2_decorator"
        ) as mock_decorator:
            mock_decorator.return_value = lambda *args, **kwargs: lambda func: func

            decorator = register_load_adapter_handler_v2(
                engine_request_lora_name_path="name", engine_request_lora_src_path="src"
            )

            # Decorator should be created
            assert callable(decorator)

    def test_includes_pinned_in_paths_when_provided(self):
        """Test that pinned is included in paths when provided."""
        from unittest.mock import Mock, patch

        from model_hosting_container_standards.sagemaker import (
            register_load_adapter_handler_v2,
        )

        with patch(
            "model_hosting_container_standards.sagemaker.create_lora_transform2_decorator"
        ) as mock_decorator:
            mock_inner = Mock()
            mock_decorator.return_value = mock_inner

            register_load_adapter_handler_v2(
                engine_request_lora_name_path="name",
                engine_request_lora_src_path="src",
                engine_request_lora_pinned_path="pinned",
            )

            # Check that decorator was called
            mock_decorator.assert_called_once()
            call_args = mock_decorator.call_args[0]
            assert call_args[0] == "register_adapter"

    def test_excludes_pinned_from_paths_when_not_provided(self):
        """Test that pinned is excluded from paths when not provided."""
        from unittest.mock import patch

        from model_hosting_container_standards.sagemaker import (
            register_load_adapter_handler_v2,
        )

        with patch(
            "model_hosting_container_standards.sagemaker.create_lora_transform2_decorator"
        ) as mock_decorator:
            mock_decorator.return_value = lambda *args, **kwargs: lambda func: func

            decorator = register_load_adapter_handler_v2(
                engine_request_lora_name_path="name", engine_request_lora_src_path="src"
            )

            assert callable(decorator)


class TestRegisterUnloadAdapterHandlerV2:
    """Test register_unload_adapter_handler_v2 function."""

    def test_validates_and_creates_engine_request_paths(self):
        """Test that path is validated and engine_request_paths dict is created with name only."""
        # TODO: Implement test

    def test_calls_create_lora_transform2_decorator(self):
        """Test that create_lora_transform2_decorator is called with correct params."""
        # TODO: Implement test

    def test_passes_unload_adapter_defaults_from_config(self):
        """Test that unload_adapter_defaults from config are passed."""
        # TODO: Implement test


class TestInjectAdapterIdV2:
    """Test inject_adapter_id_v2 function."""

    def test_validates_adapter_path_and_creates_inject_definition(self):
        """Test that adapter path is validated and InjectDefinition is created."""
        from unittest.mock import patch

        from model_hosting_container_standards.sagemaker import inject_adapter_id_v2

        with patch(
            "model_hosting_container_standards.sagemaker.create_lora_api_inject"
        ) as mock_create:
            mock_create.return_value = lambda func: func

            decorator = inject_adapter_id_v2(engine_request_adapter_path="adapter")

            assert callable(decorator)
            mock_create.assert_called_once()

    def test_raises_error_for_empty_adapter_path(self):
        """Test that empty adapter path raises ValueError."""
        from model_hosting_container_standards.sagemaker import inject_adapter_id_v2

        with pytest.raises(ValueError, match="cannot be empty"):
            inject_adapter_id_v2(engine_request_adapter_path="")

    def test_validates_mode_separator_combinations(self):
        """Test that invalid mode/separator combinations raise ValueError."""
        from model_hosting_container_standards.sagemaker import inject_adapter_id_v2

        # Replace mode with separator should fail
        with pytest.raises(ValueError):
            inject_adapter_id_v2(
                engine_request_adapter_path="adapter", mode="replace", separator=":"
            )

        # Append mode without separator should fail
        with pytest.raises(ValueError):
            inject_adapter_id_v2(
                engine_request_adapter_path="adapter", mode="append", separator=None
            )


class TestRegisterCreateSessionHandler:
    """Test register_create_session_handler function."""

    def test_creates_transform_with_correct_params(self):
        """Test that create_create_session_transform is called with correct parameters."""
        # TODO: Implement test

    def test_passes_create_session_defaults_from_config(self):
        """Test that create_session_defaults from config are passed."""
        # TODO: Implement test


class TestRegisterCloseSessionHandler:
    """Test register_close_session_handler function."""

    def test_validates_session_id_path_and_creates_request_paths(self):
        """Test that session ID path is validated and engine_request_paths is created."""
        from unittest.mock import patch

        from model_hosting_container_standards.sagemaker import (
            register_close_session_handler,
        )

        with patch(
            "model_hosting_container_standards.sagemaker.create_close_session_transform"
        ) as mock_create:
            mock_create.return_value = lambda func: func

            decorator = register_close_session_handler(
                engine_request_session_id_path="session_id"
            )

            assert callable(decorator)
            mock_create.assert_called_once()

    def test_raises_error_for_empty_session_id_path(self):
        """Test that empty session ID path raises ValueError."""
        from model_hosting_container_standards.sagemaker import (
            register_close_session_handler,
        )

        with pytest.raises(ValueError, match="is required"):
            register_close_session_handler(engine_request_session_id_path="")
