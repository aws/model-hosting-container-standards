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
