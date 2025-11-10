"""Shared fixtures for sagemaker/sessions tests."""

import os
import shutil
import tempfile
import time
from unittest.mock import Mock

import pytest
from fastapi import Request

from model_hosting_container_standards.sagemaker.sessions.manager import (
    Session,
    SessionManager,
    init_session_manager_from_env,
)
from model_hosting_container_standards.sagemaker.sessions.models import (
    SageMakerSessionHeader,
)


@pytest.fixture
def temp_session_storage():
    """Create and cleanup temporary directory for session storage."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def mock_session_manager():
    """Create a mock session manager for testing."""
    return Mock(spec=SessionManager)


@pytest.fixture
def mock_request():
    """Create a mock FastAPI request without session headers."""
    request = Mock(spec=Request)
    request.headers = {}
    return request


@pytest.fixture
def mock_request_with_session():
    """Create a mock FastAPI request with session ID header."""
    request = Mock(spec=Request)
    request.headers = {SageMakerSessionHeader.SESSION_ID: "test-session-123"}
    return request


@pytest.fixture
def mock_session():
    """Create a mock session with standard properties."""
    session = Mock(spec=Session)
    session.session_id = "test-session-123"
    session.expiration_ts = time.time() + 1000
    return session


@pytest.fixture
def session_manager(temp_session_storage):
    """Create a real session manager with temporary storage for integration tests."""
    properties = {"sessions_path": temp_session_storage, "sessions_expiration": "600"}
    return SessionManager(properties)


@pytest.fixture
def enable_sessions_env(monkeypatch, temp_session_storage):
    """Set environment variables to enable stateful sessions for tests.

    This fixture sets the necessary environment variables and reinitializes
    the global session manager from those variables.
    """
    monkeypatch.setenv("OPTION_ENABLE_STATEFUL_SESSIONS", "true")
    monkeypatch.setenv("OPTION_SESSIONS_PATH", temp_session_storage)
    monkeypatch.setenv("OPTION_SESSIONS_EXPIRATION", "600")

    # Reinitialize the global session manager with the new environment variables
    init_session_manager_from_env()

    yield

    # Clean up - reinitialize with empty env (will set session_manager to None)
    monkeypatch.delenv("OPTION_ENABLE_STATEFUL_SESSIONS", raising=False)
    monkeypatch.delenv("OPTION_SESSIONS_PATH", raising=False)
    monkeypatch.delenv("OPTION_SESSIONS_EXPIRATION", raising=False)
    init_session_manager_from_env()
