"""Unit tests for sessions manager module."""

import json
import os
import tempfile
import time
from threading import Thread
from unittest.mock import patch

import pytest

from model_hosting_container_standards.sagemaker.sessions.manager import (
    Session,
    SessionManager,
)


class TestSession:
    """Test Session class."""

    @pytest.fixture
    def session(self, temp_session_storage):
        """Create a test session."""
        session_id = "test-session-123"
        expiration_ts = time.time() + 1000
        session = Session(session_id, temp_session_storage, expiration_ts)
        os.makedirs(session.files_path, exist_ok=True)
        return session

    def test_session_initialization(self, temp_session_storage):
        """Test session initialization with provided expiration."""
        session_id = "test-session-456"
        expiration_ts = time.time() + 500
        session = Session(session_id, temp_session_storage, expiration_ts)

        assert session.session_id == session_id
        assert session.expiration_ts == expiration_ts
        assert session.files_path == os.path.join(temp_session_storage, session_id)

    def test_session_initialization_loads_expiration_from_disk(
        self, temp_session_storage
    ):
        """Test session initialization loads expiration from disk when not provided."""
        session_id = "test-session-789"
        expiration_ts = time.time() + 1000

        # Create session directory and store expiration
        session_dir = os.path.join(temp_session_storage, session_id)
        os.makedirs(session_dir)
        expiration_file = os.path.join(session_dir, ".expiration_ts")
        with open(expiration_file, "w") as f:
            json.dump(expiration_ts, f)

        # Create session without providing expiration
        session = Session(session_id, temp_session_storage, expiration_ts=None)

        assert session.expiration_ts == expiration_ts

    @pytest.mark.parametrize(
        "key,value",
        [
            ("test_key", "test_value"),  # str
            ("count", 42),  # int
            ("config", {"name": "test", "enabled": True, "count": 123}),  # dict
            ("items", ["item1", "item2", "item3"]),  # list
        ],
        ids=["string", "int", "dict", "list"],
    )
    def test_put_and_get_value(self, session, key, value):
        """Test storing and retrieving JSON-serializable values."""
        session.put(key, value)
        result = session.get(key)
        assert result == value

    def test_get_nonexistent_key_returns_default(self, session):
        """Test getting a non-existent key returns the default value."""
        result = session.get("nonexistent_key", "default_value")
        assert result == "default_value"

    def test_get_nonexistent_key_returns_none_by_default(self, session):
        """Test getting a non-existent key returns None by default."""
        result = session.get("nonexistent_key")
        assert result is None

    def test_remove_deletes_session_directory(self, session):
        """Test remove method deletes the session directory."""
        # Ensure directory exists
        assert os.path.exists(session.files_path)

        # Remove session
        result = session.remove()

        assert result is True
        assert not os.path.exists(session.files_path)

    def test_remove_nonexistent_session_raises_error(self, temp_session_storage):
        """Test removing a non-existent session raises ValueError."""
        session = Session(
            "nonexistent-session", temp_session_storage, time.time() + 1000
        )

        with pytest.raises(ValueError, match="session directory does not exist"):
            session.remove()

    def test_path_with_normal_key(self, session):
        """Test _path generates correct file path for normal key."""
        key = "test_key"
        expected_path = os.path.join(session.files_path, key)

        result = session._path(key)

        assert result == expected_path

    def test_path_sanitizes_slashes(self, session):
        """Test _path sanitizes slashes in key names."""
        key = "folder/subfolder/key"
        expected_path = os.path.join(session.files_path, "folder-subfolder-key")

        result = session._path(key)

        assert result == expected_path

    @pytest.mark.parametrize(
        "invalid_key,error_pattern",
        [
            ("../etc/passwd", "'..' not allowed"),  # Direct parent traversal
            ("/etc/passwd", "absolute paths not allowed"),  # Absolute path
            ("valid/../../../etc/passwd", "'..' not allowed"),  # Complex traversal
        ],
        ids=["parent_directory", "absolute_path", "complex_traversal"],
    )
    def test_path_rejects_invalid_keys(self, session, invalid_key, error_pattern):
        """Test _path rejects various path traversal security attacks."""
        with pytest.raises(ValueError, match=error_pattern):
            session._path(invalid_key)


class TestSessionManager:
    """Test SessionManager class."""

    def test_session_manager_initialization_default_expiration(
        self, temp_session_storage
    ):
        """Test session manager initializes with default expiration."""
        properties = {"sessions_path": temp_session_storage}
        manager = SessionManager(properties)

        assert manager.expiration == 1200  # 20 minutes default

    def test_session_manager_initialization_custom_expiration(
        self, temp_session_storage
    ):
        """Test session manager initializes with custom expiration."""
        properties = {
            "sessions_path": temp_session_storage,
            "sessions_expiration": "300",
        }
        manager = SessionManager(properties)

        assert manager.expiration == 300

    def test_session_manager_initialization_creates_sessions_directory(
        self, temp_session_storage
    ):
        """Test session manager creates sessions directory if it doesn't exist."""
        sessions_path = os.path.join(temp_session_storage, "new_sessions")
        properties = {"sessions_path": sessions_path}

        manager = SessionManager(properties)

        assert os.path.exists(sessions_path)
        assert manager.sessions_path == sessions_path

    def test_session_manager_loads_existing_sessions(self, temp_session_storage):
        """Test session manager loads existing sessions from disk on initialization."""
        # Create some existing session directories
        session_id_1 = "existing-session-1"
        session_id_2 = "existing-session-2"
        os.makedirs(os.path.join(temp_session_storage, session_id_1))
        os.makedirs(os.path.join(temp_session_storage, session_id_2))

        # Create expiration files
        for session_id in [session_id_1, session_id_2]:
            session_dir = os.path.join(temp_session_storage, session_id)
            expiration_file = os.path.join(session_dir, ".expiration_ts")
            with open(expiration_file, "w") as f:
                json.dump(time.time() + 1000, f)

        properties = {"sessions_path": temp_session_storage}
        manager = SessionManager(properties)

        assert session_id_1 in manager.sessions
        assert session_id_2 in manager.sessions
        assert len(manager.sessions) == 2

    def test_session_manager_uses_dev_shm_when_available(self):
        """Test SessionManager defaults to /dev/shm when it exists."""
        with patch(
            "model_hosting_container_standards.sagemaker.sessions.manager.os.path.exists"
        ) as mock_exists:
            with patch(
                "model_hosting_container_standards.sagemaker.sessions.manager.os.access"
            ) as mock_access:
                with patch(
                    "model_hosting_container_standards.sagemaker.sessions.manager.os.makedirs"
                ):
                    with patch(
                        "model_hosting_container_standards.sagemaker.sessions.manager.os.listdir",
                        return_value=[],
                    ):
                        # Simulate /dev/shm exists and is accessible
                        mock_exists.side_effect = lambda path: path == "/dev/shm"
                        mock_access.return_value = True

                        manager = SessionManager({})

                        assert manager.sessions_path == "/dev/shm/sagemaker_sessions"

    def test_session_manager_falls_back_to_temp_when_dev_shm_missing(self):
        """Test SessionManager falls back to temp directory when /dev/shm doesn't exist."""
        with patch(
            "model_hosting_container_standards.sagemaker.sessions.manager.os.path.exists"
        ) as mock_exists:
            with patch(
                "model_hosting_container_standards.sagemaker.sessions.manager.os.makedirs"
            ):
                with patch(
                    "model_hosting_container_standards.sagemaker.sessions.manager.os.listdir",
                    return_value=[],
                ):
                    # Simulate /dev/shm doesn't exist
                    mock_exists.return_value = False

                    manager = SessionManager({})

                    expected_path = os.path.join(
                        tempfile.gettempdir(), "sagemaker_sessions"
                    )
                    assert manager.sessions_path == expected_path

    def test_session_manager_falls_back_to_temp_when_dev_shm_not_accessible(self):
        """Test SessionManager falls back to temp directory when /dev/shm is not accessible."""
        with patch(
            "model_hosting_container_standards.sagemaker.sessions.manager.os.path.exists"
        ) as mock_exists:
            with patch(
                "model_hosting_container_standards.sagemaker.sessions.manager.os.access"
            ) as mock_access:
                with patch(
                    "model_hosting_container_standards.sagemaker.sessions.manager.os.makedirs"
                ):
                    with patch(
                        "model_hosting_container_standards.sagemaker.sessions.manager.os.listdir",
                        return_value=[],
                    ):
                        # Simulate /dev/shm exists but is not accessible
                        mock_exists.side_effect = lambda path: path == "/dev/shm"
                        mock_access.return_value = False

                        manager = SessionManager({})

                        expected_path = os.path.join(
                            tempfile.gettempdir(), "sagemaker_sessions"
                        )
                        assert manager.sessions_path == expected_path

    def test_session_manager_falls_back_on_permission_error(self):
        """Test SessionManager falls back to temp directory on permission errors."""
        with patch(
            "model_hosting_container_standards.sagemaker.sessions.manager.os.path.exists"
        ) as mock_exists:
            with patch(
                "model_hosting_container_standards.sagemaker.sessions.manager.os.access"
            ) as mock_access:
                with patch(
                    "model_hosting_container_standards.sagemaker.sessions.manager.os.makedirs"
                ) as mock_makedirs:
                    with patch(
                        "model_hosting_container_standards.sagemaker.sessions.manager.os.listdir"
                    ) as mock_listdir:
                        # Simulate /dev/shm is accessible
                        mock_exists.side_effect = lambda path: path == "/dev/shm"
                        mock_access.return_value = True

                        # First makedirs call raises PermissionError, second succeeds
                        mock_makedirs.side_effect = [PermissionError(), None]
                        mock_listdir.return_value = []

                        manager = SessionManager({})

                        # Should fall back to temp directory
                        expected_path = os.path.join(
                            tempfile.gettempdir(), "sagemaker_sessions"
                        )
                        assert manager.sessions_path == expected_path
                        # makedirs should be called twice (once for /dev/shm, once for temp)
                        assert mock_makedirs.call_count == 2

    def test_session_manager_falls_back_on_os_error(self):
        """Test SessionManager falls back to temp directory on OS errors."""
        with patch(
            "model_hosting_container_standards.sagemaker.sessions.manager.os.path.exists"
        ) as mock_exists:
            with patch(
                "model_hosting_container_standards.sagemaker.sessions.manager.os.access"
            ) as mock_access:
                with patch(
                    "model_hosting_container_standards.sagemaker.sessions.manager.os.makedirs"
                ) as mock_makedirs:
                    with patch(
                        "model_hosting_container_standards.sagemaker.sessions.manager.os.listdir"
                    ) as mock_listdir:
                        # Simulate /dev/shm is accessible
                        mock_exists.side_effect = lambda path: path == "/dev/shm"
                        mock_access.return_value = True

                        # First makedirs call raises OSError, second succeeds
                        mock_makedirs.side_effect = [OSError("Disk full"), None]
                        mock_listdir.return_value = []

                        manager = SessionManager({})

                        # Should fall back to temp directory
                        expected_path = os.path.join(
                            tempfile.gettempdir(), "sagemaker_sessions"
                        )
                        assert manager.sessions_path == expected_path

    def test_session_manager_respects_custom_sessions_path(self, temp_session_storage):
        """Test SessionManager uses custom sessions_path when provided."""
        custom_path = os.path.join(temp_session_storage, "custom_sessions")
        properties = {"sessions_path": custom_path}

        manager = SessionManager(properties)

        # Should use the custom path, not /dev/shm or temp
        assert manager.sessions_path == custom_path
        assert os.path.exists(custom_path)

    def test_session_manager_actually_creates_temp_directory(
        self, temp_session_storage
    ):
        """Test SessionManager physically creates 'sagemaker_sessions' folder in temp directory when /dev/shm unavailable."""
        # Use a subdirectory of temp_session_storage to simulate temp directory
        fake_temp_dir = os.path.join(temp_session_storage, "fake_tmp")
        os.makedirs(fake_temp_dir, exist_ok=True)

        with patch(
            "model_hosting_container_standards.sagemaker.sessions.manager.tempfile.gettempdir",
            return_value=fake_temp_dir,
        ):
            with patch(
                "model_hosting_container_standards.sagemaker.sessions.manager.os.path.exists"
            ) as mock_exists:
                with patch(
                    "model_hosting_container_standards.sagemaker.sessions.manager.os.access"
                ) as mock_access:
                    # Simulate /dev/shm doesn't exist, but let other paths work normally
                    def exists_side_effect(path):
                        if path == "/dev/shm":
                            return False
                        # Use real exists for other paths
                        return (
                            os.path.exists.__wrapped__(path)
                            if hasattr(os.path.exists, "__wrapped__")
                            else True
                        )

                    mock_exists.side_effect = exists_side_effect
                    mock_access.return_value = False

                    manager = SessionManager({})

                    # Verify the path is in the temp directory and ends with 'sagemaker_sessions'
                    expected_path = os.path.join(fake_temp_dir, "sagemaker_sessions")
                    assert manager.sessions_path == expected_path

                    # Check using real filesystem (outside the patch context for exists)
                    import pathlib

                    assert pathlib.Path(expected_path).exists()
                    assert pathlib.Path(expected_path).is_dir()

                    # Verify it's named 'sagemaker_sessions'
                    assert os.path.basename(expected_path) == "sagemaker_sessions"

    def test_create_session_generates_unique_id(self, session_manager):
        """Test create_session generates a unique session ID."""
        session1 = session_manager.create_session()
        session2 = session_manager.create_session()

        assert session1.session_id != session2.session_id
        assert len(session1.session_id) == 36  # UUID format
        assert len(session2.session_id) == 36

    def test_create_session_sets_expiration(self, session_manager):
        """Test create_session sets expiration timestamp."""
        before_time = time.time()
        session = session_manager.create_session()
        after_time = time.time()

        expected_min = before_time + session_manager.expiration
        expected_max = after_time + session_manager.expiration

        assert expected_min <= session.expiration_ts <= expected_max

    def test_create_session_creates_directory(self, session_manager):
        """Test create_session creates the session directory."""
        session = session_manager.create_session()

        assert os.path.exists(session.files_path)
        assert os.path.isdir(session.files_path)

    def test_create_session_persists_expiration(self, session_manager):
        """Test create_session persists expiration to disk."""
        session = session_manager.create_session()

        expiration_file = os.path.join(session.files_path, ".expiration_ts")
        assert os.path.exists(expiration_file)

        with open(expiration_file, "r") as f:
            stored_expiration = json.load(f)

        assert stored_expiration == session.expiration_ts

    def test_create_session_adds_to_registry(self, session_manager):
        """Test create_session adds session to internal registry."""
        session = session_manager.create_session()

        assert session.session_id in session_manager.sessions
        assert session_manager.sessions[session.session_id] == session

    def test_get_session_returns_valid_session(self, session_manager):
        """Test get_session returns a valid non-expired session."""
        created_session = session_manager.create_session()

        retrieved_session = session_manager.get_session(created_session.session_id)

        assert retrieved_session is not None
        assert retrieved_session.session_id == created_session.session_id

    @pytest.mark.parametrize(
        "session_id",
        [
            "NEW_SESSION",  # Reserved keyword for new session requests
            "",  # Empty string
        ],
        ids=["new_session_keyword", "empty_id"],
    )
    def test_get_session_returns_none_for_special_ids(
        self, session_manager, session_id
    ):
        """Test get_session returns None for reserved/empty IDs without raising errors."""
        result = session_manager.get_session(session_id)
        assert result is None

    def test_get_session_raises_error_for_nonexistent_session(self, session_manager):
        """Test get_session raises ValueError for non-existent session."""
        with pytest.raises(ValueError, match="session not found"):
            session_manager.get_session("nonexistent-session-id")

    def test_get_session_returns_none_and_cleans_expired_session(self, session_manager):
        """Test get_session returns None and cleans up expired sessions."""
        # Create session with very short expiration
        session_manager.expiration = 0.1
        session = session_manager.create_session()
        session_id = session.session_id

        # Wait for expiration
        time.sleep(0.2)

        # Try to get expired session
        result = session_manager.get_session(session_id)

        assert result is None
        assert session_id not in session_manager.sessions
        assert not os.path.exists(session.files_path)

    def test_close_session_removes_from_registry(self, session_manager):
        """Test close_session removes session from registry."""
        session = session_manager.create_session()
        session_id = session.session_id

        session_manager.close_session(session_id)

        assert session_id not in session_manager.sessions

    def test_close_session_deletes_directory(self, session_manager):
        """Test close_session deletes the session directory."""
        session = session_manager.create_session()
        session_id = session.session_id
        session_path = session.files_path

        assert os.path.exists(session_path)

        session_manager.close_session(session_id)

        assert not os.path.exists(session_path)

    @pytest.mark.parametrize(
        "session_id,error_message",
        [
            ("", "invalid session_id"),  # Empty string validation
            (None, "invalid session_id"),  # None validation
            ("nonexistent-session-id", "session not found"),  # Non-existent session
        ],
        ids=["empty_id", "none_id", "nonexistent_id"],
    )
    def test_close_session_raises_error(
        self, session_manager, session_id, error_message
    ):
        """Test close_session raises ValueError for invalid/missing session IDs."""
        with pytest.raises(ValueError, match=error_message):
            session_manager.close_session(session_id)

    def test_clean_expired_session_removes_expired_sessions(
        self, session_manager, temp_session_storage
    ):
        """Test _clean_expired_session removes expired sessions."""
        # Create an expired session manually
        expired_session_id = "expired-session"
        expired_session_dir = os.path.join(temp_session_storage, expired_session_id)
        os.makedirs(expired_session_dir)

        expired_session = Session(
            expired_session_id, temp_session_storage, time.time() - 100
        )
        session_manager.sessions[expired_session_id] = expired_session

        # Create a valid session
        valid_session = session_manager.create_session()

        # Clean expired sessions
        session_manager._clean_expired_session()

        # Expired session should be removed
        assert expired_session_id not in session_manager.sessions
        assert not os.path.exists(expired_session_dir)

        # Valid session should remain
        assert valid_session.session_id in session_manager.sessions

    def test_create_session_triggers_cleanup(
        self, session_manager, temp_session_storage
    ):
        """Test create_session triggers cleanup of expired sessions."""
        # Create an expired session manually
        expired_session_id = "expired-session"
        expired_session_dir = os.path.join(temp_session_storage, expired_session_id)
        os.makedirs(expired_session_dir)

        expired_session = Session(
            expired_session_id, temp_session_storage, time.time() - 100
        )
        session_manager.sessions[expired_session_id] = expired_session

        # Create new session (should trigger cleanup)
        new_session = session_manager.create_session()

        # Expired session should be removed
        assert expired_session_id not in session_manager.sessions
        # New session should exist
        assert new_session.session_id in session_manager.sessions

    def test_thread_safety_concurrent_create_sessions(self, session_manager):
        """Test thread safety when creating sessions concurrently."""
        session_ids = []

        def create_session_thread():
            session = session_manager.create_session()
            session_ids.append(session.session_id)

        threads = [Thread(target=create_session_thread) for _ in range(10)]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # All session IDs should be unique
        assert len(session_ids) == len(set(session_ids))
        # All sessions should be in registry
        for session_id in session_ids:
            assert session_id in session_manager.sessions

    def test_thread_safety_concurrent_close_sessions(self, session_manager):
        """Test thread safety when closing sessions concurrently."""
        # Create multiple sessions
        sessions = [session_manager.create_session() for _ in range(5)]
        session_ids = [s.session_id for s in sessions]

        def close_session_thread(session_id):
            session_manager.close_session(session_id)

        threads = [
            Thread(target=close_session_thread, args=(sid,)) for sid in session_ids
        ]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # All sessions should be closed
        for session_id in session_ids:
            assert session_id not in session_manager.sessions

    def test_session_data_persists_across_get_calls(self, session_manager):
        """Test session data persists across multiple get_session calls."""
        session = session_manager.create_session()
        session_id = session.session_id

        # Store some data
        session.put("key1", "value1")
        session.put("key2", {"nested": "data"})

        # Retrieve session again
        retrieved_session = session_manager.get_session(session_id)

        # Data should persist
        assert retrieved_session.get("key1") == "value1"
        assert retrieved_session.get("key2") == {"nested": "data"}
