# Adapted from deepjavalibrary/djl-serving on github
# https://github.com/deepjavalibrary/djl-serving/blob/master/engines/python/setup/djl_python/session_manager.py

import json
import logging
import os
import shutil
import tempfile
import time
import uuid
from threading import RLock
from typing import Dict, Optional

from ..config import get_configs_from_env_vars


class Session:
    """Represents a single stateful session with file-based storage.

    Each session has a unique directory where it stores key-value data
    as JSON files. Sessions have an expiration timestamp and are automatically
    cleaned up when expired.
    """

    def __init__(
        self, session_id: str, session_root: str, expiration_ts: Optional[float] = None
    ):
        """Initialize a session instance.

        Args:
            session_id: Unique identifier for this session (typically UUID)
            session_root: Root directory where session directories are stored
            expiration_ts: Unix timestamp when session expires, or None to load from disk
        """
        self.session_id = session_id
        self.files_path = os.path.join(session_root, session_id)
        self.expiration_ts = expiration_ts
        # If expiration not provided, try to load it from session file
        if self.expiration_ts is None:
            self.expiration_ts = self.get(".expiration_ts")

    def put(self, key: str, value):
        """Store a JSON-serializable value in the session.

        Args:
            key: The key to store the value under
            value: Must be JSON-serializable (str, int, float, bool, list, dict, None)

        Raises:
            TypeError: If value is not JSON-serializable
        """
        with open(self._path(key), "w") as f:
            json.dump(value, f)

    def get(self, key: str, d=None):
        """Retrieve a value from the session.

        Args:
            key: The key to retrieve
            d: Default value if key doesn't exist

        Returns:
            The stored value or default if key not found
        """
        path = self._path(key)
        if not os.path.isfile(path):
            return d

        with open(path, "r") as f:
            return json.load(f)

    def remove(self):
        """Delete the session and all its stored data from disk.

        Returns:
            bool: True if deletion successful

        Raises:
            ValueError: If session directory doesn't exist
        """
        if not os.path.exists(self.files_path):
            raise ValueError(f"session directory does not exist: {self.session_id}")
        logging.info(f"closing session: {self.session_id}")
        shutil.rmtree(self.files_path)
        return True

    def _path(self, key: str):
        """Generate a safe file path within the session directory.

        Args:
            key: The key name for the session data file

        Returns:
            Absolute path to the file within the session directory

        Raises:
            ValueError: If key contains path traversal attempts
        """
        # Validate against path traversal attacks
        if ".." in key:
            raise ValueError(f"Invalid key: '..' not allowed in key '{key}'")
        if os.path.isabs(key):
            raise ValueError(f"Invalid key: absolute paths not allowed '{key}'")

        # Sanitize the key by replacing slashes
        sanitized_key = key.replace("/", "-")
        file_path = os.path.join(self.files_path, sanitized_key)

        # Double-check the resolved path is still within session directory
        resolved_path = os.path.abspath(file_path)
        if not resolved_path.startswith(os.path.abspath(self.files_path) + os.sep):
            raise ValueError(f"Invalid key: path traversal detected '{key}'")

        return file_path


class SessionManager:
    """Manages the lifecycle of stateful sessions with automatic expiration and cleanup.

    SessionManager maintains a registry of active sessions, each stored in its own
    directory on disk. It handles session creation, retrieval, expiration checking,
    and cleanup. Thread-safe for concurrent access.
    """

    def __init__(self, properties: dict):
        """Initialize the SessionManager with configuration properties.

        Args:
            properties: Configuration dict with optional keys:
                - sessions_expiration: Session lifetime in seconds (default: 1200)
                - sessions_path: Root directory for session storage (default: /dev/shm/sagemaker_sessions)
        """
        # Session expiration time in seconds (default: 20 minutes)
        self.expiration = int(properties.get("sessions_expiration", str(20 * 60)))

        # Prefer /dev/shm (memory-backed filesystem) for performance
        if os.path.exists("/dev/shm"):
            session_dir = "/dev/shm/sagemaker_sessions"
        else:
            session_dir = os.path.join(tempfile.gettempdir(), "sagemaker_sessions")

        self.sessions_path = properties.get("sessions_path", session_dir)
        self.sessions: dict[str, Session] = {}
        self._lock = RLock()  # Thread safety for concurrent session access

        # Create sessions directory if it doesn't exist
        if not os.path.exists(self.sessions_path):
            os.makedirs(self.sessions_path, exist_ok=True)

        # Load any previously saved sessions from disk (for persistence across restarts)
        for session_id in os.listdir(self.sessions_path):
            self.sessions[session_id] = Session(session_id, self.sessions_path)

    def create_session(self) -> Session:
        """Create a new session with a unique ID and expiration timestamp.

        Also triggers cleanup of any expired sessions before creating the new one.

        Returns:
            Session: The newly created session instance with UUID and expiration

        Thread-safe: Uses internal lock for concurrent access
        """
        with self._lock:
            # Clean up expired sessions before creating new one
            self._clean_expired_session()

            # Generate unique session ID
            session_id = str(uuid.uuid4())
            expiration_ts = time.time() + self.expiration

            # Create and register the session
            session = Session(session_id, self.sessions_path, expiration_ts)
            self.sessions[session_id] = session

            # Create session directory and persist expiration
            os.makedirs(session.files_path)
            session.put(".expiration_ts", expiration_ts)

            return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """Retrieve a session by ID, checking for expiration.

        Args:
            session_id: The unique session identifier

        Returns:
            Session instance if found and not expired, None if session doesn't exist
            or has expired, or if session_id is "NEW_SESSION" or empty

        Raises:
            ValueError: If session_id is not found in registry

        Thread-safe: Uses internal lock for concurrent access
        """
        with self._lock:
            if session_id == "NEW_SESSION" or not session_id:
                return None

            if session_id not in self.sessions:
                raise ValueError(f"session not found: {session_id}")
            session = self.sessions[session_id]

            # Session expired - clean it up immediately to prevent memory leak
            if (
                session.expiration_ts is not None
                and time.time() > session.expiration_ts
            ):
                logging.info(f"Session expired: {session_id}")
                self.close_session(session_id)
                return None

            return session

    def close_session(self, session_id):
        """Close and remove a session, deleting all its data.

        Args:
            session_id: The unique session identifier to close

        Raises:
            ValueError: If session_id is empty/None or not found in registry

        Thread-safe: Uses internal lock for concurrent access
        """
        with self._lock:
            if not session_id:
                raise ValueError(f"invalid session_id: {session_id}")

            if session_id not in self.sessions:
                raise ValueError(f"session not found: {session_id}")
            session = self.sessions[session_id]

            session.remove()

            del self.sessions[session_id]

    def _clean_expired_session(self):
        """Internal method to remove all expired sessions.

        Iterates through all sessions and closes any that have expired.
        Called automatically during session creation to prevent stale session buildup.

        Thread-safe: Uses internal lock for concurrent access
        """
        with self._lock:
            for session_id, session in list(self.sessions.items()):
                if session.expiration_ts is None or time.time() > session.expiration_ts:
                    self.close_session(session_id)


def _init_session_manager(sessions_configs: Dict[str, str]) -> SessionManager | None:
    enable_stateful_sessions = sessions_configs.get(
        "enable_stateful_sessions", "false"
    ).lower()
    if enable_stateful_sessions == "true":
        return SessionManager(sessions_configs)
    return None


_sessions_configs = get_configs_from_env_vars()
# Global SessionManager instance
session_manager = _init_session_manager(_sessions_configs)
