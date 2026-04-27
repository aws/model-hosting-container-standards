"""Pre-launch dependency installation from customer model artifacts.

Installs Python dependencies from a requirements.txt file found in the model
directory before the framework server process starts. Designed to be called
from the standard-supervisor CLI launcher.

Uses ``uv pip install`` when ``uv`` is available on PATH (significantly faster
dependency resolution), falling back to ``python -m pip install`` otherwise.

Environment Variables:
    STANDARD_AUTO_INSTALL_REQ: Enable/disable automatic installation (default: true).
    STANDARD_PIP_ARGS: Explicit install arguments. When set, replaces
        auto-discovery entirely and runs ``uv pip install <args>`` (or
        ``pip install <args>``). Use this to point to a custom requirements
        file (e.g. "-r /path/to/req.txt"), set custom indexes
        (e.g. "--index-url https://my-index/simple"), or pass any other
        pip install flags.
    SAGEMAKER_MODEL_PATH: Model artifact directory (default: /opt/ml/model).
"""

import logging
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "/opt/ml/model"
REQUIREMENTS_FILENAME = "requirements.txt"

# Environment variable names
STANDARD_AUTO_INSTALL_REQ = "STANDARD_AUTO_INSTALL_REQ"
STANDARD_PIP_ARGS = "STANDARD_PIP_ARGS"


def is_auto_install_enabled() -> bool:
    """Check if automatic dependency installation is enabled."""
    enabled = os.getenv(STANDARD_AUTO_INSTALL_REQ, "true")
    return enabled.lower() in ("true", "1", "yes", "on")


def resolve_python_from_command(launch_command: List[str]) -> str:
    """Extract the Python interpreter from a launch command.

    Inspects the launch command to determine which Python interpreter the
    framework will run under, so that pip installs packages into the correct
    site-packages.

    Resolution order:
        1. If the command starts with an explicit Python path or name
           (e.g. "python3", "/usr/bin/python3.12", "python"),
           return that value resolved to an absolute path via shutil.which
           if it is a bare name.
        2. Otherwise fall back to sys.executable (the Python running
           standard-supervisor itself).

    Args:
        launch_command: The launch command as a list of strings, e.g.
            ["python3", "-m", "vllm.entrypoints.openai.api_server", "--port", "8080"]
            or ["vllm", "serve", "model"].

    Returns:
        Absolute path to the Python interpreter to use for pip install.
    """
    if not launch_command:
        return sys.executable

    first = launch_command[0]
    basename = os.path.basename(first)

    # Match python, python3, python3.12, etc.
    if basename == "python" or basename.startswith("python3"):
        resolved = shutil.which(first)
        if resolved:
            logger.debug(
                "Resolved Python from launch command: %s -> %s", first, resolved
            )
            return resolved
        # If which() fails (e.g. absolute path that exists but isn't on PATH)
        if os.path.isfile(first) and os.access(first, os.X_OK):
            logger.debug("Using Python from launch command directly: %s", first)
            return first

    # Not an explicit python invocation (e.g. "vllm serve ...")
    logger.debug(
        "No Python interpreter found in launch command, using sys.executable: %s",
        sys.executable,
    )
    return sys.executable


def _build_install_prefix(python: str) -> List[str]:
    """Build the install command prefix, preferring uv over pip.

    Uses ``uv pip install --python <python>`` if ``uv`` is on PATH,
    otherwise falls back to ``<python> -m pip install``. If pip is also
    missing, the subsequent subprocess call will fail and
    ``install_requirements`` handles the error.

    Args:
        python: Path to the target Python interpreter.

    Returns:
        Command prefix list.
    """
    if shutil.which("uv"):
        logger.debug("Using uv for dependency installation")
        return ["uv", "pip", "install", "--python", python]

    logger.debug("uv not found, falling back to pip")
    return [python, "-m", "pip", "install"]


def install_requirements(
    model_path: Optional[str] = None,
    requirements_filename: str = REQUIREMENTS_FILENAME,
    pip_args: Optional[str] = None,
    python_executable: Optional[str] = None,
) -> bool:
    """Install Python dependencies from requirements.txt or explicit args.

    Uses ``uv pip install`` when available, falling back to ``pip install``.

    Two modes of operation:

    1. **Auto-discovery** (default, no STANDARD_PIP_ARGS):
       Looks for requirements_filename in model_path. If found, runs
       ``uv pip install -r <file>``. Also checks for a ``requirements/``
       subdirectory to support offline installs via ``--find-links``.

    2. **Explicit** (STANDARD_PIP_ARGS is set):
       Runs ``uv pip install <args>`` using exactly the customer-provided
       arguments. Auto-discovery is skipped entirely to avoid duplicate
       ``-r`` flags pointing to the same file.

    Args:
        model_path: Model artifact directory for auto-discovery. Defaults to
                    SAGEMAKER_MODEL_PATH env var, then /opt/ml/model.
        requirements_filename: Filename to look for during auto-discovery.
        pip_args: Explicit pip install arguments. When set, replaces
                        auto-discovery entirely. Parsed with shlex.split.
        python_executable: Python interpreter to use for pip. Defaults to
                           sys.executable.

    Returns:
        True if installation succeeded or was skipped (no file found).
        False if installation failed.
    """
    python = python_executable or sys.executable

    if pip_args:
        # Explicit mode: customer owns the install arguments entirely.
        # No auto-discovery to avoid duplicate -r flags.
        try:
            parsed_args = shlex.split(pip_args)
        except ValueError as e:
            logger.error("Invalid STANDARD_PIP_ARGS: %s", e)
            return False
        install_prefix = _build_install_prefix(python)
        cmd = install_prefix + parsed_args
        logger.info("Installing dependencies from explicit pip args...")
    else:
        # Auto-discovery mode: look for requirements.txt in model directory.
        model_dir_path = (
            model_path or os.environ.get("SAGEMAKER_MODEL_PATH") or DEFAULT_MODEL_PATH
        )
        model_dir = Path(model_dir_path)
        req_file = model_dir / requirements_filename

        if not req_file.is_file():
            logger.debug("No %s found in %s", requirements_filename, model_dir)
            return True

        install_prefix = _build_install_prefix(python)

        cmd = install_prefix + ["-r", str(req_file)]

        # Support offline installs via local packages directory
        local_packages = model_dir / "requirements"
        if local_packages.is_dir():
            cmd.extend(["--find-links", str(local_packages)])
            logger.info("Using local packages directory: %s", local_packages)

        logger.info("Installing dependencies from %s ...", req_file)

    logger.debug("pip command: %s", cmd)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("Dependency installation completed successfully")
            if result.stdout:
                logger.debug("pip output:\n%s", result.stdout)
            return True
        else:
            logger.error(
                "Dependency installation failed (exit %d):\n%s",
                result.returncode,
                result.stderr,
            )
            return False
    except Exception:
        logger.exception("Unexpected error during dependency installation")
        return False
