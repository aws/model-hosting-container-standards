"""Unit tests for dependency_manager module."""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from model_hosting_container_standards.common.dependency_manager import (
    REQUIREMENTS_FILENAME,
    STANDARD_AUTO_INSTALL_REQ,
    _build_install_prefix,
    install_requirements,
    is_auto_install_enabled,
    resolve_python_from_command,
)

# Patch target for shutil.which inside dependency_manager module
_WHICH_PATCH = (
    "model_hosting_container_standards.common.dependency_manager.shutil.which"
)


def _mock_which_has_uv(cmd):
    """Mock shutil.which that reports uv as available."""
    if cmd == "uv":
        return "/usr/bin/uv"
    return None


class TestInstallRequirements:
    """Tests for install_requirements function."""

    def test_no_requirements_file_returns_true(self, tmp_path):
        """No requirements.txt present — should succeed silently."""
        result = install_requirements(model_path=str(tmp_path))
        assert result is True

    def test_successful_install_with_uv(self, tmp_path):
        """requirements.txt exists, uv available — should use uv pip install."""
        req_file = tmp_path / REQUIREMENTS_FILENAME
        req_file.write_text("requests==2.31.0\n")

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Successfully installed requests-2.31.0"

        with patch(_WHICH_PATCH, side_effect=_mock_which_has_uv):
            with patch("subprocess.run", return_value=mock_result) as mock_run:
                result = install_requirements(model_path=str(tmp_path))

        assert result is True
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[:3] == ["uv", "pip", "install"]
        assert "--python" in cmd
        assert "-r" in cmd
        assert str(req_file) in cmd

    def test_successful_install_without_uv(self, tmp_path):
        """requirements.txt exists, uv not available — should fall back to pip."""
        req_file = tmp_path / REQUIREMENTS_FILENAME
        req_file.write_text("requests==2.31.0\n")

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Successfully installed requests-2.31.0"

        with patch(_WHICH_PATCH, return_value=None):
            with patch("subprocess.run", return_value=mock_result) as mock_run:
                result = install_requirements(model_path=str(tmp_path))

        assert result is True
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == sys.executable
        assert cmd[1:4] == ["-m", "pip", "install"]
        assert "-r" in cmd
        assert str(req_file) in cmd

    def test_failed_install_returns_false(self, tmp_path):
        """pip install fails — should return False."""
        req_file = tmp_path / REQUIREMENTS_FILENAME
        req_file.write_text("nonexistent-package-xyz\n")

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "ERROR: No matching distribution"

        with patch("subprocess.run", return_value=mock_result):
            result = install_requirements(model_path=str(tmp_path))

        assert result is False

    def test_unexpected_exception_returns_false(self, tmp_path):
        """Unexpected error during install — should return False."""
        req_file = tmp_path / REQUIREMENTS_FILENAME
        req_file.write_text("some-package\n")

        with patch("subprocess.run", side_effect=OSError("disk full")):
            result = install_requirements(model_path=str(tmp_path))

        assert result is False

    def test_local_packages_directory_adds_find_links(self, tmp_path):
        """requirements/ subdirectory present — should add --find-links."""
        req_file = tmp_path / REQUIREMENTS_FILENAME
        req_file.write_text("my-package\n")
        local_dir = tmp_path / "requirements"
        local_dir.mkdir()

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            result = install_requirements(model_path=str(tmp_path))

        assert result is True
        cmd = mock_run.call_args[0][0]
        assert "--find-links" in cmd
        find_links_idx = cmd.index("--find-links")
        assert cmd[find_links_idx + 1] == str(local_dir)

    def test_pip_args_replaces_auto_discovery(self, tmp_path):
        """STANDARD_PIP_ARGS replaces auto-discovery entirely."""
        req_file = tmp_path / REQUIREMENTS_FILENAME
        req_file.write_text("some-package\n")

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            result = install_requirements(
                model_path=str(tmp_path),
                pip_args="-r /custom/requirements.txt --no-cache-dir",
            )

        assert result is True
        cmd = mock_run.call_args[0][0]
        # Should NOT contain the auto-discovered file
        assert str(req_file) not in cmd
        # Should contain the customer's args
        assert "-r" in cmd
        assert "/custom/requirements.txt" in cmd
        assert "--no-cache-dir" in cmd

    def test_pip_args_skips_find_links(self, tmp_path):
        """When extra args are set, auto-discovered --find-links is also skipped."""
        req_file = tmp_path / REQUIREMENTS_FILENAME
        req_file.write_text("some-package\n")
        local_dir = tmp_path / "requirements"
        local_dir.mkdir()

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            result = install_requirements(
                model_path=str(tmp_path),
                pip_args="-r /custom/requirements.txt",
            )

        assert result is True
        cmd = mock_run.call_args[0][0]
        # Should NOT contain auto-discovered --find-links
        assert "--find-links" not in cmd
        assert str(local_dir) not in cmd

    def test_pip_args_works_without_requirements_file(self, tmp_path):
        """Explicit pip args should work even when no requirements.txt exists in model dir."""
        # No requirements.txt in tmp_path

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""

        with patch("subprocess.run", return_value=mock_result):
            result = install_requirements(
                model_path=str(tmp_path),
                pip_args="-r /custom/requirements.txt",
            )

        assert result is True

    def test_malformed_pip_args_returns_false(self, tmp_path):
        """Malformed quoting in pip_args should return False, not crash."""
        with patch("subprocess.run") as mock_run:
            result = install_requirements(
                model_path=str(tmp_path),
                pip_args='--index-url "https://unclosed-quote',
            )

        assert result is False
        mock_run.assert_not_called()

    def test_pip_args_none_uses_auto_discovery(self, tmp_path):
        """pip_args=None should use auto-discovery from model dir."""
        req_file = tmp_path / REQUIREMENTS_FILENAME
        req_file.write_text("some-package\n")

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""

        with patch(_WHICH_PATCH, return_value=None):
            with patch("subprocess.run", return_value=mock_result) as mock_run:
                result = install_requirements(model_path=str(tmp_path), pip_args=None)

        assert result is True
        cmd = mock_run.call_args[0][0]
        # pip fallback: [python, -m, pip, install, -r, /path/to/requirements.txt]
        assert len(cmd) == 6
        assert str(req_file) in cmd

    def test_uses_sagemaker_model_path_env(self, tmp_path):
        """Should fall back to SAGEMAKER_MODEL_PATH env var."""
        req_file = tmp_path / REQUIREMENTS_FILENAME
        req_file.write_text("some-package\n")

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""

        with patch.dict(os.environ, {"SAGEMAKER_MODEL_PATH": str(tmp_path)}):
            with patch("subprocess.run", return_value=mock_result) as mock_run:
                result = install_requirements()

        assert result is True
        cmd = mock_run.call_args[0][0]
        assert str(req_file) in cmd

    def test_explicit_model_path_overrides_env(self, tmp_path):
        """Explicit model_path arg should take precedence over env var."""
        env_dir = tmp_path / "env_dir"
        env_dir.mkdir()
        explicit_dir = tmp_path / "explicit_dir"
        explicit_dir.mkdir()
        req_file = explicit_dir / REQUIREMENTS_FILENAME
        req_file.write_text("some-package\n")

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""

        with patch.dict(os.environ, {"SAGEMAKER_MODEL_PATH": str(env_dir)}):
            with patch("subprocess.run", return_value=mock_result) as mock_run:
                result = install_requirements(model_path=str(explicit_dir))

        assert result is True
        cmd = mock_run.call_args[0][0]
        assert str(req_file) in cmd

    def test_python_executable_used_with_uv(self, tmp_path):
        """Explicit python_executable should be passed via --python when uv is available."""
        req_file = tmp_path / REQUIREMENTS_FILENAME
        req_file.write_text("some-package\n")

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""

        custom_python = "/opt/venv/bin/python3.12"

        with patch(_WHICH_PATCH, side_effect=_mock_which_has_uv):
            with patch("subprocess.run", return_value=mock_result) as mock_run:
                result = install_requirements(
                    model_path=str(tmp_path),
                    python_executable=custom_python,
                )

        assert result is True
        cmd = mock_run.call_args[0][0]
        assert cmd[:3] == ["uv", "pip", "install"]
        python_idx = cmd.index("--python")
        assert cmd[python_idx + 1] == custom_python

    def test_python_executable_used_without_uv(self, tmp_path):
        """Explicit python_executable should be cmd[0] when uv is not available."""
        req_file = tmp_path / REQUIREMENTS_FILENAME
        req_file.write_text("some-package\n")

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""

        custom_python = "/opt/venv/bin/python3.12"

        with patch(_WHICH_PATCH, return_value=None):
            with patch("subprocess.run", return_value=mock_result) as mock_run:
                result = install_requirements(
                    model_path=str(tmp_path),
                    python_executable=custom_python,
                )

        assert result is True
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == custom_python
        assert cmd[1:4] == ["-m", "pip", "install"]

    def test_python_executable_none_falls_back_to_sys(self, tmp_path):
        """python_executable=None should use sys.executable."""
        req_file = tmp_path / REQUIREMENTS_FILENAME
        req_file.write_text("some-package\n")

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""

        with patch(_WHICH_PATCH, return_value=None):
            with patch("subprocess.run", return_value=mock_result) as mock_run:
                result = install_requirements(
                    model_path=str(tmp_path),
                    python_executable=None,
                )

        assert result is True
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == sys.executable


class TestBuildInstallPrefix:
    """Tests for _build_install_prefix function."""

    def test_uses_uv_when_available(self):
        """Should use uv pip install --python when uv is on PATH."""
        with patch(_WHICH_PATCH, side_effect=_mock_which_has_uv):
            prefix = _build_install_prefix("/usr/bin/python3")
        assert prefix == ["uv", "pip", "install", "--python", "/usr/bin/python3"]

    def test_falls_back_to_pip_when_no_uv(self):
        """Should use python -m pip install when uv is not on PATH."""
        with patch(_WHICH_PATCH, return_value=None):
            prefix = _build_install_prefix("/usr/bin/python3")
        assert prefix == ["/usr/bin/python3", "-m", "pip", "install"]


class TestIsAutoInstallEnabled:
    """Tests for is_auto_install_enabled function."""

    def test_enabled_by_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(STANDARD_AUTO_INSTALL_REQ, None)
            assert is_auto_install_enabled() is True

    @pytest.mark.parametrize("value", ["true", "True", "TRUE", "1", "yes", "on"])
    def test_enabled_truthy_values(self, value):
        with patch.dict(os.environ, {STANDARD_AUTO_INSTALL_REQ: value}):
            assert is_auto_install_enabled() is True

    @pytest.mark.parametrize("value", ["false", "False", "FALSE", "0", "no", "off"])
    def test_disabled_falsy_values(self, value):
        with patch.dict(os.environ, {STANDARD_AUTO_INSTALL_REQ: value}):
            assert is_auto_install_enabled() is False


class TestInstallRequirementsNoInstaller:
    """Tests for install_requirements when no installer is available."""

    def test_returns_false_when_pip_missing(self, tmp_path):
        """Should return False when pip is not available and install fails."""
        req_file = tmp_path / REQUIREMENTS_FILENAME
        req_file.write_text("some-package\n")

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "No module named pip"

        with patch(_WHICH_PATCH, return_value=None):
            with patch("subprocess.run", return_value=mock_result):
                result = install_requirements(model_path=str(tmp_path))

        assert result is False

    def test_no_installer_check_skipped_when_no_requirements(self, tmp_path):
        """Should not probe for installer when no requirements.txt exists."""
        with patch(
            "model_hosting_container_standards.common.dependency_manager._build_install_prefix"
        ) as mock_prefix:
            result = install_requirements(model_path=str(tmp_path))

        assert result is True
        mock_prefix.assert_not_called()


class TestResolvePythonFromCommand:
    """Tests for resolve_python_from_command function."""

    def test_explicit_python3(self):
        """'python3 -m vllm ...' should resolve python3."""
        with patch(_WHICH_PATCH, return_value="/usr/bin/python3"):
            result = resolve_python_from_command(
                [
                    "python3",
                    "-m",
                    "vllm.entrypoints.openai.api_server",
                    "--port",
                    "8080",
                ]
            )
        assert result == "/usr/bin/python3"

    def test_explicit_python(self):
        """'python -m module' should resolve python."""
        with patch(_WHICH_PATCH, return_value="/usr/bin/python"):
            result = resolve_python_from_command(["python", "-m", "some_module"])
        assert result == "/usr/bin/python"

    def test_versioned_python(self):
        """'python3.12 -m module' should resolve python3.12."""
        with patch(_WHICH_PATCH, return_value="/usr/bin/python3.12"):
            result = resolve_python_from_command(["python3.12", "-m", "some_module"])
        assert result == "/usr/bin/python3.12"

    def test_absolute_python_path(self):
        """'/opt/venv/bin/python3 -m module' should use the absolute path."""
        abs_path = "/opt/venv/bin/python3"
        with patch(_WHICH_PATCH, return_value=abs_path):
            result = resolve_python_from_command([abs_path, "-m", "some_module"])
        assert result == abs_path

    def test_absolute_path_not_on_path_but_exists(self, tmp_path):
        """Absolute python path that exists but isn't found by which()."""
        fake_python = tmp_path / "python3"
        fake_python.write_text("#!/bin/sh\n")
        fake_python.chmod(0o755)

        with patch(_WHICH_PATCH, return_value=None):
            result = resolve_python_from_command([str(fake_python), "-m", "module"])
        assert result == str(fake_python)

    def test_non_python_command_falls_back(self):
        """'vllm serve model' should fall back to sys.executable."""
        result = resolve_python_from_command(["vllm", "serve", "model"])
        assert result == sys.executable

    def test_empty_command_falls_back(self):
        """Empty launch command should fall back to sys.executable."""
        result = resolve_python_from_command([])
        assert result == sys.executable

    def test_which_fails_and_not_a_file(self):
        """python3 not found by which() and not a file — falls back to sys.executable."""
        with patch(_WHICH_PATCH, return_value=None):
            result = resolve_python_from_command(["python3", "-m", "module"])
        # python3 is not an absolute path to a file, so falls back
        assert result == sys.executable


class TestSupervisorIntegration:
    """Tests for standard_supervisor.py integration with dependency_manager."""

    def test_install_dependencies_called_with_launch_command(self):
        """_install_dependencies should pass launch_command to resolve_python_from_command."""
        from model_hosting_container_standards.supervisor.scripts.standard_supervisor import (
            StandardSupervisor,
        )

        supervisor = StandardSupervisor()
        launch_cmd = ["python3", "-m", "vllm.entrypoints.openai.api_server"]

        with patch(
            "model_hosting_container_standards.common.dependency_manager.is_auto_install_enabled",
            return_value=True,
        ):
            with patch(
                "model_hosting_container_standards.common.dependency_manager.resolve_python_from_command",
                return_value="/usr/bin/python3",
            ) as mock_resolve:
                with patch(
                    "model_hosting_container_standards.common.dependency_manager.install_requirements",
                    return_value=True,
                ):
                    result = supervisor._install_dependencies(launch_cmd)

        assert result is True
        mock_resolve.assert_called_once_with(launch_cmd)

    def test_install_dependencies_aborts_on_failure(self):
        """_install_dependencies should return False when install fails."""
        from model_hosting_container_standards.supervisor.scripts.standard_supervisor import (
            StandardSupervisor,
        )

        supervisor = StandardSupervisor()

        with patch(
            "model_hosting_container_standards.common.dependency_manager.is_auto_install_enabled",
            return_value=True,
        ):
            with patch(
                "model_hosting_container_standards.common.dependency_manager.resolve_python_from_command",
                return_value="/usr/bin/python3",
            ):
                with patch(
                    "model_hosting_container_standards.common.dependency_manager.install_requirements",
                    return_value=False,
                ):
                    result = supervisor._install_dependencies(["python3", "-m", "vllm"])

        assert result is False

    def test_install_dependencies_skipped_when_disabled(self):
        """_install_dependencies should skip when STANDARD_AUTO_INSTALL_REQ=false."""
        from model_hosting_container_standards.supervisor.scripts.standard_supervisor import (
            StandardSupervisor,
        )

        supervisor = StandardSupervisor()

        with patch(
            "model_hosting_container_standards.common.dependency_manager.is_auto_install_enabled",
            return_value=False,
        ):
            with patch(
                "model_hosting_container_standards.common.dependency_manager.install_requirements",
            ) as mock_install:
                result = supervisor._install_dependencies(["python3", "-m", "vllm"])

        assert result is True
        mock_install.assert_not_called()

    def test_run_aborts_when_install_fails(self):
        """StandardSupervisor.run() should return 1 when dependency install fails."""
        from model_hosting_container_standards.supervisor.scripts.standard_supervisor import (
            StandardSupervisor,
        )

        supervisor = StandardSupervisor()

        with patch.object(
            sys, "argv", ["standard-supervisor", "python3", "-m", "vllm"]
        ):
            with patch(
                "model_hosting_container_standards.supervisor.scripts.standard_supervisor.parse_environment_variables",
            ):
                with patch.object(
                    supervisor, "_install_dependencies", return_value=False
                ):
                    result = supervisor.run()

        assert result == 1
