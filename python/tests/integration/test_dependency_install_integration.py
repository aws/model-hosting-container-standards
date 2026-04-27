"""
Integration tests for automatic dependency installation in standard-supervisor.

Tests verify real end-to-end behavior using isolated virtual environments
to avoid polluting the host Python and to prevent false positives from
pre-existing packages.

Tests cover:
1. Auto-discovery of requirements.txt and actual install
2. STANDARD_PIP_ARGS explicit mode
3. STANDARD_AUTO_INSTALL_REQ=false disables installation
4. Bad package blocks startup with non-zero exit
5. Malformed STANDARD_PIP_ARGS returns error
6. Python interpreter resolution from launch command
"""

import os
import subprocess
import textwrap
import venv
from pathlib import Path


def get_python_cwd():
    """Get the correct working directory for python module execution."""
    return str(Path(__file__).parent.parent.parent.absolute())


def create_test_venv(venv_dir):
    """Create a venv with model-hosting-container-standards installed.

    Returns the path to the venv's python executable.
    """
    venv.create(venv_dir, with_pip=True, clear=True)
    venv_python = str(venv_dir / "bin" / "python")

    # Install the local package into the venv
    subprocess.run(
        [venv_python, "-m", "pip", "install", "-q", "-e", "."],
        cwd=get_python_cwd(),
        check=True,
        capture_output=True,
    )
    return venv_python


def run_supervisor(venv_python, launch_cmd, env_overrides=None, timeout=120):
    """Run standard-supervisor from a venv as a subprocess.

    Uses PROCESS_AUTO_RECOVERY=false so standard-supervisor does execvp
    (direct launch, no supervisord), keeping the test simple and fast.
    """
    env = {
        **os.environ,
        "PROCESS_AUTO_RECOVERY": "false",
        "LOG_LEVEL": "debug",
    }
    if env_overrides:
        env.update(env_overrides)

    cmd = [
        venv_python,
        "-m",
        "model_hosting_container_standards.supervisor.scripts.standard_supervisor",
    ] + launch_cmd

    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=get_python_cwd(),
        env=env,
    )


class TestAutoDiscovery:
    """Tests for auto-discovery of requirements.txt."""

    def test_installs_from_requirements_txt(self, tmp_path):
        """A real requirements.txt should be installed and the package importable."""
        venv_python = create_test_venv(tmp_path / "venv")

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "requirements.txt").write_text("six==1.17.0\n")

        # Verify six is NOT already importable in the venv
        pre_check = subprocess.run(
            [venv_python, "-c", "import six"],
            capture_output=True,
        )
        assert pre_check.returncode != 0, "six should not be pre-installed in venv"

        script = textwrap.dedent(
            """\
            import six
            print(f"six version: {six.__version__}", flush=True)
        """
        )
        script_file = tmp_path / "verify.py"
        script_file.write_text(script)

        result = run_supervisor(
            venv_python,
            [venv_python, str(script_file)],
            env_overrides={"SAGEMAKER_MODEL_PATH": str(model_dir)},
        )

        assert (
            result.returncode == 0
        ), f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        assert "six version: 1.17.0" in result.stdout

    def test_no_requirements_file_is_noop(self, tmp_path):
        """No requirements.txt — should proceed without error."""
        venv_python = create_test_venv(tmp_path / "venv")

        model_dir = tmp_path / "model"
        model_dir.mkdir()

        script_file = tmp_path / "verify.py"
        script_file.write_text('print("started ok", flush=True)\n')

        result = run_supervisor(
            venv_python,
            [venv_python, str(script_file)],
            env_overrides={"SAGEMAKER_MODEL_PATH": str(model_dir)},
        )

        assert result.returncode == 0
        assert "started ok" in result.stdout

    def test_bad_package_fails_startup(self, tmp_path):
        """A requirements.txt with a nonexistent package should fail with non-zero exit."""
        venv_python = create_test_venv(tmp_path / "venv")

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "requirements.txt").write_text(
            "this-package-does-not-exist-xyz-12345\n"
        )

        script_file = tmp_path / "should_not_run.py"
        script_file.write_text("print('should not reach here')\n")

        result = run_supervisor(
            venv_python,
            [venv_python, str(script_file)],
            env_overrides={"SAGEMAKER_MODEL_PATH": str(model_dir)},
        )

        assert result.returncode != 0
        assert "should not reach here" not in result.stdout


class TestDisableInstallation:
    """Tests for STANDARD_AUTO_INSTALL_REQ=false."""

    def test_disabled_skips_install(self, tmp_path):
        """With auto-install disabled, a bad requirements.txt should not block startup."""
        venv_python = create_test_venv(tmp_path / "venv")

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "requirements.txt").write_text(
            "this-package-does-not-exist-xyz-12345\n"
        )

        script_file = tmp_path / "verify.py"
        script_file.write_text('print("started ok", flush=True)\n')

        result = run_supervisor(
            venv_python,
            [venv_python, str(script_file)],
            env_overrides={
                "SAGEMAKER_MODEL_PATH": str(model_dir),
                "STANDARD_AUTO_INSTALL_REQ": "false",
            },
        )

        assert result.returncode == 0
        assert "started ok" in result.stdout


class TestExplicitPipArgs:
    """Tests for STANDARD_PIP_ARGS explicit mode."""

    def test_explicit_args_used_instead_of_auto_discovery(self, tmp_path):
        """STANDARD_PIP_ARGS should be used instead of auto-discovered file."""
        venv_python = create_test_venv(tmp_path / "venv")

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        # Bad requirements.txt in model dir — should be ignored
        (model_dir / "requirements.txt").write_text(
            "this-package-does-not-exist-xyz-12345\n"
        )

        # Good requirements file elsewhere
        good_req = tmp_path / "custom_req.txt"
        good_req.write_text("six==1.17.0\n")

        script = textwrap.dedent(
            """\
            import six
            print(f"six version: {six.__version__}", flush=True)
        """
        )
        script_file = tmp_path / "verify.py"
        script_file.write_text(script)

        result = run_supervisor(
            venv_python,
            [venv_python, str(script_file)],
            env_overrides={
                "SAGEMAKER_MODEL_PATH": str(model_dir),
                "STANDARD_PIP_ARGS": f"-r {good_req}",
            },
        )

        assert (
            result.returncode == 0
        ), f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        assert "six version: 1.17.0" in result.stdout

    def test_malformed_pip_args_fails(self, tmp_path):
        """Malformed quoting in STANDARD_PIP_ARGS should fail, not crash."""
        venv_python = create_test_venv(tmp_path / "venv")

        model_dir = tmp_path / "model"
        model_dir.mkdir()

        script_file = tmp_path / "should_not_run.py"
        script_file.write_text("print('should not reach here')\n")

        result = run_supervisor(
            venv_python,
            [venv_python, str(script_file)],
            env_overrides={
                "SAGEMAKER_MODEL_PATH": str(model_dir),
                "STANDARD_PIP_ARGS": '--index-url "https://unclosed-quote',
            },
        )

        assert result.returncode != 0
        assert "should not reach here" not in result.stdout


class TestPythonResolution:
    """Tests for Python interpreter resolution from launch command."""

    def test_resolves_python_from_launch_command(self, tmp_path):
        """The installer should use the same Python as the launch command."""
        venv_python = create_test_venv(tmp_path / "venv")

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "requirements.txt").write_text("six==1.17.0\n")

        script = textwrap.dedent(
            """\
            import sys
            print(f"python: {sys.executable}", flush=True)
            import six
            print(f"six: {six.__version__}", flush=True)
        """
        )
        script_file = tmp_path / "verify.py"
        script_file.write_text(script)

        result = run_supervisor(
            venv_python,
            [venv_python, str(script_file)],
            env_overrides={"SAGEMAKER_MODEL_PATH": str(model_dir)},
        )

        assert (
            result.returncode == 0
        ), f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        assert "six: 1.17.0" in result.stdout
