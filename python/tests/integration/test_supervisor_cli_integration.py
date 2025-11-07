"""
Integration tests for standard-supervisor CLI functionality.

Tests verify:
1. Configuration file generation and validation
2. Process supervision and restart behavior
3. Startup retry limits
4. Signal handling and graceful shutdown
"""

import configparser
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest


def get_python_cwd():
    """Get the correct working directory for python module execution."""
    current_dir = Path(__file__).parent.parent.parent.absolute()
    return str(current_dir)


def parse_supervisor_config(config_path):
    """Parse supervisor configuration file and return configparser object."""
    config = configparser.ConfigParser()
    config.read(config_path)
    return config


class TestSupervisorCLIIntegration:
    """Integration tests for the standard-supervisor CLI."""

    @pytest.fixture
    def clean_env(self):
        """Provide clean environment for testing."""
        original_env = dict(os.environ)

        # Clear supervisor-related variables
        for key in list(os.environ.keys()):
            if key.startswith("SUPERVISOR_") or key in [
                "PROCESS_AUTO_RECOVERY",
                "PROCESS_MAX_START_RETRIES",
                "LOG_LEVEL",
            ]:
                del os.environ[key]

        yield

        # Restore original environment
        os.environ.clear()
        os.environ.update(original_env)

    def test_basic_cli_execution_and_config_generation(self, clean_env):
        """Test basic CLI execution with configuration generation and validation."""
        env = {
            "PROCESS_MAX_START_RETRIES": "2",
            "SUPERVISOR_PROGRAM__APP_STARTSECS": "2",
            "SUPERVISOR_PROGRAM__APP_STOPWAITSECS": "5",
            "SUPERVISOR_PROGRAM__APP_AUTORESTART": "true",
            "LOG_LEVEL": "info",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "supervisord.conf")
            env["SUPERVISOR_CONFIG_PATH"] = config_path

            # Start supervisor with a long-running server
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "model_hosting_container_standards.supervisor.scripts.standard_supervisor",
                    sys.executable,
                    "-c",
                    "import time; print('Server started', flush=True); time.sleep(30)",
                ],
                env={**os.environ, **env},
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=get_python_cwd(),
            )

            try:
                # Give it time to start and generate config
                time.sleep(3)

                # Verify config file was generated
                assert os.path.exists(
                    config_path
                ), f"Config file not found at {config_path}"

                config = parse_supervisor_config(config_path)

                # Check main sections exist
                assert "supervisord" in config.sections()
                assert "program:app" in config.sections()

                # Verify program configuration
                program_section = config["program:app"]
                assert "python" in program_section["command"]
                assert program_section["startsecs"] == "2"
                assert program_section["stopwaitsecs"] == "5"
                assert program_section["autostart"] == "true"
                assert program_section["autorestart"] == "true"
                assert program_section["stdout_logfile"] == "/dev/stdout"
                assert program_section["stderr_logfile"] == "/dev/stderr"

            finally:
                # Clean up
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.communicate(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.communicate()

    def test_ml_framework_configuration(self, clean_env):
        """Test supervisor configuration for ML framework scenarios."""
        env = {
            "PROCESS_MAX_START_RETRIES": "3",
            "SUPERVISOR_PROGRAM__APP_STARTSECS": "30",  # ML models need longer startup
            "SUPERVISOR_PROGRAM__APP_STOPWAITSECS": "60",  # Graceful shutdown time
            "SUPERVISOR_PROGRAM__APP_STARTRETRIES": "3",
            "SUPERVISOR_PROGRAM__APP_AUTORESTART": "true",
            "LOG_LEVEL": "info",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "supervisord.conf")
            env["SUPERVISOR_CONFIG_PATH"] = config_path

            # Simulate ML framework server
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "model_hosting_container_standards.supervisor.scripts.standard_supervisor",
                    sys.executable,
                    "-c",
                    "print('ML model server starting...', flush=True); import time; time.sleep(30); print('Ready')",
                ],
                env={**os.environ, **env},
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=get_python_cwd(),
            )

            try:
                # Give it time to start and generate config
                time.sleep(3)

                # Verify ML-specific configuration
                assert os.path.exists(
                    config_path
                ), f"Config file not found at {config_path}"

                config = parse_supervisor_config(config_path)
                program_section = config["program:app"]

                # ML frameworks need longer startup and shutdown times
                assert program_section["startsecs"] == "30"
                assert program_section["stopwaitsecs"] == "60"
                assert program_section["startretries"] == "3"
                assert program_section["autorestart"] == "true"

                # Verify process management settings for ML workloads
                assert program_section["stopasgroup"] == "true"
                assert program_section["killasgroup"] == "true"
                assert program_section["stopsignal"] == "TERM"

            finally:
                # Clean up
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.communicate(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.communicate()

    def test_signal_handling(self, clean_env):
        """Test that supervisor handles signals correctly."""
        env = {
            "PROCESS_MAX_START_RETRIES": "1",
            "SUPERVISOR_PROGRAM__APP_STARTSECS": "1",
            "SUPERVISOR_PROGRAM__APP_STOPWAITSECS": "5",
            "LOG_LEVEL": "info",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "supervisord.conf")
            env["SUPERVISOR_CONFIG_PATH"] = config_path

            # Start a long-running process
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "model_hosting_container_standards.supervisor.scripts.standard_supervisor",
                    sys.executable,
                    "-c",
                    "import time; print('Long running process started', flush=True); time.sleep(30)",
                ],
                env={**os.environ, **env},
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=get_python_cwd(),
            )

            try:
                # Give it time to start
                time.sleep(3)
                assert os.path.exists(config_path)

                # Send SIGTERM to test graceful shutdown
                process.send_signal(signal.SIGTERM)

                # Wait for termination with longer timeout
                # supervisord needs time to stop child processes
                stdout, stderr = process.communicate(timeout=10)

                # Should have terminated (any exit code is fine, we just want it to stop)
                assert process.returncode is not None

            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                pytest.fail("Process did not terminate gracefully within timeout")

    def test_continuous_restart_behavior(self, clean_env):
        """Test that supervisor continuously restarts processes when autorestart=true."""
        env = {
            "SUPERVISOR_PROGRAM__APP_STARTSECS": "2",
            "SUPERVISOR_PROGRAM__APP_AUTORESTART": "true",
            "SUPERVISOR_PROGRAM__APP_STARTRETRIES": "10",
            "LOG_LEVEL": "info",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "supervisord.conf")
            restart_log = os.path.join(temp_dir, "restart_log.txt")
            env["SUPERVISOR_CONFIG_PATH"] = config_path

            # Create a server that runs briefly then exits (to test restart)
            server_script_file = os.path.join(temp_dir, "test_server.py")
            with open(server_script_file, "w") as f:
                f.write(
                    f"""import time
import sys
import os

# Log each startup
with open('{restart_log}', 'a') as f:
    f.write(f'Server started at {{time.time()}}\\n')
    f.flush()

print('Server started, PID:', os.getpid(), flush=True)

# Run for 3 seconds then exit (supervisor will restart due to autorestart=true)
for i in range(3):
    time.sleep(1)
    print(f'Server running {{i+1}}/3', flush=True)

print('Server exiting (will be restarted by supervisor)', flush=True)
sys.exit(0)
"""
                )

            # Start supervisor with the server
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "model_hosting_container_standards.supervisor.scripts.standard_supervisor",
                    sys.executable,
                    server_script_file,
                ],
                env={**os.environ, **env},
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=get_python_cwd(),
            )

            try:
                # Wait for multiple restart cycles
                time.sleep(10)

                # Check restart log
                assert os.path.exists(
                    restart_log
                ), "Server should have created restart log"
                with open(restart_log, "r") as f:
                    restart_entries = f.read().strip().split("\n")
                    restart_count = len([line for line in restart_entries if line])

                print(f"Server restart count: {restart_count}")

                # Should have multiple restarts
                assert (
                    restart_count >= 2
                ), f"Server should have been restarted multiple times, got {restart_count}"

                # Verify config
                config = parse_supervisor_config(config_path)
                program_section = config["program:app"]
                assert program_section["autorestart"] == "true"

                print(
                    f"✅ Server was restarted {restart_count} times, proving continuous restart behavior"
                )

            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.communicate(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.communicate()

    def test_startup_retry_limit(self, clean_env):
        """Test that supervisor respects startretries limit."""
        env = {
            "SUPERVISOR_PROGRAM__APP_STARTSECS": "5",  # Process must run 5 seconds to be "started"
            "SUPERVISOR_PROGRAM__APP_STARTRETRIES": "3",  # Only 3 startup attempts
            "SUPERVISOR_PROGRAM__APP_AUTORESTART": "true",
            "LOG_LEVEL": "info",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "supervisord.conf")
            startup_log = os.path.join(temp_dir, "startup_attempts.txt")
            env["SUPERVISOR_CONFIG_PATH"] = config_path

            # Create script that logs startup attempts then fails before startsecs
            script_file = os.path.join(temp_dir, "failing_script.py")
            with open(script_file, "w") as f:
                f.write(
                    f"""import time
import os

# Log this startup attempt
with open('{startup_log}', 'a') as f:
    f.write(f'Startup attempt at {{time.time()}}\\n')
    f.flush()

print('Process starting up...', flush=True)
time.sleep(2)  # Run for 2 seconds (less than startsecs=5, so it's a startup failure)
print('Process failing before startsecs...', flush=True)
exit(1)
"""
                )

            # Run supervisor with the failing script
            # Use Popen since supervisord won't exit after FATAL
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "model_hosting_container_standards.supervisor.scripts.standard_supervisor",
                    sys.executable,
                    script_file,
                ],
                env={**os.environ, **env},
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=get_python_cwd(),
            )

            try:
                # Wait for retries to complete (should take ~10 seconds)
                time.sleep(15)

                # Verify config was generated
                assert os.path.exists(config_path), "Config file should exist"
                config = parse_supervisor_config(config_path)
                program_section = config["program:app"]
                assert program_section["startretries"] == "3"
                assert program_section["startsecs"] == "5"

                # Check startup attempts
                assert os.path.exists(
                    startup_log
                ), "Startup log should have been created"

                with open(startup_log, "r") as f:
                    startup_attempts = f.read().strip().split("\n")
                    attempt_count = len([line for line in startup_attempts if line])

                # Should have made exactly startretries + 1 attempts (initial + retries)
                expected_attempts = 4  # 1 initial + 3 retries
                assert (
                    attempt_count == expected_attempts
                ), f"Expected {expected_attempts} startup attempts, got {attempt_count}"

                # Check supervisord log for FATAL state
                log_path = "/tmp/supervisord-app.log"
                if os.path.exists(log_path):
                    with open(log_path, "r") as f:
                        log_content = f.read()
                        assert (
                            "gave up:" in log_content
                            and "entered FATAL state" in log_content
                        ), "Supervisor should have entered FATAL state"

                print(
                    f"✅ Supervisor made exactly {attempt_count} startup attempts before giving up"
                )

            finally:
                # Clean up
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.communicate(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.communicate()

    def test_configuration_validation_error(self, clean_env):
        """Test CLI with invalid configuration."""
        env = {
            "PROCESS_MAX_START_RETRIES": "invalid_number",  # Invalid value
        }

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "model_hosting_container_standards.supervisor.scripts.standard_supervisor",
                "echo",
                "test",
            ],
            env={**os.environ, **env},
            capture_output=True,
            text=True,
            timeout=10,
            cwd=get_python_cwd(),
        )

        # Should fail due to configuration error
        assert result.returncode == 1
        output = result.stdout + result.stderr
        assert (
            "Configuration error" in output
            or "must be an integer" in output
            or "Configuration validation failed" in output
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
