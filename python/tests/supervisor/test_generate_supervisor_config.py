"""
Unit tests for the generate-supervisor-config CLI.

These tests exercise the script end to end: they invoke ``main()``, read the
emitted supervisord config file, and assert that the ``command=`` value
round-trips back to the original argv when parsed with ``shlex.split`` (the
same parsing supervisord performs). This guards the quote-preservation fix for
ticket P446501135, where a plain ``" ".join`` corrupted arguments containing
spaces or quotes (e.g. the JSON passed to vLLM's ``--speculative-config``).
"""

import configparser
import os
import shlex  # inverse of shlex.join; supervisord parses command= the same way
import sys
import tempfile
from unittest.mock import patch

import pytest

from model_hosting_container_standards.supervisor.scripts.generate_supervisor_config import (  # noqa: E501
    main,
)

# Each case is a launch argv that must survive the round-trip intact. They cover
# JSON values, embedded spaces, single quotes, double quotes, and empty strings.
ROUND_TRIP_CASES = {
    "json_speculative_config": [
        "python3",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--speculative-config",
        (
            '{"model": "/opt/ml/additional-model-data-sources/eagle", '
            '"method": "eagle3", "num_speculative_tokens": 3, '
            '"parallel_drafting": true}'
        ),
    ],
    "args_with_spaces": [
        "vllm",
        "serve",
        "my model",
        "--served-model-name",
        "my model name",
    ],
    "single_quotes": ["echo", "it's a 'quoted' value"],
    "empty_string_arg": ["python3", "-c", "print('hi')", "--flag", ""],
    "json_and_plain_mix": [
        "vllm",
        "serve",
        "model",
        "--chat-template",
        '{"x": "a b c"}',
        "--arg",
        "plain",
    ],
}


@pytest.fixture
def clean_env():
    """Provide a clean supervisor-related environment for each test."""
    original_env = dict(os.environ)
    for key in list(os.environ.keys()):
        if key.startswith("SUPERVISOR_") or key in (
            "PROCESS_AUTO_RECOVERY",
            "PROCESS_MAX_START_RETRIES",
            "LOG_LEVEL",
        ):
            del os.environ[key]
    yield
    os.environ.clear()
    os.environ.update(original_env)


def _read_command(config_path: str) -> str:
    """Read the program command from an emitted supervisord config file."""
    # interpolation=None so '%' inside argument values is not treated as a token.
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(config_path)
    return parser["program:app"]["command"]


class TestGenerateSupervisorConfigQuoting:
    """End-to-end quote-preservation tests for the CLI."""

    @pytest.mark.parametrize(
        "argv", ROUND_TRIP_CASES.values(), ids=ROUND_TRIP_CASES.keys()
    )
    def test_command_round_trips_through_shlex(self, clean_env, argv):
        """The emitted command= must shlex.split back to the original argv."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "supervisord.conf")

            # "--" stops argparse option parsing so command args may begin with "-".
            with patch.object(
                sys,
                "argv",
                ["generate-supervisor-config", "-o", config_path, "--", *argv],
            ):
                result = main()

            assert result == 0
            command = _read_command(config_path)
            assert shlex.split(command) == argv

    def test_speculative_config_json_stays_single_token(self, clean_env):
        """Regression for P446501135: the JSON value stays one intact token."""
        speculative_config = (
            '{"model": "/opt/ml/additional-model-data-sources/eagle", '
            '"method": "eagle3", "num_speculative_tokens": 3, '
            '"parallel_drafting": true}'
        )
        argv = [
            "python3",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--speculative-config",
            speculative_config,
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "supervisord.conf")
            with patch.object(
                sys,
                "argv",
                ["generate-supervisor-config", "-o", config_path, "--", *argv],
            ):
                assert main() == 0

            command = _read_command(config_path)
            tokens = shlex.split(command)

        assert speculative_config in tokens
        # The value following --speculative-config must be the full JSON, unsplit.
        assert tokens[tokens.index("--speculative-config") + 1] == speculative_config


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
