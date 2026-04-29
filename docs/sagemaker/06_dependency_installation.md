# Automatic Dependency Installation

Install custom Python dependencies from your model artifacts before the inference server starts.

## Overview

When deploying models on SageMaker, you may need additional Python packages that aren't included in the container image. The `standard-supervisor` CLI automatically installs dependencies from a `requirements.txt` file found in your model artifacts, using `uv` for fast resolution (with `pip` as a fallback).

This runs once at container startup, before the inference server process begins — so all packages are available when your model loads.

## How It Works

```
Container starts
  → standard-supervisor parses launch command
  → Installs dependencies (if requirements.txt found)
  → Starts the inference server under supervision
```

The installer resolves the Python interpreter from the launch command (e.g., `python3` in `standard-supervisor python3 -m vllm ...`) to ensure packages are installed into the correct site-packages.

## Quick Start

Place a `requirements.txt` in your model artifact directory:

```
/opt/ml/model/
├── requirements.txt      # Your Python dependencies
├── config.json           # Model config
├── model.safetensors     # Model weights
└── tokenizer.json        # Tokenizer
```

That's it. The `standard-supervisor` discovers and installs it automatically.

### Example requirements.txt

```txt
transformers>=4.40.0
scipy==1.13.0
custom-tokenizer==1.0.0
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `STANDARD_AUTO_INSTALL_REQ` | `true` | Set to `false` to disable automatic installation entirely. |
| `STANDARD_PIP_ARGS` | *(unset)* | When set, replaces auto-discovery. The value is passed directly as arguments to `uv pip install` (or `pip install`). |

### STANDARD_AUTO_INSTALL_REQ

Controls whether the dependency installation step runs at all.

```bash
# Disable automatic installation
docker run -e STANDARD_AUTO_INSTALL_REQ=false my-image
```

### STANDARD_PIP_ARGS

When set, auto-discovery of `/opt/ml/model/requirements.txt` is skipped entirely. The value is used as the full set of arguments to the install command.

```bash
# Install from a custom file location
docker run -e STANDARD_PIP_ARGS="-r /custom/path/requirements.txt" my-image

# Install from a custom index
docker run -e STANDARD_PIP_ARGS="-r /opt/ml/model/requirements.txt --index-url https://my-index/simple" my-image

# Install specific packages directly
docker run -e STANDARD_PIP_ARGS="scipy==1.13.0 pandas" my-image
```

## Auto-Discovery Behavior

When `STANDARD_PIP_ARGS` is **not set**, the installer runs in auto-discovery mode:

1. Looks for `requirements.txt` in the model directory (`SAGEMAKER_MODEL_PATH` or `/opt/ml/model`).
2. If found, runs `uv pip install -r /opt/ml/model/requirements.txt`.
3. If a `requirements/` subdirectory exists alongside the file, adds `--find-links` for offline installs.
4. If no `requirements.txt` is found, the step is a silent no-op.

When `STANDARD_PIP_ARGS` **is set**, auto-discovery is skipped entirely to avoid duplicate `-r` flags. The customer's value is the complete set of install arguments.

## Offline / Air-Gapped Installs

For environments without internet access, bundle wheel files in a `requirements/` subdirectory:

```
/opt/ml/model/
├── requirements.txt
└── requirements/
    ├── scipy-1.13.0-cp312-cp312-linux_x86_64.whl
    └── custom_tokenizer-1.0.0-py3-none-any.whl
```

The installer automatically detects this directory and passes `--find-links` to pip, so packages resolve locally without network access.

## Installer Resolution

The installer prefers `uv` for speed, falling back to `pip` if `uv` is not available:

| Priority | Tool | Command |
|----------|------|---------|
| 1 | `uv` (on PATH) | `uv pip install --python <python> -r requirements.txt` |
| 2 | `pip` (in target Python) | `<python> -m pip install -r requirements.txt` |

If `uv` is not on PATH, the installer falls back to `pip`. If `pip` is also missing, the install command fails and the error output from pip is logged.

## Python Interpreter Resolution

The installer determines which Python to target by inspecting the launch command passed to `standard-supervisor`:

| Launch command | Resolved Python |
|----------------|-----------------|
| `standard-supervisor python3 -m vllm ...` | `python3` (resolved via PATH) |
| `standard-supervisor /opt/venv/bin/python3.12 -m vllm ...` | `/opt/venv/bin/python3.12` |
| `standard-supervisor vllm serve ...` | `sys.executable` (the Python running standard-supervisor) |

This ensures packages are installed into the same site-packages the inference server will use.

## Troubleshooting

### Dependencies not being installed

Check that:
- `requirements.txt` exists at `/opt/ml/model/requirements.txt`
- `STANDARD_AUTO_INSTALL_REQ` is not set to `false`
- The entrypoint uses `standard-supervisor` (e.g., `exec standard-supervisor python3 -m vllm ...`)

Enable debug logging to see the install command:
```bash
export LOG_LEVEL=debug
```

### Installation fails

Check the error output in the container logs. Common causes:
- The package doesn't exist on PyPI or the configured index
- Network connectivity issues (consider offline installs)
- Neither `uv` nor `pip` is available in the container

### Wrong Python — packages not importable at runtime

This happens when `standard-supervisor` installs into a different Python than the inference server uses. Ensure the launch command starts with an explicit Python path:

```bash
# Good — installer can detect python3
standard-supervisor python3 -m vllm.entrypoints.openai.api_server ...

# Risky — installer falls back to sys.executable
standard-supervisor vllm serve ...
```
