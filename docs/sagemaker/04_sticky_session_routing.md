# Sticky Session Routing

Sticky session routing enables all requests from the same session to be routed to the same model instance, allowing your application to reuse previously processed information to reduce latency and improve user experience.

> 📓 **Working Example**: See the [Sticky Session Notebook](../../examples/vllm/notebooks/sticky_session.ipynb) for a complete working example.

## Why Use Sticky Sessions?

Sticky sessions are particularly valuable for:

- **Multimodal applications** - Avoid resending large files (images, video, audio) with every request. Upload once, then ask multiple questions about the content.
- **Conversational AI** - Maintain context across multi-turn conversations without passing full history in each request.
- **Stateful inference** - Cache processed data (e.g., image tensors in GPU memory) to reduce latency on subsequent requests.

For example, in a chatbot scenario where users upload an image and ask follow-up questions, sending a 500 MB file with every request could add 3-5 seconds of latency. With sticky sessions, the file is processed once and cached for the duration of the session.

## How to Enable

Set the environment variable when deploying your model:

```python
sagemaker_client.create_model(
    ModelName='my-vllm-model',
    ExecutionRoleArn='arn:aws:iam::123456789012:role/SageMakerExecutionRole',
    PrimaryContainer={
        'Image': f'{account_id}.dkr.ecr.{region}.amazonaws.com/vllm:latest',
        'Environment': {
            'SM_VLLM_MODEL': 'meta-llama/Meta-Llama-3-8B-Instruct',
            'HUGGING_FACE_HUB_TOKEN': 'hf_your_token_here',
            # Enable sticky session routing
            'SAGEMAKER_ENABLE_STATEFUL_SESSIONS': 'true',
        }
    }
)
```

## How It Works

Once enabled, SageMaker handles session routing automatically. Clients interact with sessions using the `X-Amzn-SageMaker-Session-Id` header:

### 1. Create a Session

```python
import boto3
import json

runtime = boto3.client('sagemaker-runtime')

# Create a new session using SessionId parameter
response = runtime.invoke_endpoint(
    EndpointName='my-endpoint',
    ContentType='application/json',
    SessionId="NEW_SESSION",
    Body=json.dumps({"requestType": "NEW_SESSION"})
)

# Get session ID from response header
# Header format: "<uuid>; Expires=<timestamp>"
header_value = response['ResponseMetadata']['HTTPHeaders']['x-amzn-sagemaker-new-session-id']
session_id = header_value.split(";")[0].strip()
print(f"Session ID: {session_id}")
```

### 2. Use the Session

```python
# Send requests with the session ID
response = runtime.invoke_endpoint(
    EndpointName='my-endpoint',
    ContentType='application/json',
    SessionId=session_id,
    Body=json.dumps({"prompt": "What is in this image?"})
)

result = json.loads(response['Body'].read())
```

### 3. Close the Session

```python
# Close the session when done to free resources
response = runtime.invoke_endpoint(
    EndpointName='my-endpoint',
    ContentType='application/json',
    SessionId=session_id,
    Body=json.dumps({"requestType": "CLOSE"})
)
```

Sessions automatically expire after the configured TTL (default: 20 minutes).

## Configuration Reference

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `SAGEMAKER_ENABLE_STATEFUL_SESSIONS` | `false` | Enable sticky session routing |
| `SAGEMAKER_SESSIONS_EXPIRATION` | `1200` | Session TTL in seconds (20 minutes) |
| `SAGEMAKER_SESSIONS_PATH` | `/dev/shm` | Custom storage path for session data |

## Additional Resources

- **[Sticky Session Notebook](../../examples/vllm/notebooks/sticky_session.ipynb)** - Complete working example
- **[Quick Start Guide](01_quickstart.md)** - Basic deployment
- **[Customize Handlers](02_customize_handlers.md)** - Handler customization
