# Quick Start: Deploy vLLM on SageMaker

Deploy a vLLM-powered Large Language Model on Amazon SageMaker in minutes.

## Quick Start

**Fastest way to get started:**

👉 **[Basic Endpoint Notebook](../../examples/vllm/notebooks/basic_endpoint.ipynb)**

The notebook includes complete deployment workflow, inference examples (single, concurrent, streaming), and automatic cleanup.

## Container Images

AWS provides official vLLM container images in the [Amazon ECR Public Gallery](https://gallery.ecr.aws/deep-learning-containers/vllm).

**Example:**
```
public.ecr.aws/deep-learning-containers/vllm:0.11.2-gpu-py312-cu129-ubuntu22.04-sagemaker-v1.2
```

**Note:** Copy the public image to your private ECR repository for SageMaker deployment. See [copy_image.ipynb](../../examples/vllm/notebooks/copy_image.ipynb).

**Features:**
- vLLM inference engine
- SageMaker-compatible API
- Custom handler support
- Custom middleware support
- Custom pre/post-processing support
- Sticky routing (stateful sessions)
- Multi-LoRA adapter management

## Basic Deployment

### Required Configuration

```python
sagemaker_client.create_model(
    ModelName='my-vllm-model',
    ExecutionRoleArn='arn:aws:iam::123456789012:role/SageMakerExecutionRole',
    PrimaryContainer={
        'Image': f'{account_id}.dkr.ecr.{region}.amazonaws.com/vllm:0.11.2-sagemaker-v1.2',
        'Environment': {
            'SM_VLLM_MODEL': 'meta-llama/Meta-Llama-3-8B-Instruct',
            'HUGGING_FACE_HUB_TOKEN': 'hf_your_token_here',
        }
    }
)
```

### vLLM Engine Configuration

Configure vLLM using `SM_VLLM_*` environment variables (automatically converted to CLI arguments):

```python
'Environment': {
    'SM_VLLM_MODEL': 'meta-llama/Meta-Llama-3-8B-Instruct',
    'HUGGING_FACE_HUB_TOKEN': 'hf_your_token_here',
    'SM_VLLM_MAX_MODEL_LEN': '2048',
    'SM_VLLM_GPU_MEMORY_UTILIZATION': '0.9',
    'SM_VLLM_DTYPE': 'auto',
    'SM_VLLM_TENSOR_PARALLEL_SIZE': '1',
}
```

All vLLM CLI arguments are supported. See [vLLM CLI documentation](https://docs.vllm.ai/en/latest/cli/serve/#frontend) for available parameters.

### Model Path Options

`SM_VLLM_MODEL` accepts two types of values:

**1. Hugging Face Model ID** (downloads from HF Hub):
```python
'SM_VLLM_MODEL': 'meta-llama/Meta-Llama-3-8B-Instruct'
```

**2. Local Folder Path** (for S3 model artifacts):
```python
'SM_VLLM_MODEL': '/opt/ml/model'
```

When deploying with model artifacts from S3, SageMaker automatically downloads them to `/opt/ml/model`. Use this path to load your pre-downloaded models instead of fetching from Hugging Face.


## Making Inference Requests

```python
runtime_client = boto3.client('sagemaker-runtime')

response = runtime_client.invoke_endpoint(
    EndpointName='my-vllm-endpoint',
    ContentType='application/json',
    Body=json.dumps({
        "prompt": "What is the capital of France?",
        "max_tokens": 100,
        "temperature": 0.7
    })
)

result = json.loads(response['Body'].read())
print(result['choices'][0]['text'])
```

For complete deployment code including concurrent requests, streaming responses, and cleanup, see the [Basic Endpoint Notebook](../../examples/vllm/notebooks/basic_endpoint.ipynb).

## Next Steps

**Advanced Features:**
- [Customize Handlers](02_customize_handlers.md) - Custom ping and invocation handlers
- [Customize Pre/Post Processing](03_customize_pre_post_processing.md) - Custom middleware for request/response transformation
- [Sticky Session Routing](04_sticky_session_routing.md) - Stateful sessions for multi-turn conversations
- [LoRA Adapter Management](05_lora_adapter_management.md) - Dynamic loading/unloading of fine-tuned adapters

**Resources:**
- [Basic Endpoint Notebook](../../examples/vllm/notebooks/basic_endpoint.ipynb) - Complete deployment example
- [Handler Customization Notebook](../../examples/vllm/notebooks/handler_customization_methods.ipynb) - Handler override examples
- [Pre/Post Processing Notebook](../../examples/vllm/notebooks/preprocessing_postprocessing_methods.ipynb) - Middleware examples
- [Sticky Session Notebook](../../examples/vllm/notebooks/sticky_session.ipynb) - Stateful session examples
- [Python Package README](../../python/README.md) - Handler system documentation
- [vLLM Documentation](https://docs.vllm.ai/) - vLLM engine details
