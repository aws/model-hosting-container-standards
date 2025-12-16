# LoRA Adapter Management

LoRA (Low-Rank Adaptation) adapter management enables dynamic loading and unloading of fine-tuned adapters at runtime, allowing a single base model to serve multiple specialized use cases without redeployment.

> ⚠️ **Required**: Set `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` to enable dynamic adapter loading with inference components. See [vLLM LoRA documentation](https://docs.vllm.ai/en/latest/features/lora.html) for details.

> 📓 **Working Example**: See the [LoRA Adapters Notebook](../../examples/vllm/notebooks/lora_adapters.ipynb) for a complete working example.

## Why Use LoRA Adapters?

LoRA adapters are valuable for:

- **Multi-tenant deployments** - Serve different customers with specialized fine-tuned models from a single endpoint.
- **A/B testing** - Compare different fine-tuned versions without deploying separate endpoints.
- **Cost optimization** - Share GPU resources across multiple specialized models instead of deploying separate instances.
- **Rapid iteration** - Deploy new fine-tuned adapters without restarting the inference server.

## Architecture Overview

With SageMaker inference components, LoRA adapters are deployed as a hierarchy:

1. **Base Inference Component** - The base model with LoRA support enabled
2. **Adapter Inference Components** - Child components that reference the base and load specific adapters

```
Endpoint
└── Base Inference Component (base model + LoRA enabled)
    ├── Adapter IC: adapter-1
    ├── Adapter IC: adapter-2
    └── ...
```

## How to Enable

### Step 1: Create Endpoint

Create an endpoint configuration and endpoint to host inference components:

```python
sm_client.create_endpoint_config(
    EndpointConfigName=endpoint_config_name,
    ExecutionRoleArn=role,
    ProductionVariants=[{
        "VariantName": "main",
        "InstanceType": "ml.g6e.12xlarge",
        "InitialInstanceCount": 1,
        "ContainerStartupHealthCheckTimeoutInSeconds": 600,
        "RoutingConfig": {"RoutingStrategy": "LEAST_OUTSTANDING_REQUESTS"},
    }],
)

sm_client.create_endpoint(
    EndpointName=endpoint_name,
    EndpointConfigName=endpoint_config_name
)
```

### Step 2: Create Base Model with LoRA Enabled

```python
env = {
    "SM_VLLM_MODEL": "/opt/ml/model",
    "SM_VLLM_TENSOR_PARALLEL_SIZE": "2",
    "SM_VLLM_MAX_MODEL_LEN": "4096",
    # LoRA configuration
    "SM_VLLM_ENABLE_LORA": "true",
    "SM_VLLM_MAX_LORA_RANK": "64",
    "SM_VLLM_MAX_CPU_LORAS": "4",
    "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true"
}

sm_client.create_model(
    ModelName=base_model_name,
    ExecutionRoleArn=role,
    PrimaryContainer={
        "Image": f"{account_id}.dkr.ecr.{region}.amazonaws.com/vllm:0.11.2-sagemaker-v1.2",
        "Environment": env,
        "ModelDataSource": {
            "S3DataSource": {
                "S3Uri": "s3://bucket/models/llama-3.1-8b/",
                "S3DataType": "S3Prefix",
                "CompressionType": "None"
            }
        }
    },
)
```

### Step 3: Create Base Inference Component

```python
sm_client.create_inference_component(
    InferenceComponentName=base_ic_name,
    EndpointName=endpoint_name,
    VariantName="main",
    Specification={
        "ModelName": base_model_name,
        "StartupParameters": {
            "ModelDataDownloadTimeoutInSeconds": 600,
            "ContainerStartupHealthCheckTimeoutInSeconds": 600,
        },
        "ComputeResourceRequirements": {
            "MinMemoryRequiredInMb": 4096,
            "NumberOfAcceleratorDevicesRequired": 2,
        },
    },
    RuntimeConfig={"CopyCount": 1},
)
```

### Step 4: Create Adapter Inference Component

Package the adapter as `tar.gz` and upload to S3, then create a child inference component:

```python
# Package adapter
with tarfile.open("adapter.tar.gz", "w:gz") as tar:
    for name in os.listdir(adapter_local_path):
        tar.add(os.path.join(adapter_local_path, name), arcname=name)

# Upload to S3
s3_client.upload_file("adapter.tar.gz", bucket, "lora/chinese/adapter.tar.gz")

# Create adapter inference component
sm_client.create_inference_component(
    InferenceComponentName="adapter-chinese",
    EndpointName=endpoint_name,
    Specification={
        # Reference the base inference component
        "BaseInferenceComponentName": base_ic_name,
        "Container": {
            # S3 path to the adapter tar.gz
            "ArtifactUrl": "s3://bucket/lora/chinese/adapter.tar.gz"
        },
    },
)
```

## Making Inference Requests

### Using an Adapter

Specify the adapter's inference component name in the request:

```python
response = smr_client.invoke_endpoint(
    EndpointName=endpoint_name,
    InferenceComponentName="adapter-chinese",  # Adapter IC name
    ContentType="application/json",
    Body=json.dumps({
        "prompt": ["你好，今天天气怎么样？"],
        "max_tokens": 100,
        "temperature": 0.0,
    })
)

result = json.loads(response["Body"].read())
print(result["choices"][0]["text"])
```

### Using Base Model (No Adapter)

To use the base model without any adapter, specify the base inference component:

```python
response = smr_client.invoke_endpoint(
    EndpointName=endpoint_name,
    InferenceComponentName=base_ic_name,  # Base IC name
    ContentType="application/json",
    Body=json.dumps({
        "prompt": ["What is the capital of France?"],
        "max_tokens": 100,
    })
)
```

## Configuration Reference

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `SM_VLLM_ENABLE_LORA` | `false` | Enable LoRA adapter support |
| `SM_VLLM_MAX_LORA_RANK` | `16` | Maximum LoRA rank supported. Set to match your adapters' max rank |
| `SM_VLLM_MAX_LORAS` | `1` | Maximum number of LoRA adapters loaded simultaneously |
| `SM_VLLM_MAX_CPU_LORAS` | `None` | Maximum adapters cached in CPU memory |
| `VLLM_ALLOW_RUNTIME_LORA_UPDATING` | `false` | Allow dynamic loading/unloading of adapters |

## Additional Resources

- **[LoRA Adapters Notebook](../../examples/vllm/notebooks/lora_adapters.ipynb)** - Complete working example
- **[Multi-Adapter Inference Blog](https://aws.amazon.com/blogs/machine-learning/easily-deploy-and-manage-hundreds-of-lora-adapters-with-sagemaker-efficient-multi-adapter-inference/)** - Detailed walkthrough of SageMaker multi-adapter deployment
- **[vLLM LoRA Documentation](https://docs.vllm.ai/en/latest/features/lora.html)** - vLLM LoRA details
- **[Quick Start Guide](01_quickstart.md)** - Basic deployment
- **[Sticky Session Routing](04_sticky_session_routing.md)** - Stateful sessions
