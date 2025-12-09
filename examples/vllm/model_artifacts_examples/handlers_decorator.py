"""
Custom handlers using decorator method.

This example demonstrates Method 2: Decorator Registration
- Use @custom_ping_handler and @custom_invocation_handler decorators
- System automatically discovers and registers decorated functions
- No environment variables needed
- Clean and explicit handler registration

Usage:
------
1. Upload this file to S3 with your model artifacts
2. Set environment variable: CUSTOM_SCRIPT_FILENAME=handlers_decorator.py
3. Deploy to SageMaker - handlers will be automatically registered
"""

import json
import logging

from fastapi import Request, Response

import model_hosting_container_standards.sagemaker as sagemaker_standards

logger = logging.getLogger(__name__)


@sagemaker_standards.custom_ping_handler
async def custom_health_check(request: Request) -> Response:
    """
    Custom health check handler.

    This handler is automatically registered via the @custom_ping_handler decorator.

    Returns:
        Response with JSON health status and appropriate HTTP status code
    """
    logger.info("Custom health check called via decorator")

    health_status = {
        "status": "healthy",
        "method": "decorator",
        "handler": "custom_health_check",
    }

    # Check vLLM engine
    try:
        await request.app.state.engine_client.check_health()
        health_status["vllm_engine"] = "healthy"
    except Exception as e:
        logger.error(f"vLLM engine check failed: {e}")
        health_status["vllm_engine"] = "unhealthy"
        health_status["status"] = "unhealthy"

    status_code = 200 if health_status["status"] == "healthy" else 503
    return Response(
        content=json.dumps(health_status),
        media_type="application/json",
        status_code=status_code,
    )


@sagemaker_standards.custom_invocation_handler
async def custom_inference(request: Request) -> Response:
    """
    Custom inference handler.

    This handler is automatically registered via the @custom_invocation_handler decorator.

    Request format:
        {
            "prompt": "Your question here",
            "max_tokens": 100,
            "temperature": 0.7
        }

    Returns:
        Response with JSON containing predictions and usage stats
    """
    try:
        body = await request.json()

        # Validate required fields
        if "prompt" not in body:
            return Response(
                content=json.dumps({"error": "Missing required field: prompt"}),
                media_type="application/json",
                status_code=400,
            )

        prompt = body["prompt"]
        max_tokens = body.get("max_tokens", 100)
        temperature = body.get("temperature", 0.7)

        logger.info(f"Inference via decorator handler - prompt length: {len(prompt)}")

        # Call vLLM engine
        import uuid

        from vllm import SamplingParams

        engine = request.app.state.engine_client

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )

        request_id = str(uuid.uuid4())
        results_generator = engine.generate(prompt, sampling_params, request_id)

        # Collect final output
        final_output = None
        async for request_output in results_generator:
            final_output = request_output

        # Extract generated text
        if final_output and final_output.outputs:
            generated_text = final_output.outputs[0].text
            prompt_tokens = (
                len(final_output.prompt_token_ids)
                if hasattr(final_output, "prompt_token_ids")
                else 0
            )
            completion_tokens = len(final_output.outputs[0].token_ids)
        else:
            generated_text = ""
            prompt_tokens = 0
            completion_tokens = 0

        response_data = {
            "predictions": [generated_text],
            "model": body.get("model", "vllm"),
            "method": "decorator",
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

        return Response(
            content=json.dumps(response_data),
            media_type="application/json",
            status_code=200,
            headers={"X-Request-Id": request_id},
        )

    except json.JSONDecodeError:
        return Response(
            content=json.dumps({"error": "Invalid JSON format"}),
            media_type="application/json",
            status_code=400,
        )
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return Response(
            content=json.dumps({"error": "Internal server error"}),
            media_type="application/json",
            status_code=500,
        )
