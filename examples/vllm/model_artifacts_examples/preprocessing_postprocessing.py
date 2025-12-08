"""
Pre/Post Processing Customization Examples.

This file demonstrates two methods for customizing request preprocessing
and response postprocessing in vLLM on SageMaker.

⚠️ IMPORTANT: Pre/post processors run on ALL endpoints (/ping, /invocations, etc.)
Always check request.url.path to filter which endpoints to process!

💡 VERIFICATION: This example modifies the prompt to include "nya nya nya" instruction.
Check the response text - if it starts with "nya nya nya", pre-processing worked!

Method 1: Decorator-based (Recommended)
---------------------------------------
Use @input_formatter and @output_formatter decorators for clean separation.

Usage:
    1. Upload this file to S3 with your model artifacts
    2. Set environment variable: CUSTOM_SCRIPT_FILENAME=preprocessing_postprocessing.py
    3. Deploy to SageMaker - formatters will be automatically registered

Method 2: Environment Variables
--------------------------------
Point to specific functions via CUSTOM_PRE_PROCESS and CUSTOM_POST_PROCESS.

Usage:
    1. Upload this file to S3 with your model artifacts
    2. Set environment variables:
       - CUSTOM_PRE_PROCESS=preprocessing_postprocessing.py:custom_pre_process
       - CUSTOM_POST_PROCESS=preprocessing_postprocessing.py:custom_post_process
    3. Deploy to SageMaker

What Works:
-----------
✅ Request preprocessing (add defaults, validate, transform, modify prompts)
✅ Adding custom response headers
✅ Verifying pre-processing via modified model output
"""

import json
import logging

from fastapi import Request, Response
from model_hosting_container_standards.common.fastapi.middleware import (
    input_formatter, output_formatter)

logger = logging.getLogger(__name__)


# ============================================================
# Method 1: Decorator-based Pre/Post Processing (Recommended)
# ============================================================


@input_formatter
async def pre_process_request(request: Request) -> Request:
    """Pre-process incoming requests using decorator."""
    if request.url.path != "/invocations":
        return request

    logger.info("[DECORATOR] Pre-processing /invocations request")

    body = await request.json()

    if "max_tokens" not in body:
        body["max_tokens"] = 100
        logger.debug("Added default max_tokens=100")

    if "temperature" not in body:
        body["temperature"] = 0.7
        logger.debug("Added default temperature=0.7")

    original_prompt = body.get("prompt", "")
    body["prompt"] = "Say 'nya nya nya' first, then answer: " + original_prompt
    logger.info("[DECORATOR] Modified prompt to include 'nya nya nya' instruction")

    request._body = json.dumps(body).encode()
    logger.info(f"[DECORATOR] Request pre-processed: {len(body)} fields")

    return request


@output_formatter
async def post_process_response(response: Response) -> Response:
    """Post-process outgoing responses using decorator."""
    logger.info(
        f"[DECORATOR] Post-processing response: type={type(response).__name__}, "
        f"status={response.status_code}, has_body={hasattr(response, 'body')}"
    )

    if not hasattr(response, "body"):
        logger.info("[DECORATOR] Streaming response detected")
        return response

    try:
        body = json.loads(response.body)
        logger.info(f"[DECORATOR] Response body: {json.dumps(body, indent=2)}")
    except (json.JSONDecodeError, AttributeError) as e:
        logger.warning(f"[DECORATOR] Could not parse response body: {e}")

    return response


# ============================================================
# Method 2: Environment Variable-based Pre/Post Processing
# ============================================================


async def custom_pre_process(request: Request) -> Request:
    """Pre-process incoming requests via environment variable."""
    if request.url.path != "/invocations":
        return request

    logger.info("[ENV_VAR] Pre-processing /invocations request")

    body = await request.json()

    if "max_tokens" not in body:
        body["max_tokens"] = 150
        logger.debug("Added default max_tokens=150")

    if "temperature" not in body:
        body["temperature"] = 0.8
        logger.debug("Added default temperature=0.8")

    original_prompt = body.get("prompt", "")
    body["prompt"] = "Say 'nya nya nya' first, then answer: " + original_prompt
    logger.info("[ENV_VAR] Modified prompt to include 'nya nya nya' instruction")

    request._body = json.dumps(body).encode()
    logger.info(f"[ENV_VAR] Request pre-processed: {len(body)} fields")

    return request


async def custom_post_process(response: Response) -> Response:
    """Post-process outgoing responses via environment variable."""
    logger.info(
        f"[ENV_VAR] Post-processing response: status={response.status_code}, "
        f"has_body={hasattr(response, 'body')}"
    )

    if not hasattr(response, "body"):
        logger.info("[ENV_VAR] Streaming response detected")
        return response

    try:
        body = json.loads(response.body)
        logger.info(f"[ENV_VAR] Response body: {json.dumps(body, indent=2)}")
    except (json.JSONDecodeError, AttributeError) as e:
        logger.warning(f"[ENV_VAR] Could not parse response body: {e}")

    return response
