"""AWS Bedrock client for the company-news gold judge.

Copied from self-serve-backend (qualify_icp_person.get_bedrock_client +
search/wiki/bedrock_utils.call_bedrock_json). Uses Bedrock converse only.
Does not use ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]

# Same us. inference-profile prefix as self-serve-backend's Sonnet id
# (us.anthropic.claude-sonnet-4-6). Opus 5 on Bedrock: us.anthropic.claude-opus-5
DEFAULT_OPUS_MODEL = "us.anthropic.claude-opus-5"

_BEDROCK_CONFIG = Config(
    max_pool_connections=50,
    retries={"max_attempts": 3, "mode": "adaptive"},
    read_timeout=180,
    connect_timeout=10,
)
_LOCK = threading.Lock()
_CLIENT = None

AWS_ENV_KEYS = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_REGION",
    "AWS_DEFAULT_REGION",
    "AWS_PROFILE",
    "AWS_BEARER_TOKEN_BEDROCK",
    "BEDROCK_JUDGE_MODEL",
)


def load_bedrock_environment() -> None:
    """Load AWS/Bedrock vars from this repo only."""
    load_dotenv(ROOT / ".env.local")
    load_dotenv(ROOT / ".env")


def _has_bedrock_creds() -> bool:
    if os.getenv("AWS_BEARER_TOKEN_BEDROCK"):
        return True
    if os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"):
        return True
    if (os.getenv("AWS_PROFILE") or "").strip():
        return True
    return False


def judge_model_id() -> str:
    return os.getenv("BEDROCK_JUDGE_MODEL") or DEFAULT_OPUS_MODEL


def get_bedrock_client():
    """Bedrock Runtime client. Prefer explicit keys, then profile, then default chain.

    Default chain includes AWS_BEARER_TOKEN_BEDROCK (self-serve-backend .env).
    """
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    with _LOCK:
        if _CLIENT is not None:
            return _CLIENT
        region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
        access_key = os.getenv("AWS_ACCESS_KEY_ID")
        secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        profile = (os.getenv("AWS_PROFILE") or "").strip()
        if access_key and secret_key:
            session = boto3.Session(
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                region_name=region,
            )
            _CLIENT = session.client("bedrock-runtime", config=_BEDROCK_CONFIG)
        elif profile:
            session = boto3.Session(profile_name=profile, region_name=region)
            _CLIENT = session.client("bedrock-runtime", config=_BEDROCK_CONFIG)
        else:
            _CLIENT = boto3.client(
                "bedrock-runtime",
                region_name=region,
                config=_BEDROCK_CONFIG,
            )
        return _CLIENT


def _extract_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON found in Bedrock response: {text[:200]}")
    return json.loads(text[start : end + 1])


def call_bedrock_json(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str | None = None,
    temperature: float | None = 0.0,
    max_tokens: int = 1024,
) -> tuple[dict[str, Any], dict[str, int]]:
    """converse() + JSON parse. Returns (parsed, token usage)."""
    client = get_bedrock_client()
    model_id = model or judge_model_id()
    last: Exception | None = None
    backoff = 1.0
    response = None
    # Opus 5 turns adaptive thinking on by default. Disable it so the
    # judge stays a short JSON score, not a long reasoning trace.
    extra_fields: dict[str, Any] | None = {"thinking": {"type": "disabled"}}
    for attempt in range(4):
        try:
            inference: dict[str, Any] = {"maxTokens": max_tokens}
            if temperature is not None:
                inference["temperature"] = temperature
            kwargs: dict[str, Any] = {
                "modelId": model_id,
                "messages": [{"role": "user", "content": [{"text": user_prompt}]}],
                "system": [{"text": system_prompt}],
                "inferenceConfig": inference,
            }
            if extra_fields:
                kwargs["additionalModelRequestFields"] = extra_fields
            response = client.converse(**kwargs)
            break
        except ClientError as exc:
            last = exc
            code = exc.response.get("Error", {}).get("Code", "")
            message = str(exc)
            if extra_fields and (
                "thinking" in message.lower() or "temperature" in message.lower()
            ):
                extra_fields = None
                temperature = None
                continue
            if code in (
                "ThrottlingException",
                "ServiceUnavailableException",
                "ModelStreamErrorException",
            ) and attempt < 3:
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)
                continue
            raise
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt < 3:
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)
                continue
            raise
    if response is None:
        raise RuntimeError(f"Bedrock converse failed: {last}")
    chunks = []
    for block in response.get("output", {}).get("message", {}).get("content") or []:
        if isinstance(block, dict) and block.get("text"):
            chunks.append(block["text"])
    text = "\n".join(chunks)
    parsed = _extract_json(text)
    usage = response.get("usage") or {}
    tokens = {
        "input_tokens": int(usage.get("inputTokens") or 0),
        "output_tokens": int(usage.get("outputTokens") or 0),
    }
    return parsed, tokens
