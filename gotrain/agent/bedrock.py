"""Bedrock model factory. Credentials come from the SageMaker execution role."""

from __future__ import annotations

DEFAULT_MODEL_ID = "eu.anthropic.claude-sonnet-5-5"
DEFAULT_REGION = "eu-central-1"


def make_bedrock_model(model_id: str = DEFAULT_MODEL_ID, region: str = DEFAULT_REGION, max_tokens: int = 2048):
    """No sampling parameters are sent: recent Claude models reject non-default temperature."""
    from strands.models import BedrockModel

    return BedrockModel(model_id=model_id, region_name=region, max_tokens=max_tokens)
