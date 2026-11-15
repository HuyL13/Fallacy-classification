"""Repository-local dotenv loading and explicit provider endpoint overrides."""
import os
from pathlib import Path

from dotenv import load_dotenv

from src.llm.config import ModelConfig


def load_environment(root):
    """Match the existing runner: shell > .env > .env2; never log values."""
    for name in ('.env', '.env2'):
        load_dotenv(Path(root) / name, override=False)


def resolve_model_config(settings):
    settings = dict(settings)
    endpoint_variable = settings.pop('base_url_env', None)
    if endpoint_variable is not None:
        if not isinstance(endpoint_variable, str) or not endpoint_variable.strip():
            raise ValueError('base_url_env must be a nonempty environment variable name')
        endpoint = os.environ.get(endpoint_variable)
        if endpoint:
            settings['base_url'] = endpoint
    return ModelConfig(**settings)
