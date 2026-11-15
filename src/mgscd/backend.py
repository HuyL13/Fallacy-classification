"""Strict adapter to existing provider transports; no legacy output repairs."""
import json
import time
from dataclasses import asdict
from urllib.error import HTTPError, URLError

from src.io_utils import append_jsonl, digest
from src.llm.client import accumulate_usage, estimate_cost
from src.llm.gemini_client import GeminiTransport
from src.llm.openai_client import OpenAITransport
from .pipeline import STAGE_MODELS
from .prompts import PROMPT_VERSION


class ProviderBackend:
    def __init__(self, config, audit_path, *, pattern_version, pattern_hash, transport=None):
        if config.provider == 'mock' and transport is None:
            raise ValueError('Use MockBackend for mock runs')
        self.config, self.audit_path = config, audit_path
        self.pattern_version, self.pattern_hash = pattern_version, pattern_hash
        self.transport = transport or (GeminiTransport(config) if config.provider == 'gemini'
                                      else OpenAITransport(config))
        self.usage = dict(api_calls=0, prompt_tokens=0, completion_tokens=0, total_tokens=0,
                          provider_cached_tokens=0, missing_usage_responses=0)

    def complete(self, stage, system, user, sample_id):
        kind, _, role = stage.partition('.')
        model = STAGE_MODELS[kind]
        schema = model.model_json_schema()
        # Schema stays in system for providers without native structured output.
        payload = {'model': self.config.name, 'messages': [
            {'role': 'system', 'content': system + '\nREQUIRED SCHEMA:\n' + json.dumps(schema)},
            {'role': 'user', 'content': user}],
            'max_completion_tokens': self.config.max_completion_tokens}
        if self.config.temperature is not None:
            payload['temperature'] = self.config.temperature
        if self.config.structured_output:
            payload['response_format'] = {'type': 'json_schema', 'json_schema':
                                         {'name': 'mgscd_' + kind, 'strict': True, 'schema': schema}}
        for attempt in range(self.config.max_attempts):
            started = time.monotonic()
            event = dict(sample_id=sample_id, stage=kind, role=role or None,
                         model_id=self.config.name, model_revision=None, prompt_version=PROMPT_VERSION,
                         pattern_version=self.pattern_version, pattern_hash=self.pattern_hash,
                         temperature=self.config.temperature, attempt=attempt+1, retries=attempt,
                         input_tokens=None, output_tokens=None, status='error', request=payload)
            retry = True
            response = None
            self.usage['api_calls'] += 1
            try:
                response = self.transport.complete(payload)
                event['response'] = response
                event['response_hash'] = digest(response)
                event['model_revision'] = response.get('model')
                usage = response.get('usage')
                accumulate_usage(self.usage, usage)
                if isinstance(usage, dict):
                    for dest, source in (('input_tokens','prompt_tokens'),('output_tokens','completion_tokens')):
                        value = usage.get(source)
                        event[dest] = value if type(value) is int and value >= 0 else None
                choice = response['choices'][0]
                if choice.get('finish_reason') != 'stop' or choice['message'].get('refusal'):
                    raise ValueError('Refused, truncated, or non-final response')
                raw = choice['message']['content']
                # Strict JSON only: reject extra fields, wrappers and duplicate keys.
                def pairs(items):
                    result = {}
                    for key, value in items:
                        if key in result:
                            raise ValueError('Duplicate JSON key')
                        result[key] = value
                    return result
                parsed = json.loads(raw, object_pairs_hook=pairs)
                result = model.model_validate(parsed).model_dump()
                event['status'] = 'ok'
                return result
            except Exception as exc:
                if response is None:
                    self.usage['missing_usage_responses'] += 1
                event['error'] = type(exc).__name__ + ': ' + str(exc)
                if isinstance(exc, HTTPError):
                    retry = exc.code in (408, 409, 429) or 500 <= exc.code <= 599
                elif not isinstance(exc, (ValueError, KeyError, IndexError, TypeError,
                                          URLError, TimeoutError, ConnectionError)):
                    retry = False
                if not retry or attempt + 1 == self.config.max_attempts:
                    raise
            finally:
                event['latency_ms'] = (time.monotonic() - started) * 1000
                append_jsonl(self.audit_path, event)
            time.sleep(min(60, self.config.backoff_seconds * 2**attempt))
        raise RuntimeError('Unreachable')

    def summary(self):
        return {'model_config': asdict(self.config), 'usage': self.usage,
                'estimated_cost_usd': estimate_cost(self.config, self.usage)}
