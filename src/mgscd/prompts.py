"""Single-pass substitution: untrusted sample text cannot create template slots."""
import json
import re
from pathlib import Path

PROMPT_VERSION = 'mgscd-v1.0'
PROMPT_DIR = Path(__file__).with_suffix('')
SYSTEM = (PROMPT_DIR / 'system.txt').read_text(encoding='utf-8')


def render(stage, **values):
    template = (PROMPT_DIR / f'{stage}.txt').read_text(encoding='utf-8')
    def replace(match):
        key = match.group(1)
        value = values[key]
        if hasattr(value, 'model_dump'):
            value = value.model_dump()
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    return re.sub(r'\{\{([A-Z0-9_]+)\}\}', replace, template)
