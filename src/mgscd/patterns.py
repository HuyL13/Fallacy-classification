import hashlib
from pathlib import Path
from .schemas import PatternBank

DEFAULT_BANK = Path(__file__).with_suffix('') / 'eight_labels.v1.json'


def load_bank(path=DEFAULT_BANK):
    raw = Path(path).read_bytes()
    return PatternBank.model_validate_json(raw), hashlib.sha256(raw).hexdigest()
