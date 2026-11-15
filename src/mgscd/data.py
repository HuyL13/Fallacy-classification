"""Preserve official splits, reject unknown labels, never re-split comments."""
import json
from pathlib import Path

from src.data.loader import load_split, plain_text, select_task
from src.data.verify import verify_dataset
from .schemas import LABELS, Sample


def load_samples(directory, split='dev', context='paper'):
    if split not in ('train', 'dev', 'test') or context not in ('paper','article','comment_only'):
        raise ValueError('Unknown split/context')
    verification = verify_dataset(directory)  # Includes cross-split article/comment checks.
    path = Path(directory) / f'{split}.json'
    raw = json.loads(path.read_text(encoding='utf-8'))
    if raw and 'comments' not in raw[0]:
        # Legacy flat loader maps unrecognized labels to none; reject them before conversion.
        allowed = {'none', *[label.replace('_',' ') for label in LABELS]}
        if any(str(r.get('gold',r.get('fallacy',r.get('label','none')))).lower() not in allowed for r in raw):
            raise ValueError('Unknown flat dataset label')
        if any('article_id' not in r for r in raw):
            raise ValueError('Flat records require explicit article_id to audit group separation')
    records = select_task(load_split(path), 'classification')
    samples = [Sample(sample_id=s.sample_id, target_comment=s.comment,
                      parent_comment=s.parent_comment if context != 'comment_only' else '',
                      title=s.title if context != 'comment_only' else '',
                      article=plain_text(s.article) if context == 'article' else '',
                      gold_label=s.gold('classification').lower().replace(' ','_')) for s in records]
    return samples, {s.sample_id: str(s.article_id) for s in records}, verification
