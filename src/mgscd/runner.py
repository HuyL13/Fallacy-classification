"""Isolated MG-SCD entry point: python -m src.mgscd.runner --config ..."""
import argparse
import json
import platform
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import yaml

from src.io_utils import append_jsonl, digest, file_hash, write_json
from src.llm.config import ModelConfig
from .backend import ProviderBackend
from .calibration import CorrectnessCalibrator
from .data import load_samples
from .evaluation import evaluate
from .mock import MockBackend
from .patterns import DEFAULT_BANK, load_bank
from .pipeline import Pipeline, PipelineConfig
from .prompts import PROMPT_VERSION
from .schemas import Sample

ROOT = Path(__file__).resolve().parents[2]
KEYS = {'task','data_dir','split','context','pipeline','model','pattern_bank','output_dir'}


def load_config(path):
    path=Path(path).resolve()
    config=yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(config,dict) or set(config)!=KEYS or config['task']!='classification':
        raise ValueError('MG-SCD config must be eight-way classification with keys: '+str(sorted(KEYS)))
    if config['split'] not in ('train','dev','test') or config['context'] not in ('paper','article','comment_only'):
        raise ValueError('Unknown split/context')
    PipelineConfig(**config['pipeline'])
    ModelConfig(**config['model'])
    base=path.parent.parent
    for key in ('data_dir','pattern_bank','output_dir'):
        if not isinstance(config[key],str) or not config[key]:
            raise ValueError(key+' must be a path')
        value=Path(config[key])
        config[key]=str(value if value.is_absolute() else (base/value).resolve())
    return config


def run(config, *, mock=False, limit=None, calibrator_path=None):
    if limit is not None and (type(limit) is not int or limit<1):
        raise ValueError('limit must be positive')
    bank,pattern_hash=load_bank(config.get('pattern_bank',DEFAULT_BANK))
    pipeline_config=PipelineConfig(**config.get('pipeline',{}))
    model_config=ModelConfig(**config['model'])
    if model_config.provider=='mock' and not mock:
        raise ValueError('Mock provider requires --mock acknowledgment')
    if mock:
        samples=[Sample(sample_id='mock-001',target_comment='If we permit X, then Y, and eventually Z will happen.',
                        gold_label='slippery_slope')]
        groups={'mock-001':'synthetic'}
        verification={'synthetic':True}
    else:
        samples,groups,verification=load_samples(config['data_dir'],config['split'],config['context'])
    if limit:
        samples=samples[:limit]
    if not samples:
        raise ValueError('No classification samples selected')
    output=Path(config['output_dir'])
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory is not empty; use a new versioned directory')
    output.mkdir(parents=True,exist_ok=True)
    backend=MockBackend() if mock else ProviderBackend(model_config,output/'raw_calls.jsonl',
                                                       pattern_version=bank.version,pattern_hash=pattern_hash)
    pipeline=Pipeline(backend,pipeline_config,bank)
    calibrator=CorrectnessCalibrator.load(calibrator_path) if calibrator_path else None
    if calibrator and set(calibrator.fitted_sample_ids)&{s.sample_id for s in samples}:
        raise ValueError('Calibration fit/evaluation IDs overlap')
    sources={str(p.relative_to(ROOT)):file_hash(p) for p in sorted((ROOT/'src').rglob('*'))
             if p.suffix in ('.py','.txt','.json')}
    baseline_paths=['src/grounded_classification.py','src/adversarial_debate.py','src/adversarial_prompts.py',
                    'src/adversarial_schemas.py','src/engine.py','src/llm/client.py',
                    'configs/classification.adversarial.grounded-v3.nvidia.yaml','docs/GROUNDED_V3.md']
    manifest=dict(config=config,mock=mock,limit=limit,prompt_version=PROMPT_VERSION,
                  pattern_version=bank.version,pattern_hash=pattern_hash,source_hashes=sources,
                  effective_pattern_hash=digest(pipeline.bank.model_dump()),
                  baseline_hashes={p:file_hash(ROOT/p) for p in baseline_paths if (ROOT/p).exists()},
                  selected_ids=[s.sample_id for s in samples],
                  article_groups={s.sample_id:groups[s.sample_id] for s in samples},
                  dataset_verification=verification,python=platform.python_version(),
                  dependencies={name:version(name) for name in ('pydantic','PyYAML','jsonschema')},
                  pipeline=asdict(pipeline_config),calibrator_hash=file_hash(calibrator_path) if calibrator_path else None)
    write_json(output/'manifest.json',manifest)
    # Store exact resources alongside hashes for review/reproduction.
    write_json(output/'pattern_bank.json',pipeline.bank.model_dump())
    predictions,failures=[],[]
    started=time.monotonic()
    for sample in samples:
        try:
            prediction=pipeline.predict(sample)
            if calibrator:
                prediction.calibrated_confidence=calibrator.predict(prediction)
            predictions.append(prediction)
            append_jsonl(output/'predictions.jsonl',prediction.model_dump())
        except Exception as exc:
            failures.append(sample.sample_id)
            append_jsonl(output/'failures.jsonl',dict(sample_id=sample.sample_id,
                         error=type(exc).__name__+': '+str(exc)))
    result=evaluate(samples,predictions,failures)
    result['wall_seconds']=time.monotonic()-started
    if mock:
        result.update(mock=True,scientific_performance_evidence=False,
                      backend={'stage_calls':len(backend.calls),'tokens':None,'estimated_cost_usd':0.0})
    else:
        result['backend']=backend.summary()
    write_json(output/'metrics.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--mock',action='store_true',help='Synthetic contract exercise; never dataset performance')
    parser.add_argument('--limit',type=int)
    parser.add_argument('--calibrator')
    args=parser.parse_args()
    result=run(load_config(args.config),mock=args.mock,limit=args.limit,calibrator_path=args.calibrator)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 1 if result['failed'] else 0


if __name__=='__main__':
    raise SystemExit(main())
