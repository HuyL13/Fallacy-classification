"""Offline scoring, paired article bootstrap and development calibration."""
import argparse
import json
from pathlib import Path

from src.io_utils import read_jsonl
from .calibration import CorrectnessCalibrator
from .data import load_samples
from .evaluation import evaluate, paired_bootstrap
from .schemas import Prediction


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,help='MG-SCD output directory')
    parser.add_argument('--compare-run',help='Paired second output directory (same selected IDs)')
    parser.add_argument('--fit-calibrator',help='Save dev-only calibrator; needs optional scikit-learn')
    parser.add_argument('--bootstrap-repeats',type=int,default=2000)
    args=parser.parse_args()
    directory=Path(args.run)
    manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    if manifest['mock']:
        raise ValueError('Synthetic traces cannot be used as scientific evaluation data')
    config=manifest['config']
    samples,groups,verification=load_samples(config['data_dir'],config['split'],config['context'])
    if verification != manifest['dataset_verification']:
        raise ValueError('Dataset differs from the recorded run')
    selected=set(manifest['selected_ids'])
    samples=[s for s in samples if s.sample_id in selected]
    if {s.sample_id for s in samples} != selected:
        raise ValueError('Selected samples are missing from dataset')
    predictions=[Prediction.model_validate(p) for p in read_jsonl(directory/'predictions.jsonl')] if (directory/'predictions.jsonl').exists() else []
    failures=[p['sample_id'] for p in read_jsonl(directory/'failures.jsonl')] if (directory/'failures.jsonl').exists() else []
    result=evaluate(samples,predictions,failures)
    if args.fit_calibrator:
        CorrectnessCalibrator.fit(samples,predictions,split=config['split']).save(args.fit_calibrator)
        result['calibrator_saved']=args.fit_calibrator
        result['calibration_note']='Fit on dev only; evaluate Brier/ECE on a separate held-out split.'
    if args.compare_run:
        other=Path(args.compare_run)
        other_manifest=json.loads((other/'manifest.json').read_text(encoding='utf-8'))
        if other_manifest['mock'] or set(other_manifest['selected_ids'])!=selected:
            raise ValueError('Comparison requires identical non-synthetic selected IDs')
        if other_manifest['dataset_verification'] != manifest['dataset_verification']:
            raise ValueError('Comparison requires identical dataset versions')
        second=[Prediction.model_validate(p) for p in read_jsonl(other/'predictions.jsonl')]
        result['paired_comparison']=paired_bootstrap(samples,predictions,second,groups,
                                                     repeats=args.bootstrap_repeats)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
