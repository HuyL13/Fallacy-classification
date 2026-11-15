"""Offline evaluation: gold is used here only. Failed screens stay in denominators."""
import random
from .schemas import LABELS


def evaluate(samples, predictions, failures=()):
    by_id = {s.sample_id: s for s in samples}
    pred = {p.sample_id: p for p in predictions}
    failed = set(failures)
    if len(by_id) != len(samples) or len(pred) != len(predictions):
        raise ValueError('Duplicate sample/prediction IDs')
    if set(pred) & failed or set(pred) | failed != set(by_id):
        raise ValueError('Prediction/failure coverage must match input exactly')
    if not samples or any(s.gold_label is None for s in samples):
        raise ValueError('Evaluation requires nonempty labeled samples')
    n = len(samples)
    correct = initial_correct = beneficial = harmful = flips = top2 = top3 = eligible = corrected = 0
    initial_wrong = gate_missed = 0
    for sid, s in by_id.items():
        if sid not in pred:
            continue
        p = pred[sid]
        right, before = p.label == s.gold_label, p.initial_label == s.gold_label
        correct += right
        initial_correct += before
        beneficial += not before and right
        harmful += before and not right
        flips += p.label != p.initial_label
        top2 += s.gold_label in p.screen.ranked_labels[:2]
        top3 += s.gold_label in p.screen.ranked_labels[:3]
        initial_wrong += not before
        gate_missed += not before and p.route == 'fast'
        if not before and s.gold_label in p.screen.ranked_labels[:2]:
            eligible += 1
            corrected += right
    per_class = {}
    for label in LABELS:
        tp = sum(s.gold_label == label and sid in pred and pred[sid].label == label for sid,s in by_id.items())
        support = sum(s.gold_label == label for s in samples)
        predicted = sum(p.label == label for p in predictions)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        per_class[label] = dict(precision=precision,recall=recall,
            f1=2*precision*recall/(precision+recall) if precision+recall else 0.0,support=support)
    result = dict(n=n,completed=len(predictions),failed=len(failed),accuracy=correct/n,
        initial_accuracy=initial_correct/n,macro_f1=sum(c['f1'] for c in per_class.values())/8,
        per_class=per_class,top2_recall=top2/n,top3_recall=top3/n,
        beneficial_flips=beneficial,harmful_flips=harmful,net_correction=beneficial-harmful,
        delta_accuracy=(beneficial-harmful)/n,flip_rate=flips/n,
        debate_activation=sum(p.route == 'debate' for p in predictions)/n,
        debate_attempt_rate=sum(p.route != 'fast' for p in predictions)/n,
        degraded_rate=sum(p.degraded for p in predictions)/n,unresolved_rate=sum(p.unresolved for p in predictions)/n,
        oracle_top2_coverage=eligible/initial_wrong if initial_wrong else None,
        eligible_correction_accuracy=corrected/eligible if eligible else None,
        gate_miss_rate=gate_missed/initial_wrong if initial_wrong else None,
        mean_latency_seconds=sum(p.latency_seconds for p in predictions)/len(predictions) if predictions else None,
        stage_calls=sum(p.stage_calls for p in predictions))
    revisions = [(h,r) for p in predictions for h,r in
                 ((p.hypothesis_h1,p.revision_h1),(p.hypothesis_h2,p.revision_h2)) if h and r]
    status_changes = sum(a.status != next(old.status for old in h.criteria if old.criterion_id==a.criterion_id)
                         for h,r in revisions for a in r.revisions)
    result['revision_diagnostics'] = dict(
        validated_revisions=len(revisions), criterion_status_changes=status_changes,
        assessed_criteria=sum(len(h.criteria) for h,r in revisions),
        valid_new_quotes=sum(len(r.new_target_quotes) for h,r in revisions),
        concessions=sum(r.concedes_opponent_better for h,r in revisions),
        concession_rate=sum(r.concedes_opponent_better for h,r in revisions)/len(revisions) if revisions else None,
        changed_sample_beneficial_flips=sum(
            p.initial_label!=by_id[p.sample_id].gold_label and p.label==by_id[p.sample_id].gold_label
            for p in predictions if any(r and r.revisions for r in (p.revision_h1,p.revision_h2))))
    calibrated = [p for p in predictions if p.calibrated_confidence is not None]
    if calibrated:
        pairs=[(p.calibrated_confidence, int(p.label==by_id[p.sample_id].gold_label)) for p in calibrated]
        ece=0.0
        for i in range(10):
            group=[(c,y) for c,y in pairs if min(int(c*10),9)==i]
            if group:
                ece += abs(sum(c-y for c,y in group))/len(pairs)
        result['calibration'] = dict(n=len(pairs),brier=sum((c-y)**2 for c,y in pairs)/len(pairs),ece=ece)
    return result


def paired_bootstrap(samples, first, second, groups=None, repeats=2000, seed=0):
    """Paired accuracy delta; resample whole articles when group IDs are supplied."""
    a,b = {p.sample_id:p for p in first},{p.sample_id:p for p in second}
    ids = {s.sample_id for s in samples}
    if not ids or set(a) != ids or set(b) != ids or len(a)!=len(first) or len(b)!=len(second):
        raise ValueError('Bootstrap requires complete paired coverage')
    if repeats < 1:
        raise ValueError('repeats must be positive')
    clusters={}
    for s in samples:
        if s.gold_label is None:
            raise ValueError('Gold required offline')
        group = groups[s.sample_id] if groups is not None else s.sample_id
        delta = int(b[s.sample_id].label==s.gold_label)-int(a[s.sample_id].label==s.gold_label)
        clusters.setdefault(group,[]).append(delta)
    rng=random.Random(seed)
    keys=sorted(clusters)
    draws=[]
    for _ in range(repeats):
        values=[v for key in rng.choices(keys,k=len(keys)) for v in clusters[key]]
        draws.append(sum(values)/len(values))
    draws.sort()
    return dict(delta_accuracy=sum(v for c in clusters.values() for v in c)/len(samples),
                lower_95=draws[int(.025*(repeats-1))],upper_95=draws[int(.975*(repeats-1))],
                repeats=repeats,seed=seed,clustered=groups is not None)
