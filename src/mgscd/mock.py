"""Contract fixture only: these outputs do not classify text or measure model quality."""
import json


class MockBackend:
    def __init__(self, decision='H2'):
        self.calls = []
        self.decision = decision

    def complete(self, stage, system, user, sample_id):
        self.calls.append(dict(stage=stage, system=system, user=user, sample_id=sample_id))
        kind = stage.split('.')[0]
        def section(name, following):
            return json.loads(user.split(name + ':\n', 1)[1].split('\n' + following, 1)[0])
        if kind == 'screen':
            return dict(ranked_labels=['appeal_to_worse_problems','slippery_slope','false_dilemma'],
                        ordinal_scores=dict(appeal_to_worse_problems=.7,slippery_slope=.65,false_dilemma=.2),
                        ambiguity_flags=['label_boundary'], notes='Synthetic test fixture')
        if kind == 'builder':
            sample = section('SAMPLE_JSON', 'OWN_PATTERN:')
            pattern = section('OWN_PATTERN', 'REQUIRED JSON OBJECT:')
            return dict(label=pattern['label'], premise='Synthetic premise', conclusion='Synthetic conclusion',
                        defect_mechanism='Synthetic mechanism; not a semantic judgment',
                        criteria=[dict(criterion_id=c['id'],status='unknown',quote=None,
                                       rationale='Fixture cannot determine support') for c in pattern['criteria']],
                        overall_support='weak',strongest_quote=sample['target_comment'])
        if kind == 'revision':
            own = section('OWN_PATTERN','OPPONENT_PATTERN:')
            other = section('OPPONENT_PATTERN','OWN_ASSESSMENT:')
            return dict(label=own['label'],opponent_label=other['label'],criticisms=[],revisions=[],
                        concedes_opponent_better=False,new_target_quotes=[],summary='No synthetic revision')
        if kind == 'arbiter':
            p = section('H1_PATTERN','H2_PATTERN:')
            return dict(decision=self.decision,decisive_criteria=[p['criteria'][0]['id']],
                        justification='Synthetic control decision, not scientific evidence',needs_more_context=False)
        raise ValueError(stage)
