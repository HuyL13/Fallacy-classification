"""Check provenance and references, never claim to verify semantic truth."""
from .schemas import ArbiterResult, Hypothesis, Pattern, Revision


def quote_check(quote, target, required=False):
    if quote is None and not required:
        return
    if not isinstance(quote, str) or not quote.strip() or quote not in target:
        raise ValueError('Evidence must be a nonempty verbatim target substring')


def assess_check(assessment, target):
    quote_check(assessment.quote, target, assessment.status == 'supported')


def validate_hypothesis(h: Hypothesis, pattern: Pattern, target: str):
    if h.label != pattern.label:
        raise ValueError('Builder changed assigned label')
    ids = [a.criterion_id for a in h.criteria]
    if len(set(ids)) != len(ids) or set(ids) != {c.id for c in pattern.criteria}:
        raise ValueError('Builder must cover each own criterion exactly once')
    for assessment in h.criteria:
        assess_check(assessment, target)
    quote_check(h.strongest_quote, target)


def validate_revision(r: Revision, own: Hypothesis, opponent: Hypothesis, target: str):
    if r.label != own.label or r.opponent_label != opponent.label:
        raise ValueError('Revision changed assigned pair')
    own_by_id = {a.criterion_id: a for a in own.criteria}
    other_ids = {a.criterion_id for a in opponent.criteria}
    ids = [a.criterion_id for a in r.revisions]
    if len(set(ids)) != len(ids) or not set(ids) <= set(own_by_id):
        raise ValueError('Revision can change only distinct own criteria')
    for assessment in r.revisions:
        assess_check(assessment, target)
        if assessment == own_by_id[assessment.criterion_id]:
            raise ValueError('Unchanged assessment is not a revision')
    if any(c.opponent_criterion_id not in other_ids for c in r.criticisms):
        raise ValueError('Criticism references nonexistent opponent criterion')
    previous_quotes = {a.quote for h in (own, opponent) for a in h.criteria if a.quote}
    previous_quotes.update(h.strongest_quote for h in (own, opponent) if h.strongest_quote)
    if len(set(r.new_target_quotes)) != len(r.new_target_quotes):
        raise ValueError('New quotes must be distinct')
    for quote in r.new_target_quotes:
        quote_check(quote, target, required=True)
        if quote in previous_quotes:
            raise ValueError('Previously supplied quote is not new evidence')


def validate_arbiter(a: ArbiterResult, p1: Pattern, p2: Pattern):
    ids = {c.id for p in (p1, p2) for c in p.criteria}
    if len(set(a.decisive_criteria)) != len(a.decisive_criteria) or not set(a.decisive_criteria) <= ids:
        raise ValueError('Arbiter referenced duplicate or nonexistent criterion')


def evidence_polarity(criterion, assessment):
    """Diagnostic only; not a numerical decision rule. Unknown stays unknown."""
    if criterion.id != assessment.criterion_id:
        raise ValueError('Criterion mismatch')
    if assessment.status == 'unknown':
        return None
    return (1 if criterion.kind == 'support' else -1) * (1 if assessment.status == 'supported' else -1)
