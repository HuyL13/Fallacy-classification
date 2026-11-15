"""Strict stage contracts. UNRESOLVED is never a classification label."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

LABELS = (
    'appeal_to_authority', 'appeal_to_majority', 'appeal_to_nature',
    'appeal_to_tradition', 'appeal_to_worse_problems', 'false_dilemma',
    'hasty_generalization', 'slippery_slope',
)
Label = Literal[
    'appeal_to_authority', 'appeal_to_majority', 'appeal_to_nature',
    'appeal_to_tradition', 'appeal_to_worse_problems', 'false_dilemma',
    'hasty_generalization', 'slippery_slope',
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, allow_inf_nan=False)


class Sample(StrictModel):
    sample_id: str = Field(min_length=1)
    target_comment: str = Field(min_length=1)
    parent_comment: str = ''
    title: str = ''
    article: str = ''
    gold_label: Label | None = None

    @model_validator(mode='after')
    def nonblank(self):
        if not self.sample_id.strip() or not self.target_comment.strip():
            raise ValueError('sample ID and target must not be blank')
        return self

    def model_input(self, article_max_chars=6000):
        return {**self.model_dump(exclude={'gold_label', 'article'}),
                'article': self.article[:article_max_chars]}


class Criterion(StrictModel):
    id: str = Field(min_length=1)
    kind: Literal['support', 'countercheck']
    statement: str = Field(min_length=1)


class Pattern(StrictModel):
    label: Label
    definition: str = Field(min_length=1)
    mechanism: str = Field(min_length=1)
    logical_form: str = Field(min_length=1)
    cue_examples: list[str]
    criteria: list[Criterion] = Field(min_length=1)
    contrasts: list[Label]
    source_note: str = Field(min_length=1)


class PatternBank(StrictModel):
    version: str = Field(min_length=1)
    patterns: list[Pattern]

    @model_validator(mode='after')
    def valid_bank(self):
        if len(self.patterns) != 8 or {p.label for p in self.patterns} != set(LABELS):
            raise ValueError('Pattern bank must contain exactly eight labels')
        ids = [c.id for p in self.patterns for c in p.criteria]
        if len(ids) != len(set(ids)):
            raise ValueError('Criterion IDs must be globally unique')
        for p in self.patterns:
            if p.label in p.contrasts or len(p.contrasts) != len(set(p.contrasts)):
                raise ValueError('Invalid contrast labels')
            if not any(c.kind == 'support' for c in p.criteria):
                raise ValueError('Every pattern needs a support criterion')
        return self


class ScreenResult(StrictModel):
    ranked_labels: list[Label] = Field(min_length=3, max_length=8)
    ordinal_scores: dict[Label, float]
    ambiguity_flags: list[str]
    notes: str

    @model_validator(mode='after')
    def valid_ranking(self):
        labels = self.ranked_labels
        if len(set(labels)) != len(labels) or set(labels) != set(self.ordinal_scores):
            raise ValueError('Distinct ranked labels must match score keys exactly')
        if any(not 0 <= score <= 1 for score in self.ordinal_scores.values()):
            raise ValueError('Ordinal scores must be finite and within [0,1]')
        expected = sorted(labels, key=lambda label: (-self.ordinal_scores[label], label))
        if labels != expected:
            raise ValueError('Scores must descend; tied labels must use canonical ID order')
        return self


class CriterionAssessment(StrictModel):
    criterion_id: str = Field(min_length=1)
    status: Literal['supported', 'refuted', 'unknown']
    quote: str | None
    rationale: str = Field(min_length=1)


class Hypothesis(StrictModel):
    label: Label
    premise: str = Field(min_length=1)
    conclusion: str = Field(min_length=1)
    defect_mechanism: str = Field(min_length=1)
    criteria: list[CriterionAssessment]
    overall_support: Literal['strong', 'mixed', 'weak']
    strongest_quote: str | None


class Criticism(StrictModel):
    opponent_criterion_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class Revision(StrictModel):
    label: Label
    opponent_label: Label
    criticisms: list[Criticism]
    revisions: list[CriterionAssessment]
    concedes_opponent_better: bool
    new_target_quotes: list[str]
    summary: str


class ArbiterResult(StrictModel):
    decision: Literal['H1', 'H2', 'UNRESOLVED']
    decisive_criteria: list[str]
    justification: str = Field(min_length=1)
    needs_more_context: bool


class Prediction(StrictModel):
    sample_id: str
    label: Label
    initial_label: Label
    runner_up: Label
    route: Literal['fast', 'debate', 'degraded']
    judge_decision: Literal['H1', 'H2', 'UNRESOLVED'] | None = None
    unresolved: bool = False
    degraded: bool = False
    error: str | None = None
    screen: ScreenResult
    hypothesis_h1: Hypothesis | None = None
    hypothesis_h2: Hypothesis | None = None
    revision_h1: Revision | None = None
    revision_h2: Revision | None = None
    arbiter: ArbiterResult | None = None
    calibrated_confidence: float | None = Field(default=None, ge=0, le=1)
    article_truncated: bool = False
    article_original_chars: int = 0
    article_used_chars: int = 0
    latency_seconds: float = 0.0
    stage_calls: int = 0
