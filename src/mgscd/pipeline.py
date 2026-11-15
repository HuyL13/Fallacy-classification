"""One screening, two independent builds, one simultaneous revision round, one judge."""
import time
from dataclasses import dataclass
from typing import Protocol

from .patterns import load_bank
from .prompts import SYSTEM, render
from .schemas import ArbiterResult, Hypothesis, Prediction, Revision, Sample, ScreenResult
from .validators import validate_arbiter, validate_hypothesis, validate_revision

STAGE_MODELS = {'screen': ScreenResult, 'builder': Hypothesis,
                'revision': Revision, 'arbiter': ArbiterResult}


class JSONLLM(Protocol):
    def complete(self, stage: str, system: str, user: str, sample_id: str) -> dict: ...


@dataclass(frozen=True)
class PipelineConfig:
    route_mode: str = 'debate_all'
    margin_threshold: float = 0.12  # Placeholder, must tune on dev before scientific use.
    article_max_chars: int = 6000
    variant: str = 'full'

    def __post_init__(self):
        if self.route_mode not in ('debate_all', 'margin_or_flags'):
            raise ValueError('Unknown route mode')
        if type(self.margin_threshold) not in (int, float) or not 0 <= self.margin_threshold <= 1:
            raise ValueError('Invalid margin threshold')
        if type(self.article_max_chars) is not int or self.article_max_chars < 0:
            raise ValueError('article_max_chars must be a nonnegative integer')
        if self.variant not in ('full', 'builders_aggregate', 'no_revision', 'no_counterchecks'):
            raise ValueError('Unknown ablation variant')


class Pipeline:
    def __init__(self, llm: JSONLLM, config=None, bank=None):
        self.llm = llm
        self.config = config or PipelineConfig()
        self.bank = (bank or load_bank()[0]).model_copy(deep=True)
        if self.config.variant == 'no_counterchecks':
            for pattern in self.bank.patterns:
                pattern.criteria = [c for c in pattern.criteria if c.kind != 'countercheck']
        self.patterns = {p.label: p for p in self.bank.patterns}

    def predict(self, sample: Sample) -> Prediction:
        start = time.monotonic()
        calls = 0
        data = sample.model_input(self.config.article_max_chars)
        def call(stage, role='', **values):
            nonlocal calls
            calls += 1
            output = self.llm.complete(stage + ('.' + role if role else ''), SYSTEM,
                                       render(stage, SAMPLE_JSON=data, **values), sample.sample_id)
            return STAGE_MODELS[stage].model_validate(output)
        # A failed screen is an explicit failed sample, never a fabricated class.
        screen = call('screen', ALL_PATTERNS=self.bank.model_dump())
        h1, h2 = screen.ranked_labels[:2]
        p1, p2 = self.patterns[h1], self.patterns[h2]
        pred = Prediction(sample_id=sample.sample_id, label=h1, initial_label=h1,
                          runner_up=h2, route='fast', screen=screen,
                          article_truncated=len(sample.article) > self.config.article_max_chars,
                          article_original_chars=len(sample.article), article_used_chars=len(data['article']))
        margin = screen.ordinal_scores[h1] - screen.ordinal_scores[h2]
        debate = (self.config.route_mode == 'debate_all' or
                  margin < self.config.margin_threshold or bool(screen.ambiguity_flags))
        try:
            if debate:
                pred.route = 'debate'
                built = call('builder', 'H1', OWN_PATTERN=p1)
                validate_hypothesis(built, p1, sample.target_comment)
                pred.hypothesis_h1 = built
                built = call('builder', 'H2', OWN_PATTERN=p2)
                validate_hypothesis(built, p2, sample.target_comment)
                pred.hypothesis_h2 = built
                a1, a2 = pred.hypothesis_h1, pred.hypothesis_h2
                if self.config.variant == 'builders_aggregate':
                    # Declared B2 baseline: ordinal overall-support comparison, ties retain H1.
                    ranks = {'weak': 0, 'mixed': 1, 'strong': 2}
                    pred.label = h2 if ranks[a2.overall_support] > ranks[a1.overall_support] else h1
                else:
                    if self.config.variant != 'no_revision':
                        revised = call('revision', 'H1', OWN_PATTERN=p1, OPPONENT_PATTERN=p2,
                                                OWN_ASSESSMENT=a1, OPPONENT_ASSESSMENT=a2)
                        validate_revision(revised, a1, a2, sample.target_comment)
                        pred.revision_h1 = revised
                        revised = call('revision', 'H2', OWN_PATTERN=p2, OPPONENT_PATTERN=p1,
                                                OWN_ASSESSMENT=a2, OPPONENT_ASSESSMENT=a1)
                        validate_revision(revised, a2, a1, sample.target_comment)
                        pred.revision_h2 = revised
                    judged = call('arbiter', H1_PATTERN=p1, H2_PATTERN=p2,
                                        H1_ASSESSMENT=a1, H2_ASSESSMENT=a2,
                                        H1_REVISION=pred.revision_h1, H2_REVISION=pred.revision_h2)
                    validate_arbiter(judged, p1, p2)
                    pred.arbiter = judged
                    pred.judge_decision = pred.arbiter.decision
                    pred.unresolved = pred.arbiter.decision == 'UNRESOLVED'
                    pred.label = h2 if pred.arbiter.decision == 'H2' else h1
        except Exception as exc:
            pred.label = h1
            pred.route, pred.degraded = 'degraded', True
            pred.error = f'{type(exc).__name__}: {exc}'
        pred.stage_calls = calls
        pred.latency_seconds = time.monotonic() - start
        return pred
