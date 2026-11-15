"""Optional dev-only logistic correctness calibration; stdlib inference."""
import json
import math
from dataclasses import dataclass
from pathlib import Path


def features(p):
    scores=p.screen.ordinal_scores
    return [scores[p.initial_label],scores[p.initial_label]-scores[p.runner_up],
            float(p.label!=p.initial_label),float(p.route!='fast'),float(p.unresolved),float(p.degraded)]


@dataclass
class CorrectnessCalibrator:
    coefficients: list[float]
    intercept: float
    fitted_sample_ids: list[str]

    @classmethod
    def fit(cls, dev_samples, dev_predictions, *, split):
        if split != 'dev':
            raise ValueError('Calibrator may be fitted only on development predictions')
        from sklearn.linear_model import LogisticRegression
        samples={s.sample_id:s for s in dev_samples}
        if len(samples)!=len(dev_samples) or {p.sample_id for p in dev_predictions}!=set(samples) or len(dev_predictions)!=len(samples):
            raise ValueError('Calibration requires exact dev coverage')
        if any(s.gold_label is None for s in dev_samples):
            raise ValueError('Calibration needs dev gold labels')
        y=[int(p.label==samples[p.sample_id].gold_label) for p in dev_predictions]
        if len(set(y)) != 2:
            raise ValueError('Calibration requires both correct and incorrect dev predictions')
        model=LogisticRegression(random_state=0).fit([features(p) for p in dev_predictions],y)
        return cls(model.coef_[0].tolist(),float(model.intercept_[0]),sorted(samples))

    def predict(self, prediction):
        if prediction.sample_id in self.fitted_sample_ids:
            raise ValueError('Use held-out predictions to assess calibration')
        if len(self.coefficients)!=6 or not all(math.isfinite(v) for v in [*self.coefficients,self.intercept]):
            raise ValueError('Invalid calibrator parameters')
        z=self.intercept+sum(w*x for w,x in zip(self.coefficients,features(prediction)))
        return 1/(1+math.exp(-z)) if z>=0 else math.exp(z)/(1+math.exp(z))

    def save(self,path):
        Path(path).write_text(json.dumps(self.__dict__,allow_nan=False)+'\n',encoding='utf-8')

    @classmethod
    def load(cls,path):
        return cls(**json.loads(Path(path).read_text(encoding='utf-8')))
