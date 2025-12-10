import numpy as np
from typing import Callable, List, Tuple




class EarlyClassifierSimulator:
"""Simulates streaming/early classification on per-flow incremental features.


Strategy implemented:
- At each time-step t we compute an aggregated vector over the prefix (e.g. mean/std)
- Pass aggregated vector to the trained classifier to get class probabilities
- If highest class probability >= threshold, we stop and output that class
- Otherwise, move to next time step. If end reached, output final decision.
"""


def __init__(self, model, threshold: float = 0.9, max_steps: int | None = None):
self.model = model
self.threshold = threshold
self.max_steps = max_steps


def decision_on_sequence(self, prefix_vectors: List[np.ndarray]) -> Tuple[int, int, float]:
"""Run early decision on a list of aggregated vectors for t=1..T


Returns (predicted_label, decision_time_step, confidence)
"""
T = len(prefix_vectors)
if self.max_steps is not None:
T = min(T, self.max_steps)
for t in range(T):
x = prefix_vectors[t].reshape(1, -1)
probs = self.model.predict_proba(x)[0]
pred = int(np.argmax(probs))
conf = float(np.max(probs))
if conf >= self.threshold:
return pred, t + 1, conf
# fallback: final decision
x = prefix_vectors[-1].reshape(1, -1)
probs = self.model.predict_proba(x)[0]
pred = int(np.argmax(probs))
conf = float(np.max(probs))
return pred, len(prefix_vectors), conf




def simulate_dataset_early_detection(sequences: List[List[np.ndarray]], labels: List[int], model, threshold: float = 0.9) -> dict:
"""Given sequences where each sequence is a list of aggregated vectors per time-step,
run early classifier and produce metrics including mean decision step.
"""
sim = EarlyClassifierSimulator(model, threshold=threshold)
preds = []
decision_steps = []
confidences = []
for seq, lbl in zip(sequences, labels):
pred, step, conf = sim.decision_on_sequence(seq)
preds.append(pred)
decision_steps.append(step)
confidences.append(conf)
preds = np.array(preds)
decision_steps = np.array(decision_steps)
confidences = np.array(confidences)
acc = np.mean(preds == np.array(labels))
return {
'accuracy': float(acc),
'mean_decision_step': float(np.mean(decision_steps)),
'median_decision_step': float(np.median(decision_steps)),
'confidences_mean': float(np.mean(confidences))
}