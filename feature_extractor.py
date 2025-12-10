import numpy as np
import pandas as pd
from typing import List, Tuple




def make_basic_flow_features(df: pd.DataFrame, time_col: str | None = None) -> pd.DataFrame:
"""If dataset is already in flow format (rows = flows), return numeric-only features for ML.
Otherwise the user should aggregate raw packets to flow summaries prior to using this helper.
"""
numeric = df.select_dtypes(include=[np.number]).copy()
return numeric




def cumulative_prefixes(features: np.ndarray) -> List[np.ndarray]:
"""Given a 2D array representing a sequence of incremental sub-features for one flow,
return the cumulative prefixes used in early classification simulation.


Input shape: (T, F) where T = number of incremental steps available for a flow.
Output: list of arrays of shape (t, F) for t=1..T (or optionally aggregated to single-vector per t).
"""
prefixes = []
for t in range(1, features.shape[0] + 1):
# Simple aggregation: mean across first t rows -> single vector
agg = np.mean(features[:t, :], axis=0)
prefixes.append(agg)
return prefixes




def windowed_feature_vector(packet_seq: np.ndarray) -> np.ndarray:
"""Example: convert a sequence of packet-level features to a single fixed-size vector
by computing summary statistics (mean, std, min, max) per feature.


packet_seq shape: (T, F)
returns: (F*4,)
"""
mean = np.mean(packet_seq, axis=0)
std = np.std(packet_seq, axis=0)
mn = np.min(packet_seq, axis=0)
mx = np.max(packet_seq, axis=0)
return np.concatenate([mean, std, mn, mx])