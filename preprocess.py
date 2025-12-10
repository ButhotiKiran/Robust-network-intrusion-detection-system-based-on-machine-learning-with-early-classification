import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Tuple




def clean_df(df: pd.DataFrame) -> pd.DataFrame:
"""Basic cleaning: drop constant columns, fill nans.
User may customize for protocol fields, IPs, ports, timestamps, etc.
"""
df = df.copy()
# Drop columns with single unique value
nunique = df.nunique()
to_drop = nunique[nunique <= 1].index.tolist()
if to_drop:
df.drop(columns=to_drop, inplace=True)
df.fillna(0, inplace=True)
return df




def encode_labels(y: pd.Series) -> Tuple[np.ndarray, LabelEncoder]:
le = LabelEncoder()
y_enc = le.fit_transform(y)
return y_enc, le




def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
return X_train_scaled, X_test_scaled, scaler