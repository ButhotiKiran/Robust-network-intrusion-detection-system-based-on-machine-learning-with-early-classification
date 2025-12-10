import pandas as pd
from typing import Tuple
def load_csv(path: str, max_rows: int | None = None) -> pd.DataFrame:
df = pd.read_csv(path)
if max_rows is not None:
df = df.head(max_rows)
return df




def split_features_label(df: pd.DataFrame, label_col: str = 'label') -> Tuple[pd.DataFrame, pd.Series]:
if label_col not in df.columns:
alt = [c for c in df.columns if c.lower() == 'label' or c.lower()=='class']
if alt:
label_col = alt[0]
else:
raise ValueError("No label column found. Ensure a 'label' or 'Label' column exists.")
X = df.drop(columns=[label_col])
y = df[label_col]
return X, ya