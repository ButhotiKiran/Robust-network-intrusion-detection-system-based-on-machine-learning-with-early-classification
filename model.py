from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import numpy as np
from typing import Tuple




def train_random_forest(X: np.ndarray, y: np.ndarray, save_path: str | None = None) -> RandomForestClassifier:
clf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
clf.fit(X, y)
if save_path:
joblib.dump(clf, save_path)
return clf




def load_model(path: str):
return joblib.load(path)




def quick_train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)




def predict_proba_batch(model, X: np.ndarray) -> np.ndarray:
return model.predict_proba(X)