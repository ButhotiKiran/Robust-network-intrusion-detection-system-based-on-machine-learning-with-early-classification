from sklearn.metrics import classification_report, confusion_matrix
import numpy as np




def print_classification_report(y_true, y_pred, target_names=None):
print(classification_report(y_true, y_pred, target_names=target_names))
print('Confusion matrix:')
print(confusion_matrix(y_true, y_pred))




def compute_detection_speed_stats(decision_steps):
arr = np.array(decision_steps)
return {
'mean': float(np.mean(arr)),
'median': float(np.median(arr)),
'std': float(np.std(arr))
}