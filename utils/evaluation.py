from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment as linear_assignment
import numpy as np

def _make_cost_m(cm):
    s = np.max(cm)
    return (- cm + s)

def cluster_accuracy(labels, predicted_labels):
    cm = confusion_matrix(labels, predicted_labels)
    indexes = linear_assignment(_make_cost_m(cm))
    js = [e[1] for e in sorted(indexes, key=lambda x: x[0])]
    cm2 = cm[:, js]
    acc = np.trace(cm2) / np.sum(cm2)
    return acc
