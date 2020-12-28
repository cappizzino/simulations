#!/usr/bin/env python
import numpy as np
from sklearn.metrics import precision_recall_curve

#y_true = np.array([0, 0, 1, 1])
#y_scores = np.array([0.1, 0.4, 0.35, 0.8])
y_true = np.array([[0, 0],[1, 1]])
y_scores = np.array([[0.35, 0.8],[0.1, 0.4]])
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

print precision
print recall
print thresholds