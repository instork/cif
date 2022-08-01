import torch
import numpy as np
from transformers.trainer_utils import EvalPrediction
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from typing import Dict


def compute_metrics(p: EvalPrediction) -> Dict:
    predictions, labels = p
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(labels, axis=1)
    return {
        'precision': precision_score(y_true=y_true, y_pred=y_pred, average='micro'),
        'recall': recall_score(y_true=y_true, y_pred=y_pred, average='micro'),
        'f1': f1_score(y_true=y_true, y_pred=y_pred, average='micro'),
        'accuracy': accuracy_score(y_true=y_true, y_pred=y_pred),
    }

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
