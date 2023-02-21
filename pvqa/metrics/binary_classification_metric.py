from typing import Literal

import torch
import torchmetrics as tm
from torch import nn


class BinaryClassificationMetrics(nn.Module):
    def __init__(self, mode: Literal["train", "val", "test"]):
        super().__init__()
        self.mode = mode

        self.accuracy = tm.Accuracy(task="binary")
        self.precision = tm.Precision(task="binary")
        self.recall = tm.Recall(task="binary")
        self.f1_score = tm.F1Score(task="binary")
        self.auroc = tm.AUROC(task="binary")

        self.scalars = {"precision": self.precision, "recall": self.recall, "accuracy": self.accuracy,
                        "f1_score": self.f1_score, "auroc": self.auroc}

        self.cnfs_mat = tm.ConfusionMatrix(normalize="true", task="binary")
        self.roc = tm.ROC(task="binary")
        self.prc = tm.PrecisionRecallCurve(task="binary")

    def to(self, device: torch.device):
        for metric in self.scalars.values():
            metric.to(device)
        self.cnfs_mat.to(device)

    def forward(self, x, y):
        for metric in self.scalars.values():
            metric(x, y)
        self.cnfs_mat(x, y)
        self.roc(x, y)
        self.prc(x, y)
