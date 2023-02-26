from typing import List, Literal

import torch
import torchmetrics as tm
from torch import nn

from pvqa.metrics.classification_report import ClassificationReport


class MulticlassClassificationMetrics(nn.Module):
    def __init__(self, num_classes: int,
                 mode: Literal["train", "val", "test"],
                 labels: List[str]):
        super().__init__()
        self.num_classes = num_classes
        self.mode = mode

        self.macro_f1 = tm.F1Score(num_classes=self.num_classes, average="macro", task="multiclass")
        self.accuracy = tm.Accuracy(num_classes=self.num_classes, task="multiclass")
        self.macorcoef = tm.MatthewsCorrCoef(num_classes=self.num_classes, task="multiclass")

        self.scalars = {"f1_score": self.macro_f1, "accuracy": self.accuracy, "mcc": self.macorcoef}

        self.cnfs_mat = tm.ConfusionMatrix(num_classes=self.num_classes, normalize="true", task="multiclass")
        self.class_report = ClassificationReport(num_classes=self.num_classes, labels=labels)

    def to(self, device: torch.device):
        for metric in self.scalars.values():
            metric.to(device)
        self.cnfs_mat.to(device)
        self.class_report.to(device)

    def forward(self, x, y):
        for metric in self.scalars.values():
            metric(x, y)
        stat_scores = tm.functional.classification.stat_scores(x, y, num_classes=self.num_classes, average="none",
                                                               task="multiclass")
        self.cnfs_mat(x, y)
        self.class_report.update(stat_scores)
