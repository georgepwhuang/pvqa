from typing import List, Literal

import torch
import torchmetrics as tm
from sklearn import metrics
from pvqa.metrics.base import ClassificationMetric
from pvqa.metrics.classification_report import ClassificationReport


class MulticlassClassificationMetrics(ClassificationMetric):
    def __init__(self, num_classes: int,
                 mode: Literal["train", "val", "test"],
                 labels: List[str]):
        super().__init__()
        self.num_classes = num_classes
        self.mode = mode
        self.labels = labels

        self.macro_f1 = tm.F1Score(num_classes=self.num_classes, average="macro", task="multiclass")
        self.accuracy = tm.Accuracy(num_classes=self.num_classes, task="multiclass")
        self.macorcoef = tm.MatthewsCorrCoef(num_classes=self.num_classes, task="multiclass")

        self.cnfs_mat = tm.ConfusionMatrix(num_classes=self.num_classes, normalize="true", task="multiclass")
        self.class_report = ClassificationReport(num_classes=self.num_classes, labels=self.labels)

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

    @property
    def scalars(self):
        return {"f1_score": self.macro_f1, "accuracy": self.accuracy, "mcc": self.macorcoef}

    def nonscalars(self, current_epoch):
        return self.plot_confusion_matrix( current_epoch), self.write_classification_report()

    def plot_confusion_matrix(self, current_epoch):
        cf_matrix = self.cnfs_mat.compute().cpu().numpy()
        fig = metrics.ConfusionMatrixDisplay(cf_matrix, display_labels=self.labels).plot(values_format='.1%').figure_
        fig.set_size_inches(10, 10)
        fig.suptitle(f"Confusion Matrix, Epoch {current_epoch}")
        return {"name": "cnfs_mat", "type": "fig", "data": fig}

    def write_classification_report(self):
        report = self.class_report.compute()
        self.class_report.reset()
        return {"name": "class_rprt", "type": "text", "data": report}
