from typing import Literal

import torch
import torchmetrics as tm
from sklearn import metrics
from pvqa.metrics import ClassificationMetric


class BinaryClassificationMetrics(ClassificationMetric):
    def __init__(self, mode: Literal["train", "val", "test"]):
        super().__init__()
        self.mode = mode

        self.accuracy = tm.Accuracy(task="binary")
        self.precision = tm.Precision(task="binary")
        self.recall = tm.Recall(task="binary")
        self.f1_score = tm.F1Score(task="binary")
        self.auroc = tm.AUROC(task="binary")

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

    @property
    def scalars(self):
        return {"precision": self.precision, "recall": self.recall, "accuracy": self.accuracy,
                "f1_score": self.f1_score, "auroc": self.auroc}

    def nonscalars(self, current_epoch):
        return self.plot_confusion_matrix(current_epoch), \
            self.plot_roc(current_epoch), \
            self.plot_pr_curve(current_epoch)

    def plot_confusion_matrix(self, current_epoch):
        cf_matrix = self.cnfs_mat.compute().cpu().numpy()
        fig = metrics.ConfusionMatrixDisplay(
            cf_matrix).plot(values_format='.1%').figure_
        fig.set_size_inches(10, 10)
        fig.suptitle(f"Confusion Matrix, Epoch {current_epoch}")
        return {"name": "cnfs_mat", "type": "fig", "data": fig}

    def plot_roc(self, current_epoch):
        fpr, tpr, thresholds = [o.cpu().numpy() for o in self.roc.compute()]
        fig = metrics.RocCurveDisplay(
            fpr=fpr, tpr=tpr, roc_auc=self.auroc.compute().cpu()).plot().figure_
        fig.set_size_inches(10, 10)
        fig.suptitle(f"ROC Curve, Epoch {current_epoch}")
        return {"name": "roc", "type": "fig", "data": fig}

    def plot_pr_curve(self, current_epoch):
        precision, recall, thresholds = [
            o.cpu().numpy() for o in self.prc.compute()]
        fig = metrics.PrecisionRecallDisplay(
            precision=precision, recall=recall).plot().figure_
        fig.set_size_inches(10, 10)
        fig.suptitle(f"Precision Recall Curve, Epoch {current_epoch}")
        return {"name": "prc", "type": "fig", "data": fig}
