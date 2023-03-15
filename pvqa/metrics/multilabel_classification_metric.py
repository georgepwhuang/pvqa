from typing import List, Literal

import torch
import torchmetrics as tm
from matplotlib import pyplot as plt
from sklearn import metrics
from pvqa.metrics.base import ClassificationMetric
from pvqa.metrics.classification_report import ClassificationReport


class MultilabelClassificationMetrics(ClassificationMetric):
    def __init__(self, num_classes: int,
                 mode: Literal["train", "val", "test"],
                 labels: List[str]):
        super().__init__()
        self.num_classes = num_classes
        self.mode = mode
        self.labels = labels

        self.accuracy = tm.Accuracy(num_labels=self.num_classes, task="multilabel")
        self.precision = tm.Precision(num_labels=self.num_classes, task="multilabel")
        self.recall = tm.Recall(num_labels=self.num_classes, task="multilabel")
        self.f1_score = tm.F1Score(num_labels=self.num_classes, task="multilabel")
        self.auroc = tm.AUROC(num_labels=self.num_classes, task="multilabel")
        self.auroc_cat = tm.AUROC(num_labels=self.num_classes, task="multilabel", average="none")

        self.cnfs_mat = tm.ConfusionMatrix(num_labels=self.num_classes, normalize="true", task="multilabel")
        self.roc = tm.ROC(num_labels=self.num_classes, task="multilabel")
        self.prc = tm.PrecisionRecallCurve(num_labels=self.num_classes, task="multilabel")
        self.class_report = ClassificationReport(num_classes=self.num_classes, labels=self.labels)

    def to(self, device: torch.device):
        for metric in self.scalars.values():
            metric.to(device)
        self.cnfs_mat.to(device)
        self.class_report.to(device)
        self.auroc_cat.to(device)

    def forward(self, x, y):
        for metric in self.scalars.values():
            metric(x, y)
        stat_scores = tm.functional.classification.stat_scores(x, y, num_labels=self.num_classes, average="none",
                                                               task="multilabel")
        self.cnfs_mat(x, y)
        self.roc(x, y)
        self.prc(x, y)
        self.auroc_cat(x, y)
        self.class_report.update(stat_scores)

    @property
    def scalars(self):
        return {"precision": self.precision, "recall": self.recall, "accuracy": self.accuracy,
                "f1_score": self.f1_score, "auroc": self.auroc}

    def nonscalars(self, current_epoch):
        return self.plot_confusion_matrix(current_epoch), \
            self.write_classification_report(), \
            self.plot_roc(current_epoch), \
            self.plot_pr_curve(current_epoch)

    def plot_confusion_matrix(self, current_epoch):
        cf_matrix = self.cnfs_mat.compute().cpu().numpy()
        rows = self.num_classes//2 if  self.num_classes% 2 == 0 else self.num_classes//2 + 1
        fig, axs = plt.subplots(2, rows)
        for i in range(rows):
            for j in range(2):
                idx = i*2+j
                metrics.ConfusionMatrixDisplay(cf_matrix[idx]).plot(ax=axs[j, i], values_format='.1%')
                axs[j, i].set_title(f"Label {self.labels[idx]}")
        fig.suptitle(f"Confusion Matrixes, Epoch {current_epoch}")
        return {"name": "cnfs_mat", "type": "fig", "data": fig}

    def write_classification_report(self):
        report = self.class_report.compute()
        self.class_report.reset()
        return {"name": "class_rprt", "type": "text", "data": report}

    def plot_roc(self, current_epoch):
        fpr, tpr, _ = self.roc.compute()
        auroc = self.auroc_cat.compute().cpu()
        rows = self.num_classes//2 if  self.num_classes% 2 == 0 else self.num_classes//2 + 1
        fig, axs = plt.subplots(2, rows)
        for i in range(rows):
            for j in range(2):
                idx = 2*i+j
                metrics.RocCurveDisplay(fpr=fpr[idx].cpu().numpy(), tpr=tpr[idx].cpu().numpy(), roc_auc=auroc[idx]).plot(ax=axs[j, i])
                axs[j, i].set_title(f"Label {self.labels[idx]}")
        fig.suptitle(f"ROC Curve, Epoch {current_epoch}")
        return {"name": "roc", "type": "fig", "data": fig}

    def plot_pr_curve(self, current_epoch):
        precision, recall, _ = self.prc.compute()
        rows = self.num_classes//2 if  self.num_classes% 2 == 0 else self.num_classes//2 + 1
        fig, axs = plt.subplots(2, rows)
        for i in range(rows):
            for j in range(2):
                idx = 2 * i + j
                metrics.PrecisionRecallDisplay(precision=precision[idx].cpu().numpy(), recall=recall[idx].cpu().numpy()).plot(ax=axs[j, i])
                axs[j, i].set_title(f"Label {self.labels[idx]}")
        fig.suptitle(f"Precision Recall Curve, Epoch {current_epoch}")
        return {"name": "prc", "type": "fig", "data": fig}
