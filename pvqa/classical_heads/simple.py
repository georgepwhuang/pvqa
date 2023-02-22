from typing import List, Union

import pytorch_lightning as pl
import torch
from sklearn import metrics
from torch import nn
from torch.nn import functional as F

from pvqa.metrics import MulticlassClassificationMetrics, BinaryClassificationMetrics


class SimpleClassifier(pl.LightningModule):
    def __init__(self, input_dim: int, labels: Union[List[str], int], learning_rate: float):
        super().__init__()
        self.input_dim = input_dim
        if isinstance(labels, int):
            self.num_classes = labels
            self.labels = list(map(lambda x: str(x), range(labels)))
        elif isinstance(labels, list):
            self.labels = labels
            self.num_classes = len(self.labels)
        self.learning_rate = learning_rate
        self.save_hyperparameters()

        self.num_classes = 1 if self.num_classes == 2 else self.num_classes
        self.task = "binary" if self.num_classes == 1 else "multiclass"

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.num_classes))

        if self.task == "multiclass":
            self.val_metrics = MulticlassClassificationMetrics(self.num_classes, "val", self.labels)
            self.test_metrics = MulticlassClassificationMetrics(self.num_classes, "test", self.labels)
        elif self.task == "binary":
            self.val_metrics = BinaryClassificationMetrics("val")
            self.test_metrics = BinaryClassificationMetrics("test")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data, label = batch
        output = self(data)
        if self.task == "binary":
            output = output.squeeze(1)
            loss = F.binary_cross_entropy_with_logits(output, label.type(torch.float))
        else:
            loss = F.cross_entropy(output, label)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        data, label = batch
        output = self(data)
        if self.task == "binary":
            output = output.squeeze(1)
            loss = F.binary_cross_entropy_with_logits(output, label.type(torch.float))
            logits = F.sigmoid(output)
        else:
            loss = F.cross_entropy(output, label)
            out = F.softmax(output, -1)
            logits = out.argmax(dim=1)
        self.log('val_loss', loss, on_epoch=True)
        self.val_metrics(logits, label)
        self.log_scalars(self.val_metrics)

    def validation_epoch_end(self, outputs):
        self.log_nonscalars(self.val_metrics)

    def test_step(self, batch, batch_idx):
        data, label = batch
        output = self(data)
        if self.task == "binary":
            output = output.squeeze(1)
            loss = F.binary_cross_entropy_with_logits(output, label.type(torch.float))
            logits = F.sigmoid(output)
        else:
            loss = F.cross_entropy(output, label)
            out = F.softmax(output, -1)
            logits = out.argmax(dim=1)
        self.log('test_loss', loss, on_epoch=True)
        self.test_metrics(logits, label)
        self.log_scalars(self.test_metrics)

    def test_epoch_end(self, outputs):
        self.log_nonscalars(self.test_metrics)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

    def log_scalars(self, metric):
        for k, v in metric.scalars.items():
            self.log(f"{metric.mode}_{k}", v, on_epoch=True, prog_bar=True)

    def log_nonscalars(self, metric):
        self.plot_confusion_matrix(metric)
        if self.task == "multiclass":
            self.write_classification_report(metric)
        elif self.task == "binary":
            self.plot_roc(metric)
            self.plot_pr_curve(metric)

    def plot_confusion_matrix(self, metric):
        cf_matrix = metric.cnfs_mat.compute().cpu().numpy()
        fig = metrics.ConfusionMatrixDisplay(cf_matrix, display_labels=self.labels).plot(values_format='.1%').figure_
        fig.set_size_inches(10, 10)
        fig.suptitle(f"Confusion Matrix, Epoch {self.current_epoch}")
        self.logger.experiment.add_figure(f"{metric.mode}_cnfs_mat", fig, global_step=self.current_epoch)

    def write_classification_report(self, metric):
        report = metric.class_report.compute()
        self.logger.experiment.add_text(f"{metric.mode}_report", report, global_step=self.current_epoch)
        metric.class_report.reset()

    def plot_roc(self, metric):
        fpr, tpr, thresholds = [o.cpu().numpy() for o in metric.roc.compute()]
        fig = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=metric.auroc.compute().cpu()).plot().figure_
        fig.set_size_inches(10, 10)
        fig.suptitle(f"ROC Curve, Epoch {self.current_epoch}")
        self.logger.experiment.add_figure(f"{metric.mode}_roc", fig, global_step=self.current_epoch)

    def plot_pr_curve(self, metric):
        precision, recall, thresholds = [o.cpu().numpy() for o in metric.prc.compute()]
        fig = metrics.PrecisionRecallDisplay(precision=precision, recall=recall).plot().figure_
        fig.set_size_inches(10, 10)
        fig.suptitle(f"Precision Recall Curve, Epoch {self.current_epoch}")
        self.logger.experiment.add_figure(f"{metric.mode}_prc", fig, global_step=self.current_epoch)
