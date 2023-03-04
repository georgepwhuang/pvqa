from abc import ABC
from typing import List, Union

import pytorch_lightning as pl
import torch
from torch.nn import functional as F

from pvqa.metrics import *


class BaseClassifier(pl.LightningModule, ABC):
    def __init__(self, input_dim: int, labels: Union[List[Union[str, int]], int], multilabel: bool = False):
        super().__init__()
        self.input_dim = input_dim
        if isinstance(labels, int):
            self.num_classes = labels
            self.labels = list(map(lambda x: str(x), range(labels)))
        elif isinstance(labels, list):
            self.labels = list(map(lambda x: str(x), labels))
            self.num_classes = len(self.labels)
        self.save_hyperparameters()

        if multilabel:
            self.task = "multilabel"
        else:
            self.num_classes = 1 if self.num_classes == 2 else self.num_classes
            self.task = "binary" if self.num_classes == 1 else "multiclass"

        if self.task == "multiclass":
            self.val_metrics = MulticlassClassificationMetrics(self.num_classes, "val", self.labels)
            self.test_metrics = MulticlassClassificationMetrics(self.num_classes, "test", self.labels)
        elif self.task == "binary":
            self.val_metrics = BinaryClassificationMetrics("val")
            self.test_metrics = BinaryClassificationMetrics("test")
        elif self.task == "multilabel":
            self.val_metrics = MultilabelClassificationMetrics(self.num_classes, "val", self.labels)
            self.test_metrics = MultilabelClassificationMetrics(self.num_classes, "test", self.labels)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data, label = batch
        output = self(data)
        if self.task == "binary":
            output = output.squeeze(1)
            loss = F.binary_cross_entropy_with_logits(output, label.type(torch.float))
        elif self.task == "multiclass":
            loss = F.cross_entropy(output, label)
        elif self.task == "multilabel":
            loss = F.binary_cross_entropy_with_logits(output, label.type(torch.float))
        else:
            raise RuntimeError(f"No task {self.task} is defined")
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        data, label = batch
        output = self(data)
        if self.task == "binary":
            output = output.squeeze(1)
            loss = F.binary_cross_entropy_with_logits(output, label.type(torch.float))
            logits = torch.sigmoid(output)
        elif self.task == "multiclass":
            loss = F.cross_entropy(output, label)
            out = F.softmax(output, -1)
            logits = out.argmax(dim=1)
        elif self.task == "multilabel":
            loss = F.binary_cross_entropy_with_logits(output, label.type(torch.float))
            logits = torch.sigmoid(output)
        else:
            raise RuntimeError(f"No task {self.task} is defined")
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
            logits = torch.sigmoid(output)
        elif self.task == "multiclass":
            loss = F.cross_entropy(output, label)
            out = F.softmax(output, -1)
            logits = out.argmax(dim=1)
        elif self.task == "multilabel":
            loss = F.binary_cross_entropy_with_logits(output, label.type(torch.float))
            logits = torch.sigmoid(output)
        else:
            raise RuntimeError(f"No task {self.task} is defined")
        self.log('test_loss', loss, on_epoch=True)
        self.test_metrics(logits, label)
        self.log_scalars(self.test_metrics)

    def test_epoch_end(self, outputs):
        self.log_nonscalars(self.test_metrics)

    def log_scalars(self, metric):
        for k, v in metric.scalars.items():
            self.log(f"{metric.mode}_{k}", v, on_epoch=True, prog_bar=True)

    def log_nonscalars(self, metric):
        for m_out in metric.nonscalars(self.current_epoch):
            if m_out["type"] == "fig":
                self.logger.experiment.add_figure(f"{metric.mode}_{m_out['name']}",
                                                  m_out['data'], global_step=self.current_epoch)
            elif m_out["type"] == "text":
                self.logger.experiment.add_text(f"{metric.mode}_{m_out['name']}",
                                                m_out['data'], global_step=self.current_epoch)
