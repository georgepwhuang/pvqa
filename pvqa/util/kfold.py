from pytorch_lightning.loops.fit_loop import FitLoop
from sklearn.model_selection import KFold
from torch.utils.data import Subset

from typing import Any, Optional


class KFoldLoop(FitLoop):
    def __init__(self, folds, min_epochs: Optional[int] = 0, max_epochs: Optional[int] = None) -> None:
        super().__init__(min_epochs, max_epochs)
        self.folds = folds
        self.current_fold: int = 0

    def on_run_start(self, *args: Any, **kwargs: Any) -> None:
        self.trainer.datamodule.setup_folds(self.folds)
        super().on_run_start(*args, **kwargs)

    def on_advance_start(self, *args: Any, **kwargs: Any) -> None:
        self.trainer.datamodule.setup_fold_index(self.trainer.current_epoch)
        super().on_advance_start(*args, **kwargs)

def setup_folds(self, folds) -> None:
    self.folds = folds
    self.splits = [split for split in KFold(self.folds).split(range(len(self.full_dataset)))]

def setup_fold_index(self, fold_index: int) -> None:
    fold_index = fold_index % self.folds
    train_indices, val_indices = self.splits[fold_index]
    self.train_dataset = Subset(self.full_dataset, train_indices)
    self.val_dataset = Subset(self.full_dataset, val_indices)