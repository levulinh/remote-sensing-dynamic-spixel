import albumentations as alb
import os
import pytorch_lightning as pl
import random
import torch

from torch import nn
from torch.utils.data.dataloader import DataLoader
from torchmetrics import MetricCollection, Accuracy, Precision
from typing import List, Union

from loader import IDXDataset
from models.custom.model import CustomModel

CRITERION = {
    "CE": nn.CrossEntropyLoss(),
    "MSE": nn.MSELoss(),
    "NL": nn.NLLLoss(),
    "HE": nn.HingeEmbeddingLoss(),
    "KL": nn.KLDivLoss(),
    "BCE": nn.BCEWithLogitsLoss(),
}

DATASETS = {  # validate and test these sequentially
    "./datasets/AID": "*.jpg",
    "./datasets/NWPU": "*.jpg",
    "./datasets/PNET": "*.jpg",
    "./datasets/UCM": "*.tif",
}

random.seed(314)

class IDXDataModule(pl.LightningDataModule):
    def __init__(self, args: "Namespace") -> None:
        super().__init__()

        self.args = args

        self.train_datapath = os.path.join(args.dataset_dir, args.dataset, "train")
        img_format = DATASETS.get(os.path.join(args.dataset_dir, args.dataset))
        self.batch_size = args.batch_size

        base_transform = [alb.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
        train_transforms = list()
        if args.augmentation:
            train_transforms += [alb.HorizontalFlip(), alb.ColorJitter()]
        train_transforms += base_transform

        self.val_transforms = alb.Compose(base_transform)

        self.train_dataset = IDXDataset(
            sep=args.sep,
            transform=alb.Compose(train_transforms),
            image_format=img_format,
            datapath=self.train_datapath,
            labels_txt=args.label_txt
        )

    def train_dataloader(self) -> Union["DataLoader", List["DataLoader"]]:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.args.workers,
            shuffle=True,
            pin_memory=False,
        )

    def val_dataloader(self) -> List["DataLoader"]:
        combined_loaders = list()
        for datapath, img_format in DATASETS.items():
            dataset_type = datapath.split("/")[2]
            if dataset_type in self.args.val_sets:
                val_dataset = IDXDataset(
                    sep=self.args.sep,
                    transform=self.val_transforms,
                    image_format=img_format,
                    datapath=os.path.join(datapath, "val"),
                    labels_txt=self.args.label_txt
                )
                combined_loaders.append(
                    DataLoader(
                        val_dataset,
                        batch_size=self.batch_size,
                        num_workers=self.args.workers,
                        shuffle=False,
                        pin_memory=True
                    )
                )
        return combined_loaders

    def test_dataloader(self) -> List["DataLoader"]:
        combined_loaders = list()
        for datapath, img_format in DATASETS.items():
            dataset_type = datapath.split("/")[2]
            if dataset_type in self.args.val_sets:
                val_dataset = IDXDataset(
                    sep=self.args.sep,
                    transform=self.val_transforms,
                    image_format=img_format,
                    datapath=os.path.join(datapath, "val"),
                    labels_txt=self.args.label_txt
                )
                combined_loaders.append(
                    DataLoader(
                        val_dataset,
                        batch_size=self.batch_size,
                        num_workers=self.args.workers,
                        shuffle=False,
                        pin_memory=True
                    )
                )
        return combined_loaders


class IDXModel(pl.LightningModule):
    def __init__(self, args: "Namespace") -> None:
        super().__init__()

        self.args = args
        self.num_classes = args.num_classes
        self.lr = args.learning_rate

        self.model = CustomModel(args)
        self.criterion = CRITERION.get(args.loss_function)

        # metric trackers
        metrics = {
            "top1": Accuracy(
                compute_on_step=False,
                dist_sync_on_step=True,
                num_classes=2, # one-hot
                multiclass=True,
            ),
            "precision": Precision(
                compute_on_step=False,
                dist_sync_on_step=True,
                num_classes=2, # one-hot
                multiclass=True,
                mdmc_average='global'
            )
        }

        self.train_metrics = MetricCollection(metrics=metrics, prefix="train_")
        self.valid_metrics = MetricCollection(metrics=metrics, prefix="valid_")
        self.test_metrics = MetricCollection(metrics=metrics, prefix="test_")

        self.train_matrix = None
        self.best_accuracy = -1

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def training_step(self, batch, batch_id) -> float:
        inputs, labels = batch
        logits = self(inputs)
        losses = self.criterion(logits, labels.float())
        self.log(
            "train_loss",
            losses,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
            logger=True,
        )
        self.train_metrics(logits.softmax(-1), labels)
        return losses

    def training_epoch_end(self, outputs) -> None:
        scores = self.train_metrics.compute()
        self.train_metrics.reset()
        self.log_dict(scores)

    def validation_step(self, batch, batch_id) -> None:
        inputs, labels = batch
        logits = self(inputs)
        losses = self.criterion(logits, labels.float())
        self.log(
            "valid_loss",
            losses,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            logger=True,
            add_dataloader_idx=True,
        )
        self.valid_metrics(logits.softmax(-1), labels)

    def validation_epoch_end(self, outputs) -> None:
        scores = self.valid_metrics.compute()
        self.valid_metrics.reset()
        if self.best_accuracy < scores["valid_top1"]:
            self.best_accuracy = scores["valid_top1"]
        self.log_dict(scores)

    def test_step(self, batch, batch_idx) -> None:
        inputs, labels = batch
        logits = self(inputs)
        self.test_metrics(logits.softmax(-1), labels)

    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx) -> None:
        scores = self.test_metrics.compute()
        self.test_metrics.reset()
        self.log_dict(scores)

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        grouped_parms = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if (not any(nd in n for nd in no_decay) and p.requires_grad)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if (any(nd in n for nd in no_decay) and p.requires_grad)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = getattr(torch.optim, self.args.optimizer)
        if self.args.optimizer == "AdamW" or self.args.optimizer == "Adam":
            optimizer = optimizer(
                grouped_parms,
                lr=self.lr,
                betas=(self.args.beta_1, self.args.beta_2),
                eps=self.args.eps_adam,
                weight_decay=self.args.weight_decay,
            )
        elif self.args.optimizer == "RMSProp":
            optimizer = optimizer(
                grouped_parms,
                lr=self.lr,
                alpha=self.args.alpha,
                eps=self.args.eps_rms,
                weight_decay=self.args.weight_decay,
            )
        else:  # SGD
            optimizer = optimizer(grouped_parms, lr=self.lr)

        lambda1 = lambda epoch: epoch // 30
        lambda2 = lambda epoch: 0.95 ** epoch
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=[lambda1, lambda2], last_epoch=-1, verbose=False
        )

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}