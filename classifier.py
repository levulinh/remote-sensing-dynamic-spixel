import albumentations as alb
import os
import pytorch_lightning as pl
import torch
from torch_geometric.data import Batch
from models.custom.model import DynamicModel
from utils.loss_functions import Criterion
from torchmetrics import MetricCollection, Recall, Precision, F1, FBeta
from sklearn.metrics import precision_score
from gnn_model_util import FeatureMapExtractor
from models.custom.model import CustomModel
from torch.nn import Upsample
from torch_scatter import scatter_max, scatter_mean
from torch_geometric.data import DataLoader
from loader import SpixelDataset

DATASETS = {  # validate and test these sequentially
    "./datasets/AID": "*.jpg",
    "./datasets/NWPU": "*.jpg",
    "./datasets/PNET": "*.jpg",
    "./datasets/UCM": "*.tif",
}

class SpixelModel(pl.LightningModule):
    def __init__(self, args: "Namespace") -> None:
        super().__init__()

        self.args = args

        self.model = DynamicModel(args)
        self.criterion = getattr(Criterion, args.loss_function)

        pretrained_cnn = CustomModel(args)
        print('Loading pretrained cnn model...')
        pretrained_cnn_state_dict = torch.load(args.model_dir)["state_dict"]
        pretrained_cnn.load_state_dict(pretrained_cnn_state_dict, strict=False)
        print('Done!')

        print("here")
        self.extractor = FeatureMapExtractor(pretrained_cnn)
        for n, p in self.extractor.named_parameters():
            p.requires_grad = False

        # metric trackers
        metrics = {
            "precision": Precision(
                compute_on_step=False,
                dist_sync_on_step=True,
                multiclass=True,
                num_classes=2,
                mdmc_average='samplewise',
                average='macro'
            ),
            "recall": Recall(
                compute_on_step=False,
                dist_sync_on_step=True,
                multiclass=True,
                num_classes=2,
                mdmc_average='samplewise',
                average='macro'
            ),
            "F1": F1(
                compute_on_step=False,
                dist_sync_on_step=True,
                multiclass=True,
                num_classes=2,
                mdmc_average='samplewise',
                average='macro'
            ),
            "F2": FBeta(
                beta=2.0,
                compute_on_step=False,
                dist_sync_on_step=True,
                num_classes=2,
                multiclass=True,
                mdmc_average='samplewise',
                average='macro'
            )

        }

        self.train_metrics = MetricCollection(metrics = metrics, prefix='train_')
        self.val_metrics = MetricCollection(metrics = metrics, prefix='val_')
        self.test_metrics = MetricCollection(metrics = metrics, prefix='test_')


    def forward(self, x: Batch):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.args.optimizer)
        optimizer = optimizer(self.parameters(), lr=self.args.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def training_step(self, batch, batch_idx):
        # we load images here, pass inside the get_graph func
        image = batch[0][0]
        graph = batch[0][1]
        labels = batch[1]
        graph_inputs = self.get_graph_from_image(image, graph)
        logits = self(graph_inputs)
        losses = self.criterion(logits.sigmoid(), labels.float())
        self.log(
            "train_loss",
            losses,
            logger=True,
            on_step=True,
            on_epoch=False,
            sync_dist=True
        )
        self.train_metrics(logits.sigmoid(), labels)
        return losses
    
    def training_epoch_end(self, outputs) -> None:
        scores = self.train_metrics.compute()
        self.train_metrics.reset()
        self.log_dict(scores)

    def validation_step(self, batch, batch_idx):
        image = batch[0][0]
        graph = batch[0][1]
        labels = batch[1]
        graph_inputs = self.get_graph_from_image(image, graph)
        logits = self(graph_inputs)
        losses = self.criterion(logits.sigmoid(), labels.float())
        self.log(
            "val_loss",
            losses,
            logger=True,
            on_step=True,
            on_epoch=False,
            sync_dist=True
        )

        self.val_metrics(logits.sigmoid(), labels)

    def validation_epoch_end(self, outputs) -> None:
        scores = self.val_metrics.compute()
        self.val_metrics.reset()
        self.log_dict(scores)

    def test_step(self, batch, batch_idx):
        image = batch[0][0]
        graph = batch[0][1]
        labels = batch[1]
        graph_inputs = self.get_graph_from_image(image, graph)
        logits = self(graph_inputs)

        self.test_metrics(logits.sigmoid(), labels)

    def on_test_batch_end(self, outputs, batch, batch_idx, _) -> None:
        scores = self.test_metrics.compute()
        self.test_metrics.reset()
        self.log_dict(scores)

    @staticmethod
    def _scatter_max(features, index):
        features = features.reshape(512, 256*256)
        index = index.reshape(256*256)
        out_max, _ = scatter_max(features, index)
        return torch.transpose(out_max, 0, 1)

    @staticmethod
    def _scatter_mean(features, index):
        features = features.reshape(512, 256*256)
        index = index.reshape(256*256)
        out_mean = scatter_max(features, index)
        return torch.transpose(out_mean, 0, 1)

    # @torch.no_grad
    def get_graph_from_image(self, image, graph):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        upsampler = Upsample(size=(256, 256))
        
        index = graph.seg
        index = index.to(device)
        self.extractor.to(device)
        self.extractor.eval()
        image = image.to(device)

        block4, block5 = self.extractor(image)
        block4, block5 = upsampler(block4), upsampler(block5)

        cnn_features = []

        for i in range(block4.shape[0]):
            block4_layer = block4[i]
            block5_layer = block5[i]
            index_layer = index[i]
            if self.args.aggr == 'max':
                feature4_scatter = self._scatter_max(block4_layer, index_layer)
                feature5_scatter = self._scatter_max(block5_layer, index_layer)
            else:
                feature4_scatter = self._scatter_mean(block4_layer, index_layer)
                feature5_scatter = self._scatter_mean(block5_layer, index_layer)
            feature_scatter_layer = torch.cat((feature4_scatter, feature5_scatter), dim=1)
            cnn_features.append(feature_scatter_layer)

        graph_features = torch.cat(cnn_features, dim=0)
        graph.x_cnn = graph_features
        return graph

class SpixelDataModule(pl.LightningDataModule):
    def __init__(self, args: "Namespace"):
        super().__init__()

        self.args = args

        train_datapath = os.path.join(args.dataset_dir, args.dataset, "train")
        val_datapath = os.path.join(args.dataset_dir, args.dataset, "val")
        img_format = DATASETS.get(os.path.join(args.dataset_dir, args.dataset))
        self.batch_size = args.batch_size

        base_transform = [alb.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
        train_transforms = list()
        if args.augmentation:
            train_transforms += [alb.HorizontalFlip(), alb.ColorJitter()]
        train_transforms += base_transform

        self.train_dataset = SpixelDataset(
            sep=args.sep,
            transform=alb.Compose(train_transforms),
            image_format=img_format,
            datapath=train_datapath,
            labels_txt=args.label_txt,
            n_segments=args.num_seg,
        )

        self.val_dataset = SpixelDataset(
            sep=args.sep,
            transform=alb.Compose(base_transform),
            image_format=img_format,
            datapath=val_datapath,
            labels_txt=args.label_txt,
            n_segments=args.num_seg,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.args.workers,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.args.workers,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.args.workers,
            shuffle=False
        )
