import albumentations as alb
from loader import SpixelDataset
import os
import torch 
import yaml
import pdb
import torch.nn.functional as F
from torch_geometric.nn import DynamicEdgeConv, global_max_pool
from torch.nn import Sequential as Seq, Dropout, Linear as Lin, ReLU, BatchNorm1d as BN 

from arguments import Args
from classifier import IDXModel

DATASETS = {  # validate and test these sequentially
    "./datasets/AID": "*.jpg",
    "./datasets/NWPU": "*.jpg",
    "./datasets/PNET": "*.jpg",
    "./datasets/UCM": "*.tif",
}

def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])

class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels, k=10, aggr='max'):
        super().__init__()

        self.conv1 = DynamicEdgeConv(MLP([2 * in_channels, 64]), k, aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 64, 64, 128]), k, aggr)
        self.lin1 = MLP([128 + 64, 1024])

        self.mlp = Seq(
            MLP([1024, 512]), Dropout(0.5), MLP([512, 256]), Dropout(0.5),
            Lin(256, out_channels))

    def forward(self, data):
        x, pos, batch = data.x, data.pos, data.batch
        x = torch.cat((x, pos/255.0), dim=1)
        x1 = self.conv1(x, batch)
        x2 = self.conv2(x1, batch)
        out = self.lin1(torch.cat([x1, x2], dim=1))
        out = global_max_pool(out, batch)
        out = self.mlp(out)
        return F.log_softmax(out, dim=1)

if __name__ == '__main__':
    config_input = yaml.load(open("config.yml").read(), Loader=yaml.Loader)
    args = Args(config_input)
    img_format = DATASETS.get(os.path.join(args.dataset_dir, args.dataset))
    base_transform = [alb.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
    train_transforms = list()
    if args.augmentation:
        train_transforms += [alb.HorizontalFlip(), alb.ColorJitter()]
    train_transforms += base_transform
    train_datapath = os.path.join(args.dataset_dir, args.dataset, "train")

    cnn_model = IDXModel(args)
    print('Loading pretrained model from file...')
    model_state_dict = torch.load(args.model_dir)["state_dict"]
    cnn_model.load_state_dict(model_state_dict, strict=False)
    print(f"Resume from checkpoint: {args.model_dir}!")

    spixeldataset = SpixelDataset(
        sep=args.sep,
        transform=alb.Compose(train_transforms),
        image_format=img_format,
        datapath=train_datapath,
        labels_txt=args.label_txt,
        pretrain_cnn_model=cnn_model,
        use_transform = True
    )

    pdb.set_trace()