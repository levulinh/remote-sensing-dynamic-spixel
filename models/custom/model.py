import timm
import torch
import torch.nn as nn

from torch_geometric.nn import DynamicEdgeConv, global_max_pool
from torch.nn import Sequential as Seq, Dropout, Linear as Lin, ReLU, BatchNorm1d as BN 

class CustomModel(nn.Module):
    """
    Use timm package to create any pretrained CNN model: resnet18, vgg16. Please refer to timm package
    """
    def __init__(self, args: "Namespace") -> None:
        super().__init__()
        self.model = timm.create_model(
            args.model_name,
            pretrained=args.pretrained,
            num_classes=args.num_classes
        )

    def forward(self, x):
        x = self.model(x)
        return x

def MLP(channels, batch_norm=False):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])

class DynamicModel(torch.nn.Module):
    def __init__(self, args: "Namespace"):
        super().__init__()

        # self.conv1 = DynamicEdgeConv(MLP([2 * args.in_channels, 1024]), args.topk, args.aggr)
        # self.pos_enc = MLP(2, 512, 1024)
        # self.conv2 = DynamicEdgeConv(MLP([2 * 1024, 1024, 1024, 2048]), args.topk, args.aggr)
        # self.lin1 = MLP([1024 + 2048, 4096])

        # self.mlp = Seq(
        #     MLP([4096, 2048]), Dropout(0.5), MLP([2048, 1024]), Dropout(0.5),
        #     Lin(1024, args.num_classes))
        in_channels = 1026 if args.cat else 1024
        self.conv1 = DynamicEdgeConv(MLP([2 * in_channels, 512]), args.topk, args.aggr)
        # if not args.cat:
        #     self.pos_enc = MLP([2, 128, 512])
        if args.with_pos:
            self.pos_enc = MLP([2, 128, 512])
        self.conv2 = DynamicEdgeConv(MLP([2 * 512, 512, 1024, 1024]), args.topk, args.aggr)
        self.lin1 = MLP([1024 + 512, 2048])
        self.cat = args.cat
        self.with_pos = args.with_pos

        self.mlp = Seq(
            MLP([2048, 1024]), Dropout(0.5), MLP([1024, 512]), Dropout(0.5),
            Lin(512, args.num_classes))

    def forward(self, data):
        x, pos, batch = data.x_cnn, data.pos, data.batch

        # if self.cat: # Concat center of segments to the segment feature
        #     x = torch.cat((x, pos), dim=1)
        #     x1 = self.conv1(x, batch)
        # else:        # Encode the center of segments and add to first conv result
        #     x1 = self.conv1(x, batch)
        #     pos_enc = self.pos_enc(pos)
        #     x1 = x1 + pos_enc
        x1 = self.conv1(x, batch)
        if self.with_pos:
            pos_enc = self.pos_enc(pos)
            x1 = x1 + pos_enc
        x2 = self.conv2(x1, batch)
        out = self.lin1(torch.cat([x1, x2], dim=1))
        out = global_max_pool(out, batch)
        out = self.mlp(out)
        return out