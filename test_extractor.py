import pdb
import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml

from arguments import Args
from classifier import IDXModel, IDXDataModule
from gnn_model_util import FeatureMapExtractor


if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    config_input = yaml.load(open("config.yml").read(), Loader=yaml.Loader)
    args = Args(config_input)

    pl_model = IDXModel(args)
    print('Loading pretrained model from file...')
    model_state_dict = torch.load(args.model_dir)["state_dict"]
    pl_model.load_state_dict(model_state_dict, strict=False)
    print(f"Resume from checkpoint: {args.model_dir}!")

    extractor = FeatureMapExtractor(pl_model)

    extractor.eval()
    extractor.to(device)

    pl_data = IDXDataModule(args)

    train_dataset = pl_data.train_dataset
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)

    data = next(iter(train_loader))
    x = data[0]
    x = x.to(device)

    out = extractor(x)

    upsampler = nn.Upsample(size=(256, 256))
    feature_map_block4, feature_map_block5 = upsampler(out[0]), upsampler(out[1])

    pdb.set_trace()
