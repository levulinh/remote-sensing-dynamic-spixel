# CNN Training

## Environment Setup
1. To train the network, edit ```config.yml``` and run ```python train.py```.

## File Structure
```bash
linh
├── arguments.py
├── classifier.py
├── config.yml
├── datasets/
│   ├── AID/
│   │   ├── train/
│   │   └── val/
│   ├── NWPU/
│   │   ├── train/
│   │   └── val/
│   ├── PNET/
│   │   ├── train/
│   │   └── val/
│   └── UCM/
│       ├── train/
│       └── val/
├── loader.py
├── models/
│   ├── custom/
│   │   ├── __init__.py
│   │   └── model.py
│   └── __init__.py
├── README.md
├── multilabels.txt
└── train.py
```

## Configurations
- Edit [config.yml](config.yml) to adjust different hyperparameters and switch different pretrained model ```resnet18, vgg16, vgg16_bn```. 