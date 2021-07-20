from dataclasses import dataclass, field, InitVar
from typing import Dict, Tuple, Union


@dataclass
class Args:
    gpus: Union[str, int] = field(default=0)
    distributed: bool = field(default=False)
    workers: int = field(default=8)

    wandb: bool = field(default=True)
    project_name: str = field(default="eureka")
    log_interval: int = field(default=1)
    val_check_interval: float = field(default=0.5)
    checkpoint: str = field(default="./checkpoints")
    version: str = field(default="0.0")

    dataset_dir: str = field(default="./datasets")
    dataset: str = field(default="AID")
    seed: int = field(default=314)
    use_transform: bool = field(default=False)

    val_sets: Tuple[str] = field(default=("AID", "NWPU", "PNET", "UCM"))
    label_txt: str = field(default="./multilabels.txt")

    learning_rate: float = field(default=1e-3)
    batch_size: int = field(default=16)
    max_epochs: int = field(default=5)
    sep: str = field(default=',')

    num_classes: int = field(default=12)
    model_name: str = field(default="resnet18")
    pretrained: bool = field(default=True)
    model_dir: str = field(default="./saved_models")
    resume: bool = field(default=False)

    optimizer: str = field(default="SGD")  # adam, adamw, rmsprop, sgd
    weight_decay: float = field(default=1e-3)
    eps_adam: float = field(default=1e-8)
    beta_1: float = field(default=0.99)
    beta_2: float = field(default=0.999)
    eps_rms: float = field(default=1e-8)
    alpha: float = field(default=0.99)

    loss_function: str = field(default="CE")
    augmentation: bool = field(default=True)

    accumulate_grad_batches: int = field(default=1)

    run_test: bool = field(default=True)

    cat: bool = field(default=False)
    topk: int = field(default=20)
    aggr: str = field(default='max')

    configuration: InitVar[Dict] = field(default=dict())

    def __init__(self, configuration: Dict):
        for key, value in configuration.items():
            try:
                if value is not None:
                    self.__setattr__(key, value)
            except AttributeError:
                raise Exception(f"No configuration parameter {key} is found!")
