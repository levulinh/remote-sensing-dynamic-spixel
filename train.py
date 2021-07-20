import pytorch_lightning as pl
import torch
import yaml

from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from arguments import Args
from classifier import IDXDataModule, IDXModel

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class bcolors:
    GREEN = "\033[92m"
    BOLD = "\033[1m"
    ENDC = "\033[0m"


if __name__ == "__main__":
    config_input = yaml.load(open("config.yml").read(), Loader=yaml.Loader)
    args = Args(config_input)

    pl.seed_everything(args.seed, workers=True)

    logger_name = f"{args.model_name}-V{args.version}"

    if args.wandb:
        logger = WandbLogger(project=args.project_name, name=logger_name)
    else:
        logger = TensorBoardLogger(save_dir=f"logs/{logger_name}.log")
    logger.log_hyperparams(vars(args))

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_top1",
        dirpath=args.checkpoint,
        every_n_train_steps=0,
        every_n_val_epochs=1,
        filename=str(args.dataset)+"-{epoch}-{valid_precision:.2f}",
        save_top_k=1,
        mode="max",
    )

    print(
        f"\n{bcolors.BOLD}{bcolors.GREEN}********** TRAINING CONFIGURATION **********{bcolors.ENDC}\n{args}\n"
    )

    trainer_args = {
        "gpus": args.gpus,
        "callbacks": [checkpoint_callback],
        "logger": logger,
        "log_every_n_steps": args.log_interval,
        "max_epochs": args.max_epochs,
        "val_check_interval": args.val_check_interval,
        "deterministic": True,
    }

    if args.distributed:
        trainer_args["accelerator"] = "ddp"
        trainer_args["plugins"] = DDPPlugin(find_unused_parameters=False)

    trainer = pl.Trainer(**trainer_args)

    pl_model = IDXModel(args)

    if args.resume:
        model_state_dict = torch.load(args.model_dir)["state_dict"]
        pl_model.load_state_dict(model_state_dict, strict=False)
        print(f"Resume from checkpoint: {args.model_dir}!")
    
    pl_data = IDXDataModule(args) 

    trainer.fit(pl_model, pl_data)

    if args.run_test:
        print(
            f"{bcolors.BOLD}FINISHED TRAINING: RUN TESTS ON VAL DATASET:{bcolors.ENDC}"
        )
        trainer.test(pl_model)