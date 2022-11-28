from args import *
from UNet_monai import Unet
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from dataset import SpaceNet7DataModule
from downloads import download_images, download_checkpoints
import mlflow.pytorch
import os


if __name__ == "__main__":
    os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:8000/'
    args = get_main_args()
    callbacks = []
    download_images()
    download_checkpoints()
    model = Unet(args)
    model_ckpt = ModelCheckpoint(dirpath="./", filename="last",
                                monitor="dice_mean", mode="max", save_last=True)
    callbacks.append(model_ckpt)
    dm = SpaceNet7DataModule(args)
    trainer = Trainer(callbacks=callbacks, enable_checkpointing=True, max_epochs=args.num_epochs, 
                    enable_progress_bar=True, gpus=1, accelerator="cpu", amp_backend='apex', profiler='simple')

    # train the model
    print("---------------------------------------\n")
    print("The execution mode is :", args.exec_mode)
    print("\n---------------------------------------")

    mlflow.pytorch.autolog()

    if args.exec_mode == 'train':
        trainer.fit(model, dm)
    else:
        trainer.predict(model, datamodule=dm, ckpt_path=args.ckpt_path) 
