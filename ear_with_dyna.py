import os
import glob
import click
import logging

import comet_ml

from custom_dataset_for_ear import get_dataset_by_name, TokenizerDataModule
import IPython
import pdb

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
# import pytorch_lightning.metrics.functional as plf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import torchmetrics
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
)
import pandas as pd

from ear_with_gab import *

# from aim.pytorch_lightning import AimLogger

logging.basicConfig(
    format="%(levelname)s:%(asctime)s:%(module)s:%(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# this hides a warning thrown by huggingface transformers
# https://github.com/huggingface/transformers/issues/5486
# https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning
os.environ["TOKENIZERS_PARALLELISM"] = "true"  #  set to false is processes stuck

@click.command()
@click.option("--src_model_path", type=str, required=True)
@click.option("--output_dir", type=str, default="./dumps")
@click.option("--training_dataset", type=str, default="wiki")
@click.option("--batch_size", type=int, default=32)
@click.option("--num_workers", type=int, default=0)
@click.option("--seed", type=int, default=42)
@click.option("--max_epochs", type=int, default=20)
@click.option("--gpus", type=int, default=0)
@click.option("--accelerator", type=str, default=None)
@click.option("--max_seq_length", type=int, default=None)
@click.option("--learning_rate", type=float, default=2e-5)
@click.option("--early_stop_epochs", type=int, default=5)
@click.option("--regularization", type=str, default=None)
@click.option("--reg_strength", type=float, default=0.01)
@click.option("--weight_decay", type=float, default=0.0)
@click.option("--warmup_train_perc", type=float, default=None, help="Value [0,1]")
@click.option("--accumulate_grad_batches", type=int, default=1)
@click.option("--precision", type=int, default=32)
@click.option("--run_test", is_flag=True)
@click.option("--pin_memory", is_flag=True)
@click.option("--log_every_n_steps", type=int, default=50)
@click.option("--monitor", type=str, default="val_loss")
@click.option("--checkpoint_every_n_epochs", type=int, default=None)
@click.option("--save_transformers_model", is_flag=True)
@click.option("--ckpt_save_top_k", type=int, default=1)
@click.option("--resume_from_checkpoint", type=str, default=None)
@click.option("--balanced_loss", is_flag=True)
def main(
    src_model_path,
    output_dir,
    training_dataset,
    batch_size,
    num_workers,
    seed,
    max_epochs,
    gpus,
    accelerator,
    max_seq_length,
    learning_rate,
    early_stop_epochs,
    regularization,
    reg_strength,
    weight_decay,
    warmup_train_perc,
    accumulate_grad_batches,
    precision,
    run_test,
    pin_memory,
    log_every_n_steps,
    monitor,
    checkpoint_every_n_epochs,
    save_transformers_model,
    ckpt_save_top_k,
    resume_from_checkpoint,
    balanced_loss
):
    hparams = locals()
    pl.seed_everything(seed)


    model_name = f"entropybert-{training_dataset}-{seed}-{reg_strength}"
    experiment_name = f"entropybert-{training_dataset}"
    src_model = [x for x in os.listdir(src_model_path) if x.startswith('PL-epoch')][0]



    os.makedirs(output_dir, exist_ok=True)

    model_dir = os.path.join(output_dir, model_name)

 
    tokenizer = AutoTokenizer.from_pretrained(src_model_path)
    # for round in range(1, 5):
 
    dataset_module = TokenizerDataModule(
        dataset_name='dyna',
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
        num_workers=num_workers,
        pin_memory=pin_memory,
        load_pre_tokenized=False,
        store_pre_tokenized=False,
        round=1
    )

    # check if linear lr warmup is required
    train_steps_count = None
    if warmup_train_perc:
        logger.info(f"Warmup linear LR requested with {warmup_train_perc}")
        train_steps_count = (
            int(dataset_module.train_steps / accumulate_grad_batches) * max_epochs
        )
        logger.info(f"Total training steps: {train_steps_count}")
        if gpus and gpus > 0:
            train_steps_count = train_steps_count // gpus
        logger.info(f"Total training steps (gpu-normalized): {train_steps_count}")

    if balanced_loss:
        train, val, test = get_dataset_by_name('dyna', round=1)
        labels_count = pd.Series(train.labels).value_counts()
        labels_count = labels_count / len(train.labels)
        labels_count = 1 - labels_count
        labels_count = labels_count.sort_index()
        class_weights = torch.Tensor(labels_count)
        logger.info(f"Class weights: {class_weights}")
    else:
        class_weights = None
    
    #  Instantiate a LM and create the experiment accordingly
    model = LMForSequenceClassification(
        src_model_path,
        learning_rate,
        regularization,
        reg_strength,
        weight_decay=weight_decay,
        warmup_train_perc=warmup_train_perc,
        train_steps_count=train_steps_count,
        class_weights=class_weights
    )

# set some training stuff (loggers, callback)
    loggers = list()

    #  define training callbacks
    callbacks = list()
    monitor = 'val_F1'
    mode='max'
    if early_stop_epochs > 0:
        early_stopping = pl.callbacks.EarlyStopping(monitor, mode=mode, patience=early_stop_epochs)
        callbacks.append(early_stopping)

    
    model_checkpoint = pl.callbacks.ModelCheckpoint(
        monitor=monitor,
        mode=mode,
        dirpath=model_dir,
        save_last=True,
        save_top_k=ckpt_save_top_k,
        filename="PL-{epoch}-{val_loss:.3f}-{train_loss:.3f}",
    )

    # if checkpoint_every_n_epochs:
    #     from ear.custom_callbacks import CheckpointEveryNEpochs

    #     ckpt_n_epochs = CheckpointEveryNEpochs(checkpoint_every_n_epochs)
    #     callbacks.append(ckpt_n_epochs)

    # lr_monitor = pl.callbacks.LearningRateMonitor()
    callbacks.append(model_checkpoint)
    # callbacks.append(lr_monitor)

    trainer = pl.Trainer(
        gpus=gpus,
        accelerator=accelerator,
        max_epochs=max_epochs,
        logger=loggers,
        callbacks=callbacks,
        accumulate_grad_batches=accumulate_grad_batches,
        precision=precision,
        # resume_from_checkpoint=resume_from_checkpoint,
        log_every_n_steps=log_every_n_steps,
        gradient_clip_val=1
        # plugins=pl.plugins.DDPPlugin(find_unused_parameters=True),
    )

    trainer.fit(model, datamodule=dataset_module, ckpt_path=os.path.join(src_model_path, src_model))

    logging.info(f"Best model path: {model_checkpoint.best_model_path}")
    logging.info(f"Best model val_loss: {model_checkpoint.best_model_score}")

    #  print(trainer.logger[0].experiment.get_key())
    if run_test:
        
        # test on the dataset in-distribution
        trainer.test(datamodule=dataset_module, ckpt_path="best")

    if save_transformers_model:
        #  Save the tokenizer and the backbone LM with HuggingFace's serialization.
        #  To avoid mixing PL's and HuggingFace's serialization:
        #  https://github.com/PyTorchLightning/pytorch-lightning/issues/3096#issuecomment-686877242
        best_PL = LMForSequenceClassification.load_from_checkpoint(
            model_checkpoint.best_model_path
        )
        best_PL.get_backbone().save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        
        
if __name__ == "__main__":
    main()
