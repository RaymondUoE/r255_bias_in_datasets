import os
import logging
import json

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer
import pandas as pd
from torch.utils.data import Dataset, DataLoader


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
data_path = './data/Dynamically-Generated-Hate-Speech-Dataset/Dynamically Generated Hate Dataset v0.2.3.csv'
data = pd.read_csv(data_path, index_col=0)
max_seq_length = 128

class DynaDataset(Dataset):
    def __init__(self, data: pd.DataFrame, tokenizer):
        self.texts = data["text"].tolist()
        self.labels = torch.LongTensor((data['label']=='hate').astype(int).to_list())
        self.encodings = tokenizer(
                self.texts,
                truncation=True,
                padding="max_length",
                max_length=max_seq_length,
                return_tensors="pt",
            )

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

        # return {"text": self.texts[idx], "label": self.labels[idx]}

    def __len__(self):
        return len(self.labels)

    def get_texts(self):
        return self.texts

    def get_labels(self):
        return self.labels


def main():
    for seed in range(0, 10):
        run_with_seed(seed=seed)
        break





def run_with_seed(seed):
    pl.seed_everything(seed)
    parent_path = f'./ear_bert/entropybert-gab25k-{seed}-0.01/'
    tokenizer = AutoTokenizer.from_pretrained(parent_path)
    datasets_rounds_split = []
    dataloader_rounds_split = []
    for round in range(1, 5):
        train = DynaDataset(data[(data['split']=='train')&(data['round.base']==round)], tokenizer)
        dev = DynaDataset(data[(data['split']=='dev')&(data['round.base']==round)], tokenizer)
        test = DynaDataset(data[(data['split']=='test')&(data['round.base']==round)], tokenizer)
        # datasets_rounds_split.append([train, dev, test])
        dataloader_rounds_split.append(
            [
                DataLoader(train, batch_size=32, shuffle=True),
                DataLoader(dev, batch_size=32, shuffle=False),
                DataLoader(test, batch_size=32, shuffle=False),
            ]
        )

        
    for round in range(1,5):
        round_zero_index = round - 1
        if round == 1:
            src_model_path = f'./ear_bert/entropybert-gab25k-{seed}-0.01/'
        else:
            src_model_path = f'./ear_bert/entropybert-gab25k-{seed}-0.01/R{round-1}'
        
        save_path = os.path.join(parent_path, f'R{round}')
        
        src_model = [x for x in os.listdir(src_model_path) if x.startswith('PL-epoch')][0]
        warmup_train_perc = 0.1
        max_epochs = 20
        balanced_loss = True
        
        train_steps_count = None
        if warmup_train_perc:
            logger.info(f"Warmup linear LR requested with {warmup_train_perc}")
            train_steps_count = (
                int(len(dataloader_rounds_split[round_zero_index][0].dataset) / 32) * max_epochs
            )
            logger.info(f"Total training steps: {train_steps_count}")

            logger.info(f"Total training steps (gpu-normalized): {train_steps_count}")

        if balanced_loss:
            
            labels_count = pd.Series(dataloader_rounds_split[round_zero_index][0].dataset.labels).value_counts()
            labels_count = labels_count / len(dataloader_rounds_split[round_zero_index][0].dataset.labels)
            labels_count = 1 - labels_count
            labels_count = labels_count.sort_index()
            class_weights = torch.Tensor(labels_count)
            logger.info(f"Class weights: {class_weights}")
        else:
            class_weights = None
        
        
        # model = LMForSequenceClassification.load_from_checkpoint(os.path.join(src_model_path, src_model))
        model = LMForSequenceClassification(
            src_model_path,
            learning_rate=2e-5,
            regularization='entropy',
            reg_strength=0.01,
            weight_decay=0.1,
            warmup_train_perc=warmup_train_perc,
            class_weights=class_weights
        )
        
        # model.hparams.class_weights=None
        loggers = list()

    #  define training callbacks
        callbacks = list()
        monitor = 'val_F1'
        mode='max'
        
        early_stopping = pl.callbacks.EarlyStopping(monitor, mode=mode, patience=5)
        callbacks.append(early_stopping)

        
        model_checkpoint = pl.callbacks.ModelCheckpoint(
            monitor=monitor,
            mode=mode,
            dirpath=os.path.join(parent_path, f'R{round}'),
            save_last=True,
            save_top_k=1,
            filename="PL-{epoch}-{val_loss:.3f}-{train_loss:.3f}-R{round}",
        )
        callbacks.append(model_checkpoint)
        
        trainer = pl.Trainer(
            accelerator='gpu',
            max_epochs=max_epochs,
            logger=loggers,
            callbacks=callbacks,
            accumulate_grad_batches=1,
            # precision=precision,
            # resume_from_checkpoint=resume_from_checkpoint,
            # log_every_n_steps=log_every_n_steps,
            gradient_clip_val=1
            # plugins=pl.plugins.DDPPlugin(find_unused_parameters=True),
        )
        trainer.fit(model, 
                    train_dataloaders=dataloader_rounds_split[round_zero_index][0],
                    val_dataloaders=dataloader_rounds_split[round_zero_index][1], 
                    )
        
        # data_modules[round].setup(stage='test')
        
        # trainer = pl.Trainer(accelerator='gpu')
        # model = LMForSequenceClassification.load_from_checkpoint(
        #     model_checkpoint.best_model_path
        # )
        
        # val_result = trainer.validate(model, dataloader_rounds_split[round_zero_index][1])
        all_tests = [dataloader_rounds_split[round_zero_index][1]] + [x[2] for x in dataloader_rounds_split]
        results = {}
        keys = [f'ValR{round}', 'TestR1', 'TestR2', 'TestR3', 'TestR4']
        for k, dataloader in zip(keys, all_tests):
            test_result = trainer.test(dataloaders=dataloader, ckpt_path="best")
            results[k] = test_result
            
        with open(os.path.join(save_path, f'results.json'), 'w') as fout:
            json.dump(results, fout)
        
        model = LMForSequenceClassification.load_from_checkpoint(
            model_checkpoint.best_model_path
        )
        model.get_backbone().save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

if __name__ == "__main__":
    main()
