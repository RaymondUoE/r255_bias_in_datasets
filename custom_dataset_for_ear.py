"""Collect and expose datasets for experiments."""
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import pytorch_lightning as pl
import torch
import pandas as pd
from operator import itemgetter
import logging
import os


logging.basicConfig(
    format="%(levelname)s:%(asctime)s:%(module)s:%(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


GAB_DATASETS = ["gab25k"]

DYNA_DATASETS= ['dyna']



AVAIL_DATASETS = (
    GAB_DATASETS
    + DYNA_DATASETS
)


def get_dataset_by_name(name: str, base_dir=None, config=None):
    path = os.path.join(base_dir, name) if base_dir else name
    
    train, dev, test = None, None, None
    if name in GAB_DATASETS:
        train = GabDataset(f"./soc/data/majority_gab_dataset_25k/train.jsonl")
        dev = GabDataset(f"./soc/data/majority_gab_dataset_25k/dev.jsonl")
        test = GabDataset(f"./soc/data/majority_gab_dataset_25k/test.jsonl")
    elif name in DYNA_DATASETS:
        data = pd.read_csv('./data/Dynamically-Generated-Hate-Speech-Dataset/Dynamically Generated Hate Dataset v0.2.3.csv', index_col=0)
        trains = []
        devs = []
        tests = []
        for round in config['rounds']:
            trains.append(DynaDataset(data[(data['split']=='train')&(data['round.base']==round)]))
            devs.append(DynaDataset(data[(data['split']=='dev')&(data['round.base']==round)]))
            tests.append(DynaDataset(data[(data['split']=='test')&(data['round.base']==round)]))
        train = ConcatDataset(trains)
        dev = ConcatDataset(devs)
        test = ConcatDataset(tests)
    else:
        raise ValueError(f"Can't recognize dataset name {name}")
    return train, dev, test


def get_tokenized_path(path: str):
    base_dir, filename = os.path.dirname(path), os.path.basename(path)
    return os.path.join(base_dir, f"{os.path.splitext(filename)[0]}.pt")




class GabDataset(Dataset):
    def __init__(self, path: str):
        self.path = path
        data = pd.read_json(path, lines=True)
        self.texts = data["Text"].tolist()
        self.labels = (data[['cv','hd']].sum(axis=1) > 0).astype(int).tolist()
        self.tokenized_path = get_tokenized_path(path)

    def __getitem__(self, idx):
        return {"text": self.texts[idx], "label": self.labels[idx]}

    def __len__(self):
        return len(self.labels)

    def get_texts(self):
        return self.texts

    def get_labels(self):
        return self.labels

    # @classmethod
    # def build_dataset(cls, split: str):
    #     return cls(f"./soc/data/majority_gab_dataset_25k/'{split}.jsonl")


class DynaDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.texts = data["text"].tolist()
        self.labels = (data['label']=='hate').astype(int).to_list()
        # self.tokenized_path = get_tokenized_path(path)

    def __getitem__(self, idx):
        return {"text": self.texts[idx], "label": self.labels[idx]}

    def __len__(self):
        return len(self.labels)

    def get_texts(self):
        return self.texts

    def get_labels(self):
        return self.labels



class TokenizerDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name,
        tokenizer,
        batch_size,
        max_seq_length,
        num_workers,
        pin_memory,
        load_pre_tokenized=False,
        store_pre_tokenized=False,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.load_pre_tokenized = load_pre_tokenized
        self.store_pre_tokenized = store_pre_tokenized

        self.train, self.val, self.test = get_dataset_by_name(dataset_name)
        self.train_steps = int(len(self.train) / batch_size)

    def prepare_data(self):
        train, val, test = self.train, self.val, self.test

        for split in [train, val, test]:
            if self.load_pre_tokenized and os.path.exists(split.tokenized_path):
                logging.info(
                    """
                    Loading pre-tokenized dataset.
                    Beware! Using pre-tokenized embeddings could not match you choice for max_length
                    """
                )
                continue

            if self.load_pre_tokenized:
                logging.info(f"Load tokenized but {split.tokenized_path} is not found")

            logger.info("Tokenizing...")
            encodings = self.tokenizer(
                split.get_texts(),
                truncation=True,
                padding="max_length",
                max_length=self.max_seq_length,
                return_tensors="pt",
            )

            if self.store_pre_tokenized:
                logger.info(f"Saving to {split.tokenized_path}")
                torch.save(encodings, split.tokenized_path)

    def setup(self, stage=None):
        if stage == "fit":
            train, val = self.train, self.val

            logging.info(f"TRAIN len: {len(train)}")
            logging.info(f"VAL len: {len(val)}")

            train_encodings = torch.load(train.tokenized_path)
            train_labels = torch.LongTensor([r["label"] for r in train])
            self.train_data = EncodedDataset(train_encodings, train_labels)

            val_encodings = torch.load(val.tokenized_path)
            val_labels = torch.LongTensor([r["label"] for r in val])
            self.val_data = EncodedDataset(val_encodings, val_labels)

        elif stage == "test":
            test = self.test
            logging.info(f"TEST len: {len(test)}")

            test_encodings = torch.load(test.tokenized_path)
            test_labels = torch.LongTensor([r["label"] for r in test])
            self.test_data = EncodedDataset(test_encodings, test_labels)

        else:
            raise ValueError(f"Stage {stage} not known")

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


class EncodedDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return self.labels.shape[0]


class PlainDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __getitem__(self, index):
        return {"text": self.texts[index], "label": self.labels[index]}

    def __len__(self):
        return len(self.labels)
