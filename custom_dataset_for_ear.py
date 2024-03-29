"""Collect and expose datasets for experiments."""
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import pytorch_lightning as pl
import torch
import pandas as pd
import logging
import os


logging.basicConfig(
    format="%(levelname)s:%(asctime)s:%(module)s:%(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


GAB_DATASETS = ["gab25k"]




AVAIL_DATASETS = GAB_DATASETS



def get_dataset_by_name(name: str, base_dir=None, round=0):
    path = os.path.join(base_dir, name) if base_dir else name
    
    train, dev, test = None, None, None
    if name in GAB_DATASETS:
        train = GabDataset(f"./soc/data/majority_gab_dataset_25k/train.jsonl")
        dev = GabDataset(f"./soc/data/majority_gab_dataset_25k/dev.jsonl")
        test = GabDataset(f"./soc/data/majority_gab_dataset_25k/test.jsonl")
    
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
        round=None,
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
        self.round = round

        self.train, self.val, self.test = get_dataset_by_name(dataset_name, round=round)
        self.train_steps = int(len(self.train) / batch_size)

    def prepare_data(self):
        train, val, test = self.train, self.val, self.test
        self.encodings_split = []
        for split in [train, val, test]:
            logger.info("Tokenizing...")
            encodings = self.tokenizer(
                split.get_texts(),
                truncation=True,
                padding="max_length",
                max_length=self.max_seq_length,
                return_tensors="pt",
            )
            self.encodings_split.append(encodings)

    def setup(self, stage=None):
        if stage == "fit":
            train, val = self.train, self.val

            logging.info(f"TRAIN len: {len(train)}")
            logging.info(f"VAL len: {len(val)}")

            train_encodings = self.encodings_split[0]
            train_labels = torch.LongTensor([r["label"] for r in train])
            self.train_data = EncodedDataset(train_encodings, train_labels)

            val_encodings = self.encodings_split[1]
            val_labels = torch.LongTensor([r["label"] for r in val])
            self.val_data = EncodedDataset(val_encodings, val_labels)

        elif stage == "test":
            test = self.test
            logging.info(f"TEST len: {len(test)}")

            test_encodings = self.encodings_split[2]
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
