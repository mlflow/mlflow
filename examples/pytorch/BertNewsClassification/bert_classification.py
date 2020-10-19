# pylint: disable=W0221
# pylint: disable=W0613
# pylint: disable=E1102
# pylint: disable=W0223

import os
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import mlflow
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
from torchtext.utils import download_from_url, extract_archive
from torchtext.datasets.text_classification import URLS

from mlflow.pytorch.pytorch_autolog import autolog


class GPReviewDataset(Dataset):
    def __init__(self, reviews, targets, tokenizer, max_length):
        """
        Performs initialization of tokenizer

        :param reviews: AG news text
        :param targets: labels
        :param tokenizer: bert tokenizer
        :param max_length: maximum length of the news text

        """
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """
        :return: returns the number of datapoints in the dataframe

        """
        return len(self.reviews)

    def __getitem__(self, item):
        """
        Returns the review text and the targets of the specified item

        :param item: Index of sample review

        :return: Returns the dictionary of review text, input ids, attention mask, targets
        """
        review = str(self.reviews[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
            truncation=True,
        )

        return {
            "review_text": review,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "targets": torch.tensor(target, dtype=torch.long),
        }


class BertDataModule(pl.LightningDataModule):
    def __init__(self, **kwargs):
        """
        Initialization of inherited lightning data module
        """
        super(BertDataModule, self).__init__()
        self.PRE_TRAINED_MODEL_NAME = "bert-base-cased"
        self.df_train = None
        self.df_val = None
        self.df_test = None
        self.train_data_loader = None
        self.val_data_loader = None
        self.test_data_loader = None
        self.MAX_LEN = 160
        self.BATCH_SIZE = 16
        self.encoding = None
        self.tokenizer = None
        self.args = kwargs

    @staticmethod
    def to_label(rating):
        """
        Returns the rating minus one to make it for zero position start
        """
        rating = int(rating)
        return rating - 1

    def prepare_data(self):
        """
        Implementation of abstract class
        """
        pass

    def setup(self, stage=None):
        """
        Downloads the data, parse it and split the data into train, test, validation data

        :param stage: Stage - training or testing
        """
        # reading  the input
        dataset_tar = download_from_url(URLS["AG_NEWS"], root=".data")
        extracted_files = extract_archive(dataset_tar)

        train_csv_path = None
        for fname in extracted_files:
            if fname.endswith("train.csv"):
                train_csv_path = fname

        df = pd.read_csv(train_csv_path)

        df.columns = ["label", "title", "description"]
        df.sample(frac=1)
        df = df.iloc[:15000]

        df["label"] = df.label.apply(self.to_label)

        self.tokenizer = BertTokenizer.from_pretrained(self.PRE_TRAINED_MODEL_NAME)
        sample_txt = "wall street seems to be down this whole year due to financial crisis"

        self.encoding = self.tokenizer.encode_plus(
            sample_txt,
            max_length=32,
            add_special_tokens=True,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
            truncation=True,
        )

        token_lens = []

        for txt in df.description:
            tokens = self.tokenizer.encode(txt, max_length=512, truncation=True)
            token_lens.append(len(tokens))

        RANDOM_SEED = 42
        np.random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)

        df_train, df_test = train_test_split(df, test_size=0.1, random_state=RANDOM_SEED)
        df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED)

        self.df_train = df_train
        self.df_test = df_test
        self.df_val = df_val

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Returns the review text and the targets of the specified item

        :param parent_parser: Application specific parser

        :return: Returns the augmented arugument parser
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--batch-size",
            type=int,
            default=16,
            metavar="N",
            help="input batch size for training (default: 16)",
        )
        parser.add_argument(
            "--num-workers",
            type=int,
            default=3,
            metavar="N",
            help="number of workers (default: 0)",
        )
        return parser

    def create_data_loader(self, df, tokenizer, max_len, batch_size):
        """
        Generic data loader function

        :param df: Input dataframe
        :param tokenizer: bert tokenizer
        :param max_len: Max length of the news datapoint
        :param batch_size: Batch size for training

        :return: Returns the constructed dataloader
        """
        ds = GPReviewDataset(
            reviews=df.description.to_numpy(),
            targets=df.label.to_numpy(),
            tokenizer=tokenizer,
            max_length=max_len,
        )

        return DataLoader(
            ds, batch_size=self.args["batch_size"], num_workers=self.args["num_workers"]
        )

    def train_dataloader(self):
        """
        :return: output - Train data loader for the given input
        """
        self.train_data_loader = self.create_data_loader(
            self.df_train, self.tokenizer, self.MAX_LEN, self.args["batch_size"]
        )
        return self.train_data_loader

    def val_dataloader(self):
        """
        :return: output - Validation data loader for the given input
        """
        self.val_data_loader = self.create_data_loader(
            self.df_val, self.tokenizer, self.MAX_LEN, self.args["batch_size"]
        )
        return self.val_data_loader

    def test_dataloader(self):
        """
        :return: output - Test data loader for the given input
        """
        self.test_data_loader = self.create_data_loader(
            self.df_test, self.tokenizer, self.MAX_LEN, self.args["batch_size"]
        )
        return self.test_data_loader


class BertNewsClassifier(pl.LightningModule):
    def __init__(self, **kwargs):
        """
        Initializes the network, optimizer and scheduler
        """
        super(BertNewsClassifier, self).__init__()
        self.PRE_TRAINED_MODEL_NAME = "bert-base-cased"
        self.bert_model = BertModel.from_pretrained(self.PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        # assigning labels
        self.class_names = ["world", "Sports", "Business", "Sci/Tech"]
        n_classes = len(self.class_names)
        self.out = nn.Linear(self.bert_model.config.hidden_size, n_classes)
        self.scheduler = None
        self.optimizer = None
        self.args = kwargs

    def forward(self, input_ids, attention_mask):
        """
        :param input_ids: Input data
        :param attention_maks: Attention mask value

        :return: output - Type of news for the given news snippet
        """
        _, pooled_output = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        output = self.drop(pooled_output)
        F.softmax(output, dim=1)
        return self.out(output)

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Returns the review text and the targets of the specified item

        :param parent_parser: Application specific parser

        :return: Returns the augmented arugument parser
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--lr", type=float, default=1e-3, metavar="LR", help="learning rate (default: 1e-3)",
        )
        return parser

    def training_step(self, train_batch, batch_idx):
        """
        Training the data as batches and returns training loss on each batch

        :param train_batch: Batch data
        :param batch_idx: Batch indices

        :return: output - Training loss
        """
        input_ids = train_batch["input_ids"].to(self.device)
        attention_mask = train_batch["attention_mask"].to(self.device)
        targets = train_batch["targets"].to(self.device)
        output = self.forward(input_ids, attention_mask)
        loss = F.nll_loss(output, targets)
        return {"loss": loss}

    def test_step(self, test_batch, batch_idx):
        """
        Performs test and computes the accuracy of the model

        :param test_batch: Batch data
        :param batch_idx: Batch indices

        :return: output - Testing accuracy
        """
        input_ids = test_batch["input_ids"].to(self.device)
        attention_mask = test_batch["attention_mask"].to(self.device)
        targets = test_batch["targets"].to(self.device)
        output = self.forward(input_ids, attention_mask)
        _, y_hat = torch.max(output, dim=1)
        test_acc = accuracy_score(y_hat.cpu(), targets.cpu())
        return {"test_acc": torch.tensor(test_acc)}

    def validation_step(self, val_batch, batch_idx):
        """
        Performs validation of data in batches

        :param val_batch: Batch data
        :param batch_idx: Batch indices

        :return: output - valid step loss
        """

        input_ids = val_batch["input_ids"].to(self.device)
        attention_mask = val_batch["attention_mask"].to(self.device)
        targets = val_batch["targets"].to(self.device)
        output = self.forward(input_ids, attention_mask)
        loss = F.nll_loss(output, targets)
        return {"val_step_loss": loss}

    def validation_epoch_end(self, outputs):
        """
        Computes average validation accuracy

        :param outputs: outputs after every epoch end

        :return: output - average valid loss
        """
        avg_loss = torch.stack([x["val_step_loss"] for x in outputs]).mean()
        self.log("val_loss", avg_loss)

    def test_epoch_end(self, outputs):
        """
        Computes average test accuracy score

        :param outputs: outputs after every epoch end

        :return: output - average test loss
        """
        avg_test_acc = torch.stack([x["test_acc"] for x in outputs]).mean()
        self.log("avg_test_acc", avg_test_acc)

    def configure_optimizers(self):
        """
        Initializes the optimizer and learning rate scheduler

        :return: output - Initialized optimizer and scheduler
        """
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.args["lr"])
        self.scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.2, patience=2, min_lr=1e-6, verbose=True,
            ),
            "monitor": "val_loss",
        }
        return [self.optimizer], [self.scheduler]

    def cross_entropy_loss(self):
        """
        Initializes the loss function

        :return: output - Initialized cross entropy loss function
        """
        return nn.CrossEntropyLoss().to(self.device)

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        second_order_closure=None,
        on_tpu=False,
        using_lbfgs=False,
        using_native_amp=False,
    ):
        """
        Training step function which runs for the given number of epochs

        :param epoch: Number of epochs to train
        :param batch_idx: batch indices
        :param optimizer: Optimizer to be used in training step
        """
        self.optimizer.step()
        self.optimizer.zero_grad()


if __name__ == "__main__":
    parser = ArgumentParser(description="Bert-News Classifier Example")

    # Add trainer specific arguments
    parser.add_argument(
        "--tracking-uri", type=str, default="http://localhost:5000/", help="mlflow tracking uri"
    )
    parser.add_argument(
        "--max-epochs", type=int, default=5, help="number of epochs to run (default: 5)"
    )
    parser.add_argument(
        "--gpus", type=int, default=0, help="Number of gpus - by default runs on CPU"
    )
    parser.add_argument(
        "--accelerator", type=str, default=None, help="Distributed Backend - (default: None)",
    )
    parser = BertNewsClassifier.add_model_specific_args(parent_parser=parser)

    parser = BertDataModule.add_model_specific_args(parent_parser=parser)

    autolog()

    args = parser.parse_args()
    dict_args = vars(args)
    mlflow.set_tracking_uri(dict_args["tracking_uri"])

    dm = BertDataModule(**dict_args)
    dm.prepare_data()
    dm.setup(stage="fit")

    model = BertNewsClassifier(**dict_args)
    early_stopping = EarlyStopping(monitor="val_loss", mode="min", verbose=True)

    checkpoint_callback = ModelCheckpoint(
        filepath=os.getcwd(), save_top_k=1, verbose=True, monitor="val_loss", mode="min", prefix="",
    )
    lr_logger = LearningRateMonitor()

    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[lr_logger, early_stopping],
        checkpoint_callback=checkpoint_callback,
        limit_train_batches=0.1,
    )
    trainer.fit(model, dm)
    trainer.test()
