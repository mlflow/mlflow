# pylint: disable=W0221
# pylint: disable=W0613
# pylint: disable=E1102

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
    LearningRateLogger,
)
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer

from mlflow.pytorch.pytorch_autolog import autolog


class GPReviewDataset(Dataset):
    def __init__(self, reviews, targets, tokenizer, max_length):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
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


class BertSentinmentClassifier(pl.LightningModule):
    def __init__(self, **kwargs):
        super(BertSentinmentClassifier, self).__init__()
        self.tokenizer = None
        self.encoding = None
        self.MAX_LEN = 160
        self.df_train = None
        self.df_val = None
        self.df_test = None
        self.optimizer = None
        self.scheduler = None
        self.train_data_loader = None
        self.val_data_loader = None
        self.test_data_loader = None
        self.BATCH_SIZE = 16
        self.PRE_TRAINED_MODEL_NAME = "bert-base-cased"
        self.bert_model = BertModel.from_pretrained(self.PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        # assigning labels
        self.class_names = ["negative", "neutral", "positive"]
        n_classes = len(self.class_names)
        self.out = nn.Linear(self.bert_model.config.hidden_size, n_classes)
        self.args = kwargs

    def forward(self, input_ids, attention_mask):
        """
        :param input_ids: Input sentences from the batch
        :param attention_mask: Attention mask returned by the encoder

        :return: output - sentiment for the input text
        """
        _, pooled_output = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        output = self.drop(pooled_output)
        F.softmax(output, dim=1)
        return self.out(output)

    @staticmethod
    def add_model_specific_args(parent_parser):
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
            default=1,
            metavar="N",
            help="number of workers (default: 0)",
        )
        parser.add_argument(
            "--lr", type=float, default=1e-3, metavar="LR", help="learning rate (default: 1e-3)",
        )
        return parser

    @staticmethod
    def to_sentiment(rating):
        rating = int(rating)
        if rating < 2:
            return 0
        if rating == 3:
            return 1
        else:
            return 2

    def prepare_data(self):
        """
        Prepares the data for training and prediction
        """

        print("preparing the data")

        # reading  the input
        df = pd.read_csv("https://drive.google.com/uc?id=1zdmewp7ayS4js4VtrJEHzAheSW-5NBZv")
        print("data_shape {}".format(df.shape))

        # setting sentiment
        df["sentiment"] = df.score.apply(self.to_sentiment)

        self.tokenizer = BertTokenizer.from_pretrained(self.PRE_TRAINED_MODEL_NAME)
        sample_txt = "when was i last outside? i am stuck at home for 2 weeks."

        self.encoding = self.tokenizer.encode_plus(
            sample_txt,
            max_length=32,
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors="pt",  # Return PyTorch tensors
            truncation=True,
        )

        token_lens = []

        for txt in df.content:
            tokens = self.tokenizer.encode(txt, max_length=512, truncation=True)
            token_lens.append(len(tokens))

        self.MAX_LEN = 160

        RANDOM_SEED = 42
        np.random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)

        self.df_train, df_test = train_test_split(df, test_size=0.1, random_state=RANDOM_SEED)
        self.df_val, self.df_test = train_test_split(
            df_test, test_size=0.5, random_state=RANDOM_SEED
        )

        self.BATCH_SIZE = 16

    def create_data_loader(self, df, tokenizer, max_len, batch_size):
        """
        :param df: DataFrame input
        :param tokenizer: Bert tokenizer
        :param max_len: maximum length of the input sentence
        :param batch_size: Input batch size

        :return: output - Corresponding data loader for the given input
        """
        ds = GPReviewDataset(
            reviews=df.content.to_numpy(),
            targets=df.sentiment.to_numpy(),
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
        print("In Train Data Loader")
        self.train_data_loader = self.create_data_loader(
            self.df_train, self.tokenizer, self.MAX_LEN, self.args["batch_size"]
        )
        return self.train_data_loader

    def val_dataloader(self):
        """
        :return: output - Validation data loader for the given input
        """
        print("In Val Data Loader")
        self.val_data_loader = self.create_data_loader(
            self.df_val, self.tokenizer, self.MAX_LEN, self.args["batch_size"]
        )
        return self.val_data_loader

    def test_dataloader(self):
        """
        :return: output - Test data loader for the given input
        """
        print("In Test Data Loader")
        self.test_data_loader = self.create_data_loader(
            self.df_test, self.tokenizer, self.MAX_LEN, self.args["batch_size"]
        )
        return self.test_data_loader

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
        return {"val_loss": avg_loss}

    def test_epoch_end(self, outputs):
        """
        Computes average test accuracy score
        :param outputs: outputs after every epoch end

        :return: output - average test loss
        """
        avg_test_acc = torch.stack([x["test_acc"] for x in outputs]).mean()
        return {"avg_test_acc": avg_test_acc}

    def configure_optimizers(self):
        """
        Initializes the optimizer and learning rate scheduler

        :return: output - Initialized optimizer and scheduler
        """
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.args["lr"])
        self.scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.2, patience=2, min_lr=1e-6, verbose=True,
            )
        }
        return [self.optimizer], [self.scheduler]

    def cross_entropy_loss(self):
        """
        Initializes the loss function

        :return: output - Initialized cross entropy loss function
        """
        print("In Loss Function")
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
    parser = ArgumentParser(description="Bert-Sentiment Classifier Example")

    # Add trainer specific arguments
    parser.add_argument(
        "--tracking_uri", type=str, default="http://localhost:5000/", help="mlflow tracking uri"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=5, help="number of epochs to run (default: 5)"
    )
    parser.add_argument(
        "--gpus", type=int, default=0, help="Number of gpus - by default runs on CPU"
    )
    parser.add_argument(
        "--distributed_backend",
        type=str,
        default=None,
        help="Distributed Backend - (default: None)",
    )
    parser = BertSentinmentClassifier.add_model_specific_args(parent_parser=parser)

    autolog()

    args = parser.parse_args()
    dict_args = vars(args)
    mlflow.set_tracking_uri(dict_args["tracking_uri"])
    model = BertSentinmentClassifier(**dict_args)
    early_stopping = EarlyStopping(monitor="val_loss", mode="min", verbose=True)

    checkpoint_callback = ModelCheckpoint(
        filepath=os.getcwd(), save_top_k=1, verbose=True, monitor="val_loss", mode="min", prefix="",
    )
    lr_logger = LearningRateLogger()

    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[lr_logger],
        early_stop_callback=early_stopping,
        checkpoint_callback=checkpoint_callback,
        train_percent_check=0.1,
    )
    trainer.fit(model)
    trainer.test()
