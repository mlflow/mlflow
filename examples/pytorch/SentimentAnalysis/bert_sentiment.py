import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateLogger
from pytorch_lightning.loggers import MLFlowLogger
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer

from mlflow.pytorch.pytorch_autolog import __MLflowPLCallback


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
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "review_text": review,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].faltten(),
            "targets": torch.tensor(target, dtype=torch.long)
        }


class BertSentinmentClassifier(pl.LightningModule):
    def __init__(self, **kwargs):
        super(BertSentinmentClassifier, self).__init__()
        self.PRE_TRAINED_MODEL_NAME = "bert_base_cased"
        self.bert_model = BertModel.from_pretrained(self.PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        # assigning labels
        self.class_names = ["negative", "neutral", "positive"]
        n_classes = len(self.class_names)
        self.out = nn.Linear(self.bert_model.config.hidden_size, n_classes)
        self.args = kwargs

    def forward(self, input_ids, attention_mask):
        print("inside the forward function")
        _, pooled_output = self.bert_model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        output = self.Dropout(pooled_output)
        F.softmax(output, dim=1)
        return self.out(output)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--batch-size",
            type=int,
            default=64,
            metavar="N",
            help="input batch size for training (default: 64)",
        )
        parser.add_argument(
            "--num-workers",
            type=int,
            default=0,
            metavar="N",
            help="number of workers (default: 0)",
        )
        parser.add_argument(
            "--lr",
            type=float,
            default=1e-3,
            metavar="LR",
            help="learning rate (default: 1e-3)",
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

        self.df_train, df_test = train_test_split(
            df, test_size=0.1, random_state=RANDOM_SEED
        )
        self.df_val, self.df_test = train_test_split(
            df_test, test_size=0.5, random_state=RANDOM_SEED
        )

        self.BATCH_SIZE = 16

    def create_data_loader(self, df, tokenizer, max_len, batch_size):
        ds = GPReviewDataset(
            reviews=df.content.to_numpy(),
            targets=df.sentiment.to_numpy(),
            tokenizer=tokenizer,
            max_len=max_len,
        )

        return DataLoader(ds, batch_size=batch_size, num_workers=0)

    def train_dataloader(self):
        print("In Train Data Loader")
        self.train_data_loader = self.create_data_loader(
            self.df_train, self.tokenizer, self.MAX_LEN, self.BATCH_SIZE
        )
        return self.train_data_loader

    def val_dataloader(self):
        print("In Val Data Loader")
        self.val_data_loader = self.create_data_loader(
            self.df_val, self.tokenizer, self.MAX_LEN, self.BATCH_SIZE
        )
        return self.val_data_loader

    def test_dataloader(self):
        print("In Test Data Loader")
        self.test_data_loader = self.create_data_loader(
            self.df_test, self.tokenizer, self.MAX_LEN, self.BATCH_SIZE
        )
        return self.test_data_loader

    def training_step(self, train_batch, batch_idx):
        """training the data as batches and returns training loss on each batch"""
        input_ids = train_batch["input_ids"].to(self.device)
        attention_mask = train_batch["attention_mask"].to(self.device)
        targets = train_batch["targets"].to(self.device)
        output = self.forward(input_ids, attention_mask)
        loss = F.nll_loss(output, targets)
        return {"loss": loss}

    def test_step(self, test_batch, batch_idx):
        '''Performs test and computes the accuracy of the model'''
        input_ids = test_batch["input_ids"].to(self.device)
        attention_mask = test_batch["attention_mask"].to(self.device)
        targets = test_batch["targets"].to(self.device)
        output = self.forward(input_ids, attention_mask)
        a, y_hat = torch.max(output, dim=1)
        test_acc = accuracy_score(y_hat.cpu(), targets.cpu())
        return {"test_acc": torch.tensor(test_acc)}

    def validation_step(self, val_batch, batch_idx):
        """ Performs validation of data in batches"""
        input_ids = val_batch["input_ids"].to(self.device)
        attention_mask = val_batch["attention_mask"].to(self.device)
        targets = val_batch["targets"].to(self.device)
        output = self.forward(input_ids, attention_mask)
        loss = F.nll_loss(output, targets)
        return {"val_step_loss": loss}

    def validation_epoch_end(self, outputs):
        """ Computes average validation accuracy"""
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        return {"avg_val_loss": avg_loss}

    def test_epoch_end(self, outputs):
        """Computes average test accuracy score"""
        avg_test_acc = torch.stack([x["test_acc"] for x in outputs]).mean()
        return {"avg_test_acc": avg_test_acc}



    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.args["lr"])
        self.scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.2,
                patience=2,
                min_lr=1e-6,
                verbose=True,
            )
        }
        return [self.optimizer], [self.scheduler]

    def cross_entropy_loss(self):
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
        self.optimizer.step()
        self.optimizer.zero_grad()



if __name__ == "__main__":
    parser = ArgumentParser(description="Bert-Sentiment Classifier Example")

    # Add trainer specific arguments
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

    args = parser.parse_args()
    dict_args = vars(args)
    model = BertSentinmentClassifier(**dict_args)
    mlflow_logger = MLFlowLogger(
        experiment_name="EXPERIMENT_NAME", tracking_uri="http://IP:PORT/"
    )
    early_stopping = EarlyStopping(monitor="val_loss", mode="min", verbose=True)

    checkpoint_callback = ModelCheckpoint(
        filepath=os.getcwd(),
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
        prefix="",
    )
    lr_logger = LearningRateLogger()

    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=mlflow_logger,
        callbacks=[__MLflowPLCallback(), lr_logger],
        early_stop_callback=early_stopping,
        checkpoint_callback=checkpoint_callback,
        train_percent_check=0.1,
    )
    trainer.fit(model)
    trainer.test()
