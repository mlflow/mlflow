import os
import shutil
import tempfile

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext.data
from captum.attr import LayerIntegratedGradients, TokenReferenceBase, visualization
from prettytable import PrettyTable
from torchtext import vocab

import mlflow
import spacy
from mlflow.utils.autologging_utils import try_mlflow_log

mlflow.start_run(run_name="CaptumExample")


nlp = spacy.load("en")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNN(nn.Module):
    def __init__(
        self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, pad_idx
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.convs = nn.ModuleList(
            [
                nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(fs, embedding_dim))
                for fs in filter_sizes
            ]
        )

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text = [sent len, batch size]

        # text = text.permute(1, 0)

        # text = [batch size, sent len]

        embedded = self.embedding(text)

        # embedded = [batch size, sent len, emb dim]

        embedded = embedded.unsqueeze(1)

        # embedded = [batch size, 1, sent len, emb dim]

        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]

        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        # pooled_n = [batch size, n_filters]

        cat = self.dropout(torch.cat(pooled, dim=1))

        # cat = [batch size, n_filters * len(filter_sizes)]

        return self.fc(cat)


def count_model_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0

    for name, parameter in model.named_parameters():

        if not parameter.requires_grad:
            continue

        param = parameter.nonzero(as_tuple=False).size(0)
        table.add_row([name, param])
        total_params += param
    return table, total_params


# Logic for downloading the model needs to be added

# https://github.com/pytorch/captum/blob/master/tutorials/models/imdb-model-cnn.pt

model = torch.load("imdb-model-cnn.pt")
model.eval()
model = model.to(device)

summary, params = count_model_parameters(model)

tempdir = tempfile.mkdtemp()
try:
    summary_file = os.path.join(tempdir, "model_summary.txt")
    with open(summary_file, "w") as f:
        f.write(str(summary))

    try_mlflow_log(mlflow.log_artifact, local_path=summary_file)
finally:
    shutil.rmtree(tempdir)


def forward_with_sigmoid(input):
    return torch.sigmoid(model(input))


# https://ai.stanford.edu/~amaas/data/sentiment/

TEXT = torchtext.data.Field(lower=True, tokenize="spacy")
Label = torchtext.data.LabelField(dtype=torch.float)

train, test = torchtext.datasets.IMDB.splits(
    text_field=TEXT, label_field=Label, train="train", test="test", path="data/aclImdb"
)

test, _ = test.split(split_ratio=0.04)


try_mlflow_log(mlflow.log_param, "Train Size", len(train))
try_mlflow_log(mlflow.log_param, "Test Size", len(test))


# It will automatically download Glove embedding one time

loaded_vectors = vocab.GloVe(name="6B", dim=50)

TEXT.build_vocab(train, vectors=loaded_vectors, max_size=len(loaded_vectors.stoi))
TEXT.vocab.set_vectors(
    stoi=loaded_vectors.stoi, vectors=loaded_vectors.vectors, dim=loaded_vectors.dim
)
Label.build_vocab(train)

print("Vocabulary Size: ", len(TEXT.vocab))

mlflow.log_param("Vocabulary Size", len(TEXT.vocab))

PAD_IND = TEXT.vocab.stoi["pad"]
token_reference = TokenReferenceBase(reference_token_idx=PAD_IND)
lig = LayerIntegratedGradients(model, model.embedding)

vis_data_records_ig = []


def interpret_sentence(model, sentence, min_len=7, label=0):
    text = [tok.text for tok in nlp.tokenizer(sentence)]
    if len(text) < min_len:
        text += ["pad"] * (min_len - len(text))
    indexed = [TEXT.vocab.stoi[t] for t in text]

    model.zero_grad()

    input_indices = torch.tensor(indexed, device=device)
    input_indices = input_indices.unsqueeze(0)

    # input_indices dim: [sequence_length]
    seq_length = min_len

    # predict
    pred = forward_with_sigmoid(input_indices).item()
    pred_ind = round(pred)

    # generate reference indices for each sample
    reference_indices = token_reference.generate_reference(seq_length, device=device).unsqueeze(0)

    # compute attributions and approximation delta using layer integrated gradients
    attributions_ig, delta = lig.attribute(
        input_indices, reference_indices, n_steps=500, return_convergence_delta=True
    )

    print("pred: ", Label.vocab.itos[pred_ind], "(", "%.2f" % pred, ")", ", delta: ", abs(delta))

    add_attributions_to_visualizer(
        attributions_ig, text, pred, pred_ind, label, delta, vis_data_records_ig
    )


def add_attributions_to_visualizer(
    attributions, text, pred, pred_ind, label, delta, vis_data_records
):
    attributions = attributions.sum(dim=2).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    attributions = attributions.cpu().detach().numpy()

    # storing couple samples in an array for visualization purposes
    vis_data_records.append(
        visualization.VisualizationDataRecord(
            attributions,
            pred,
            Label.vocab.itos[pred_ind],
            Label.vocab.itos[label],
            Label.vocab.itos[1],
            attributions.sum(),
            text,
            delta,
        )
    )


interpret_sentence(model, "It was a fantastic performance !", label=1)
interpret_sentence(model, "Best film ever", label=1)
interpret_sentence(model, "Such a great show!", label=1)
interpret_sentence(model, "It was a horrible movie", label=0)
interpret_sentence(model, "I've never watched something as bad", label=0)
interpret_sentence(model, "It is a disgusting movie!", label=0)


result = pd.DataFrame(
    columns=[
        "raw_input",
        "attr_class",
        "attr_score",
        "convergence_score",
        "pred_prob",
        "true_class",
        "word_attributions",
    ]
)


for i in range(0, len(vis_data_records_ig)):
    result.loc[i] = [
        vis_data_records_ig[i].raw_input,
        vis_data_records_ig[i].attr_class,
        vis_data_records_ig[i].attr_score,
        vis_data_records_ig[i].convergence_score,
        vis_data_records_ig[i].pred_prob,
        vis_data_records_ig[i].true_class,
        vis_data_records_ig[i].word_attributions,
    ]


word_attr_file = "word_attr.csv"
tempdir = tempfile.mkdtemp()
try:
    word_attr_file_path = os.path.join(tempdir, word_attr_file)
    result.to_csv(word_attr_file_path)

    try_mlflow_log(mlflow.log_artifact, local_path=word_attr_file_path)
finally:
    shutil.rmtree(tempdir)


print("Visualize attributions based on Integrated Gradients")
visualization.visualize_text(vis_data_records_ig)


for i in range(0, len(vis_data_records_ig)):
    input_string = " ".join(vis_data_records_ig[i].raw_input)
    score = vis_data_records_ig[i].attr_score
    try_mlflow_log(mlflow.log_metric, "Attribution Score" + str(i), float(score), step=i)


mlflow.end_run()
