"""
Getting started with Captum - Titanic Data Analysis
"""

# Initial imports
import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients, LayerConductance, NeuronConductance
from prettytable import PrettyTable
from scipy import stats
from sklearn.model_selection import train_test_split
from torch import nn

import mlflow


def get_titanic():
    """
    we now preprocess the data by converting some categorical features such as
    gender, location of embarcation, and passenger class into one-hot encodings
    We also remove some features that are more difficult to analyze
    After processing, the features we have are:
    Age: Passenger Age
    Sibsp: Number of Siblings / Spouses Aboard
    Parch: Number of Parents / Children Aboard
    Fare: Fare Amount Paid in British Pounds
    Female: Binary variable indicating whether passenger is female
    Male: Binary variable indicating whether passenger is male
    EmbarkC : Binary var indicating whether passenger embarked @ Cherbourg
    EmbarkQ : Binary var indicating whether passenger embarked @ Queenstown
    EmbarkS : Binary var indicating whether passenger embarked @ Southampton
    Class1 : Binary var indicating whether passenger was in first class
    Class2 : Binary var indicating whether passenger was in second class
    Class3 : Binary var indicating whether passenger was in third class
    """
    data_path = "titanic3.csv"
    titanic_data = pd.read_csv(data_path)
    titanic_data = pd.concat(
        [
            titanic_data,
            pd.get_dummies(titanic_data["sex"], dtype=np.uint8),
            pd.get_dummies(titanic_data["embarked"], prefix="embark", dtype=np.uint8),
            pd.get_dummies(titanic_data["pclass"], prefix="class", dtype=np.uint8),
        ],
        axis=1,
    )

    titanic_data["age"] = titanic_data["age"].fillna(titanic_data["age"].mean())
    titanic_data["fare"] = titanic_data["fare"].fillna(titanic_data["fare"].mean())
    return titanic_data.drop(
        [
            "passengerid",
            "name",
            "ticket",
            "cabin",
            "sex",
            "embarked",
            "pclass",
        ],
        axis=1,
    )


torch.manual_seed(1)  # Set seed for reproducibility.


class TitanicSimpleNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(12, 12)
        self.sigmoid1 = nn.Sigmoid()
        self.linear2 = nn.Linear(12, 8)
        self.sigmoid2 = nn.Sigmoid()
        self.linear3 = nn.Linear(8, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        lin1_out = self.linear1(x)
        sigmoid_out1 = self.sigmoid1(lin1_out)
        sigmoid_out2 = self.sigmoid2(self.linear2(sigmoid_out1))
        return self.softmax(self.linear3(sigmoid_out2))


def prepare():
    RANDOM_SEED = 42
    titanic_data = get_titanic()
    print(titanic_data)

    labels = titanic_data["survived"].to_numpy()
    titanic_data = titanic_data.drop(["survived"], axis=1)
    feature_names = list(titanic_data.columns)
    data = titanic_data.to_numpy()
    # Separate training and test sets using
    train_features, test_features, train_labels, test_labels = train_test_split(
        data, labels, test_size=0.3, random_state=RANDOM_SEED, stratify=labels
    )
    train_features = np.vstack(train_features[:, :]).astype(np.float32)
    test_features = np.vstack(test_features[:, :]).astype(np.float32)
    return train_features, train_labels, test_features, test_labels, feature_names


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


def visualize_importances(
    feature_names,
    importances,
    title="Average Feature Importances",
    plot=True,
    axis_title="Features",
):
    print(title)
    feature_imp = PrettyTable(["feature_name", "importances"])
    feature_imp_dict = {}
    for i in range(len(feature_names)):
        print(feature_names[i], ": ", f"{importances[i]:.3f}")
        feature_imp.add_row([feature_names[i], importances[i]])
        feature_imp_dict[str(feature_names[i])] = importances[i]
    x_pos = np.arange(len(feature_names))
    if plot:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x_pos, importances, align="center")
        ax.set(title=title, xlabel=axis_title)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(feature_names, rotation="vertical")
        mlflow.log_figure(fig, title + ".png")
    return feature_imp, feature_imp_dict


def train(USE_PRETRAINED_MODEL=False):
    net = TitanicSimpleNNModel()
    train_features, train_labels, test_features, test_labels, feature_names = prepare()
    USE_PRETRAINED_MODEL = dict_args["use_pretrained_model"]
    if USE_PRETRAINED_MODEL:
        net.load_state_dict(torch.load("models/titanic_state_dict.pt"))
        net.eval()
        print("Model Loaded!")
    else:
        criterion = nn.CrossEntropyLoss()
        num_epochs = dict_args["max_epochs"]
        mlflow.log_param("epochs", num_epochs)
        mlflow.log_param("lr", dict_args["lr"])

        optimizer = torch.optim.Adam(net.parameters(), lr=dict_args["lr"])
        print(train_features.dtype)
        input_tensor = torch.from_numpy(train_features).type(torch.FloatTensor)
        label_tensor = torch.from_numpy(train_labels)
        for epoch in range(num_epochs):
            output = net(input_tensor)
            loss = criterion(output, label_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 50 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs} => Train Loss: {loss.item():.2f}")
                mlflow.log_metric(
                    f"Epoch {epoch + 1!s} Loss",
                    float(loss.item()),
                    step=epoch,
                )
        if not os.path.isdir("models"):
            os.makedirs("models")
            torch.save(net.state_dict(), "models/titanic_state_dict.pt")
    summary, _ = count_model_parameters(net)
    mlflow.log_text(str(summary), "model_summary.txt")
    return (
        net,
        train_features,
        train_labels,
        test_features,
        test_labels,
        feature_names,
    )


def compute_accuracy(net, features, labels, title=None):
    input_tensor = torch.from_numpy(features).type(torch.FloatTensor)
    out_probs = net(input_tensor).detach().numpy()
    out_classes = np.argmax(out_probs, axis=1)
    mlflow.log_metric(title, float(sum(out_classes == labels) / len(labels)))
    print(title, sum(out_classes == labels) / len(labels))
    return input_tensor


def feature_conductance(net, test_input_tensor):
    """
    The method takes tensor(s) of input examples (matching the forward function of the model),
    and returns the input attributions for the given input example.
    The returned values of the attribute method are the attributions,
    which match the size of the given inputs, and delta,
    which approximates the error between the approximated integral and true integral.
    This method saves the distribution of avg attributions of the trained features for the given target.
    """
    ig = IntegratedGradients(net)
    test_input_tensor.requires_grad_()
    attr, _ = ig.attribute(test_input_tensor, target=1, return_convergence_delta=True)
    attr = attr.detach().numpy()
    # To understand these attributions, we can first average them across all the inputs and print and visualize the average attribution for each feature.
    feature_imp, feature_imp_dict = visualize_importances(feature_names, np.mean(attr, axis=0))
    mlflow.log_metrics(feature_imp_dict)
    mlflow.log_text(str(feature_imp), "feature_imp_summary.txt")
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.tight_layout(pad=3)
    ax1.hist(attr[:, 1], 100)
    ax1.set(title="Distribution of Sibsp Attribution Values")

    # we can bucket the examples by the value of the sibsp feature and plot the average attribution for the feature.
    # In the plot below, the size of the dot is proportional to the number of examples with that value.

    bin_means, bin_edges, _ = stats.binned_statistic(
        test_features[:, 1], attr[:, 1], statistic="mean", bins=6
    )
    bin_count, _, _ = stats.binned_statistic(
        test_features[:, 1], attr[:, 1], statistic="count", bins=6
    )

    bin_width = bin_edges[1] - bin_edges[0]
    bin_centers = bin_edges[1:] - bin_width / 2
    ax2.scatter(bin_centers, bin_means, s=bin_count)
    ax2.set(xlabel="Average Sibsp Feature Value", ylabel="Average Attribution")
    mlflow.log_figure(fig, "Average_Sibsp_Feature_Value.png")


def layer_conductance(net, test_input_tensor):
    """
    To use Layer Conductance, we create a LayerConductance object passing in the model as well as the module (layer) whose output we would like to understand.
    In this case, we choose net.sigmoid1, the output of the first hidden layer.
    Now obtain the conductance values for all the test examples by calling attribute on the LayerConductance object.
    LayerConductance also requires a target index for networks with multiple outputs, defining the index of the output for which gradients are computed.
    Similar to feature attributions, we provide target = 1, corresponding to survival.
    LayerConductance also utilizes a baseline, but we simply use the default zero baseline as in integrated gradients.
    """

    cond = LayerConductance(net, net.sigmoid1)

    cond_vals = cond.attribute(test_input_tensor, target=1)
    cond_vals = cond_vals.detach().numpy()
    # We can begin by visualizing the average conductance for each neuron.
    neuron_names = ["neuron " + str(x) for x in range(12)]
    avg_neuron_imp, neuron_imp_dict = visualize_importances(
        neuron_names,
        np.mean(cond_vals, axis=0),
        title="Average Neuron Importances",
        axis_title="Neurons",
    )
    mlflow.log_metrics(neuron_imp_dict)
    mlflow.log_text(str(avg_neuron_imp), "neuron_imp_summary.txt")
    # We can also look at the distribution of each neuron's attributions. Below we look at the distributions for neurons 7 and 9,
    # and we can confirm that their attribution distributions are very close to 0, suggesting they are not learning substantial features.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6))
    fig.tight_layout(pad=3)
    ax1.hist(cond_vals[:, 9], 100)
    ax1.set(title="Neuron 9 Distribution")
    ax2.hist(cond_vals[:, 7], 100)
    ax2.set(title="Neuron 7 Distribution")
    mlflow.log_figure(fig, "Neurons_Distribution.png")


def neuron_conductance(net, test_input_tensor, neuron_selector=None):
    """
    We have identified that some of the neurons are not learning important features, while others are.
    Can we now understand what each of these important neurons are looking at in the input?
    For instance, are they identifying different features in the input or similar ones?

    To answer these questions, we can apply the third type of attributions available in Captum, **Neuron Attributions**.
    This allows us to understand what parts of the input contribute to activating a particular input neuron. For this example,
    we will apply Neuron Conductance, which divides the neuron's total conductance value into the contribution from each individual input feature.

    To use Neuron Conductance, we create a NeuronConductance object, analogously to Conductance,
    passing in the model as well as the module (layer) whose output we would like to understand, in this case, net.sigmoid1, as before.
    """
    neuron_selector = 0
    neuron_cond = NeuronConductance(net, net.sigmoid1)

    # We can now obtain the neuron conductance values for all the test examples by calling attribute on the NeuronConductance object.
    # Neuron Conductance requires the neuron index in the target layer for which attributions are requested as well as the target index for networks with multiple outputs,
    # similar to layer conductance. As before, we provide target = 1, corresponding to survival, and compute neuron conductance for neurons 0 and 10, the significant neurons identified above.
    # The neuron index can be provided either as a tuple or as just an integer if the layer output is 1-dimensional.

    neuron_cond_vals = neuron_cond.attribute(
        test_input_tensor, neuron_selector=neuron_selector, target=1
    )
    neuron_cond, _ = visualize_importances(
        feature_names,
        neuron_cond_vals.mean(dim=0).detach().numpy(),
        title=f"Average Feature Importances for Neuron {neuron_selector}",
    )
    mlflow.log_text(
        str(neuron_cond), "Avg_Feature_Importances_Neuron_" + str(neuron_selector) + ".txt"
    )


if __name__ == "__main__":
    parser = ArgumentParser(description="Titanic Captum Example")

    parser.add_argument(
        "--use_pretrained_model",
        default=False,
        metavar="N",
        help="Use pretrained model or train from the scratch",
    )

    parser.add_argument(
        "--max_epochs",
        type=int,
        default=100,
        metavar="N",
        help="Number of epochs to be used for training",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        metavar="LR",
        help="learning rate (default: 0.1)",
    )

    args = parser.parse_args()
    dict_args = vars(args)

    with mlflow.start_run(run_name="Titanic_Captum_mlflow"):
        net, train_features, train_labels, test_features, test_labels, feature_names = train()

        compute_accuracy(net, train_features, train_labels, title="Train Accuracy")
        test_input_tensor = compute_accuracy(net, test_features, test_labels, title="Test Accuracy")
        feature_conductance(net, test_input_tensor)
        layer_conductance(net, test_input_tensor)
        neuron_conductance(net, test_input_tensor)
        mlflow.log_param("Train Size", len(train_labels))
        mlflow.log_param("Test Size", len(test_labels))
