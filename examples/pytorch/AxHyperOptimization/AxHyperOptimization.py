import argparse
import os
import shutil
import statistics
import tempfile

import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from ax.service.ax_client import AxClient
from torchvision import transforms


def create_dataset():
    """
     This function downloads the CIFAR10 dataset from torchvision.datasets performs the necessary transforms and returns the dataset
     in the form trainloader and testloader
    :return: trainloader Type:dataloader , testloader Type:dataloader
    """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    return trainloader, testloader


class Net(nn.Module):
    def __init__(self):
        """
        Initializes the neural network (CNN) model.

        """
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        """
        The forward takes input and returns a transformed output.
        :param x: input data to the model Type:tensor
        :return: output of the model Type:tensor
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


criterion = nn.CrossEntropyLoss()


def configure_optimizer():
    """
    This function initializes the SGD optimizer with a learning rate and momentum value.
    :return: optimizer
    """
    optimizer = optim.SGD(net.parameters(), lr=0.011, momentum=0.9)
    return optimizer


def train(max_epochs):
    """
    The training function of the model.
    :return: net_state_dict - state dict of the model , training_loss -loss associated with training process
    """
    train_loss = {"loss": []}
    trainloader, _ = create_dataset()
    for epoch in range(max_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            # get the inputs; data is a list of [inputs, labels]
            # zero the parameter gradients
            configure_optimizer().zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            configure_optimizer().step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))
                train_loss["loss"].append(running_loss / 2000)
                running_loss = 0.0

    return net.state_dict(), train_loss


def evaluate(net, net_state_dict):
    """
    The function is used to test the model.
    :param net: model object Type:nn.module
    :param net_state_dict: model state_dict at the end of training.
    :return: test_accuracy Type:float
    """
    class_correct = list(0.0 for i in range(10))
    class_total = list(0.0 for i in range(10))
    _, testloader = create_dataset()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    lis = []
    for i in range(10):
        test_acc = 100 * class_correct[i] / class_total[i]
        lis.append(test_acc)
    test_accuracy = statistics.mean(lis)
    return test_accuracy


def train_evaluate(parameterization=None, max_epochs=0):
    """
    The function combines both training phase and testing phase and provides the testing results as output.
    :param parameterization: parameter dictionary
    :return: test_accuracy
    """
    net = Net()
    net_state_dict, training_loss = train(max_epochs)
    mlflow.pytorch.log_model(net, "models")
    return evaluate(net, net_state_dict), training_loss


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max_epochs",
        default=2,
        help="Describes the number of times a neural network has to be trained",
    )
    parser.add_argument(
        "--total_trials",
        default=3,
        help="It indicated number of trials to be run for the optimization experiment",
    )

    args = parser.parse_args()

    if not mlflow.active_run():
        mlflow.start_run(run_name="ParentRun")
        auto_end_run = True
    else:
        auto_end_run = False
    net = Net()

    tempdir = tempfile.mkdtemp()
    try:
        summary_file = os.path.join(tempdir, "model_summary.txt")
        with open(summary_file, "w") as f:
            for name, module in net.named_modules():
                f.write(str(module))

        mlflow.log_artifact(summary_file, "summary")
    finally:
        shutil.rmtree(tempdir)

    opt = configure_optimizer()
    baseline_accuracy, loss = train_evaluate(
        {"lr": 0.011, "momentum": 0.9}, max_epochs=int(args.max_epochs)
    )
    mlflow.log_metric("baseline_accuracy", baseline_accuracy)
    for k, v in loss.items():
        for loss in v:
            mlflow.log_metric("loss", round(loss, 2))
    opt_param = opt.state_dict()["param_groups"]

    for param in opt_param:
        for k, v in param.items():
            mlflow.log_param(k, v)
    ax_client = AxClient()
    ax_client.create_experiment(
        parameters=[
            {
                "name": "lr",
                "type": "range",
                "bounds": [1e-6, 0.4],
                "log_scale": True,
            },
            {
                "name": "momentum",
                "type": "range",
                "bounds": [0.0, 1.0],
            },
        ],
        objective_name="test_accuracy",
    )

total_trials = int(args.total_trials)

for i in range(total_trials):
    with mlflow.start_run(nested=True, run_name="Trial " + str(i)) as child_run:

        parameters, trial_index = ax_client.get_next_trial()

        # log params to MLFlow
        for param_name, value in parameters.items():
            mlflow.log_param(param_name, value)

        # evaluate params
        test_accuracy, loss = train_evaluate(parameters)

        # log metric to MLFlow
        mlflow.log_metric("test_accuracy", test_accuracy)
        for k, v in loss.items():
            for loss in v:
                mlflow.log_metric("loss", round(loss, 2))

        ax_client.complete_trial(trial_index=trial_index, raw_data=test_accuracy)

best_parameters, metrics = ax_client.get_best_parameters()

for param_name, value in best_parameters.items():
    mlflow.log_param("optimum " + param_name, value)

mlflow.end_run()
