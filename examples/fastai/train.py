#
# This short example is based on the fastai GitHub Repository of vision examples
# https://github.com/fastai/fastai/blob/master/nbs/examples/mnist_blocks.py
# Modified here to show mlflow.fastai.autolog() capabilities
#
import argparse
import mlflow.fastai
from fastai.vision.all import (
    CategoryBlock,
    DataBlock,
    GrandparentSplitter,
    ImageBlock,
    PILImage,
    URLs,
)
from fastai.vision.all import cnn_learner, get_image_files, parent_label, resnet18, untar_data


def parse_args():
    parser = argparse.ArgumentParser(description="Fastai example")
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="learning rate to update step size at each step (default: 0.01)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="number of epochs (default: 5). Note it takes about 1 min per epoch",
    )
    return parser.parse_args()


def main():
    # Parse command-line arguments
    args = parse_args()

    # Split data between training and testing
    splitter = GrandparentSplitter(train_name="training", valid_name="testing")

    # Prepare DataBlock which is a generic container to quickly build Datasets and DataLoaders
    mnist = DataBlock(
        blocks=(ImageBlock(PILImage), CategoryBlock),
        get_items=get_image_files,
        splitter=splitter,
        get_y=parent_label,
    )

    # Download, untar the MNIST data set and create DataLoader from DataBlock
    data = mnist.dataloaders(untar_data(URLs.MNIST), bs=256, num_workers=0)

    # Enable auto logging
    mlflow.fastai.autolog()

    # Create Learner model
    learn = cnn_learner(data, resnet18)

    # Start MLflow session
    with mlflow.start_run():
        # Train and fit with default or supplied command line arguments
        learn.fit_one_cycle(args.epochs, args.lr)
        mlflow.fastai.log_model(learn, "fastai")


if __name__ == "__main__":
    main()
