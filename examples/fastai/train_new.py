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
import numpy as np
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec


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

    splitter = GrandparentSplitter(train_name="training", valid_name="testing")

    # Prepare DataBlock which is a generic container to quickly build Datasets and DataLoaders
    mnist = DataBlock(
        blocks=(ImageBlock(PILImage), CategoryBlock),
        get_items=get_image_files,
        splitter=splitter,
        get_y=parent_label,
    )

    # Download, untar the MNIST data set and create DataLoader from DataBlock
    data_path = untar_data(URLs.MNIST)
    image_files = get_image_files(data_path)
    print('Data located at:', data_path)

    data = mnist.dataloaders(data_path, bs=256, num_workers=0)


    input_schema = Schema([
        TensorSpec(np.dtype(np.uint8), (-1, -1, -1, 3)),
    ])
    output_schema = Schema([
        TensorSpec(np.dtype(np.float32), (-1, 10)),
    ])
    
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)
    
    # Enable auto logging
    #mlflow.fastai.autolog()

    # Create Learner model
    learn = cnn_learner(data, resnet18)

    saved_model = "/home/azureuser/.fastai/model.fastai"

    with mlflow.start_run() as run:
        # Train and fit with default or supplied command line arguments
        learn.fit_one_cycle(1, 0.1)
        learn.export(saved_model)
        model = mlflow.pyfunc.log_model("model", 
                                    registered_model_name="digits_cnn_model",
                                    data_path=saved_model, 
                                    code_path=["./fastai_model_loader.py"], 
                                    loader_module="fastai_model_loader", 
                                    signature=signature)


if __name__ == "__main__":
    main()
