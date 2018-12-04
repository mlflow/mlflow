import click

from keras_image_classifier import MLflowInceptionV3, KerasImageClassifier, MLflow_VGG16

import os


def download_input():
    import requests
    url = 'http://download.tensorflow.org/example_images/flower_photos.tgz'
    r = requests.get(url)
    with open('flower_photos.tgz', 'wb') as f:
        f.write(r.content)
    import tarfile

    with tarfile.open("flower_photos.tgz") as tar:
        tar.extractall(path="./")


@click.command(help="Trains an Keras model on flower_photos dataset."
                    "The input is expected as a directory tree with pictures for each category in a "
                    "folder named by the category."
                    "The model and its metrics are logged with mlflow.")
@click.option("--epochs", type=click.INT, default=1, help="Maximum number of epochs to evaluate.")
@click.option("--batch-size", type=click.INT, default=1,
              help="Batch size passed to the learning algo.")
@click.option("--seed", type=click.INT, default=97531, help="Seed for the random generator.")
@click.option("--training-data")
@click.option("--test-ratio", type=click.FLOAT, default=0.2)
@click.option("--pretrained-weights", type=click.STRING, default="None")
@click.option("--model-type", type=click.STRING, default="VGG16")
def run(training_data, test_ratio, epochs, batch_size, seed, pretrained_weights, model_type):
    image_files = []
    labels = []
    domain = {}

    if pretrained_weights == "None":
        pretrained_weights = None

    if training_data == "./flower_photos" and not os.path.exists(training_data):
        print("Input data not found, attempting to download the data from the web.")
        download_input()

    for (dirname, _, files) in os.walk(training_data):
        for filename in files:
            if filename.endswith("jpg"):
                image_files.append(os.path.join(dirname, filename))
                clazz = os.path.basename(dirname)
                if clazz not in domain:
                    domain[clazz] = len(domain)
                labels.append(domain[clazz])

    if model_type == "VGG16":
        classifier = MLflow_VGG16(weights=pretrained_weights)
    elif model_type == "Inception_V3":
        classifier = MLflowInceptionV3(weights=pretrained_weights)
    else:
        raise Exception("Unknown model type '{}'".format(model_type))

    classifier.train(image_files,
                     labels,
                     domain,
                     epochs=epochs,
                     test_ratio=test_ratio,
                     batch_size=batch_size,
                     seed=seed)


if __name__ == '__main__':
    run()
