import matplotlib.pyplot as plt
import pandas as pd


def to_dataframe(dataset):
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = pd.Series(dataset.target, name="target")
    return X, y


def show_image(path):
    img = plt.imread(path)
    plt.imshow(img)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
