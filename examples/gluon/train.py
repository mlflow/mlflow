from mxnet import gluon, init
from mxnet.gluon import Trainer
from mxnet.gluon.contrib import estimator
from mxnet.gluon.data.vision import datasets, transforms
from mxnet.gluon.loss import SoftmaxCrossEntropyLoss
from mxnet.gluon.nn import HybridSequential, Dense, Flatten, MaxPool2D, Conv2D
from mxnet.metric import Accuracy

# The following import and function call are the only additions to code required
# to automatically log metrics and parameters to MLflow.
import mlflow.gluon

mlflow.gluon.autolog()

mnist_train = datasets.FashionMNIST(train=True)
X, y = mnist_train[0]

text_labels = [
    "t-shirt",
    "trouser",
    "pullover",
    "dress",
    "coat",
    "sandal",
    "shirt",
    "sneaker",
    "bag",
    "ankle boot",
]
X, y = mnist_train[0:10]

transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.13, 0.31)])
mnist_train = mnist_train.transform_first(transformer)

batch_size = 256
train_data = gluon.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=4)

mnist_valid = gluon.data.vision.FashionMNIST(train=False)
valid_data = gluon.data.DataLoader(
    mnist_valid.transform_first(transformer), batch_size=batch_size, num_workers=4
)

# Only hybrid based networks can be exported
net = HybridSequential()
net.add(
    Conv2D(channels=6, kernel_size=5, activation="relu"),
    MaxPool2D(pool_size=2, strides=2),
    Conv2D(channels=16, kernel_size=3, activation="relu"),
    MaxPool2D(pool_size=2, strides=2),
    Flatten(),
    Dense(120, activation="relu"),
    Dense(84, activation="relu"),
    Dense(10),
)
net.initialize(init=init.Xavier())
# Only after hybridization a model can be exported with architecture included
net.hybridize()

trainer = Trainer(net.collect_params(), "sgd", {"learning_rate": 0.1})

est = estimator.Estimator(
    net=net, loss=SoftmaxCrossEntropyLoss(), train_metrics=Accuracy(), trainer=trainer
)
est.fit(train_data=train_data, epochs=2, val_data=valid_data)
