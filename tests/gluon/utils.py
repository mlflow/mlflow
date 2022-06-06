import mxnet as mx
from packaging.version import Version
from mxnet.gluon.contrib.estimator import estimator
from mxnet.gluon.loss import SoftmaxCrossEntropyLoss


if Version(mx.__version__) >= Version("2.0.0"):
    from mxnet.gluon.metric import Accuracy  # pylint: disable=import-error
else:
    from mxnet.metric import Accuracy  # pylint: disable=import-error


def is_mxnet_older_than_1_6_0():
    return Version(mx.__version__) < Version("1.6.0")


def get_estimator(net, trainer):
    # `metrics` argument was split into `train_metrics` and `val_metrics` in mxnet 1.6.0:
    # https://github.com/apache/incubator-mxnet/pull/17048
    acc = Accuracy()
    loss = SoftmaxCrossEntropyLoss()
    return (
        # pylint: disable=unexpected-keyword-arg
        estimator.Estimator(net=net, loss=loss, trainer=trainer, metrics=acc)
        if is_mxnet_older_than_1_6_0()
        else estimator.Estimator(net=net, loss=loss, trainer=trainer, train_metrics=acc)
    )
