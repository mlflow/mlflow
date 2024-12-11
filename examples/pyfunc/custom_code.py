flower_classes = ["setosa", "versicolor", "virginica"]


def iris_classes(preds):
    return [flower_classes[x] for x in preds]
