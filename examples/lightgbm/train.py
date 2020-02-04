import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb

import mlflow.lightgbm

X = np.random.random((10, 10))
y = np.random.randint(2, size=10)
train_set = lgb.Dataset(X, y)

mlflow.lightgbm.autolog()

for _ in range(11):
    model = lgb.train({}, train_set, num_boost_round=1, verbose_eval=0)
    print(plt.get_fignums())
