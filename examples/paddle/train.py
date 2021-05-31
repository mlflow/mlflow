import mlflow.paddle
import paddle
from paddle.nn import Linear
import paddle.nn.functional as F
import numpy as np
import os
import random
from pathlib import Path
import wget

def load_data():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'

    datafile = Path('housing.data')
    if not datafile.exists():
        datafile = wget.download(url)

    data = np.fromfile(datafile, sep=' ', dtype=np.float32)
    feature_names = [ 'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', \
                      'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV' ]
    feature_num = len(feature_names)
    data = data.reshape([data.shape[0] // feature_num, feature_num])
    ratio = 0.8
    offset = int(data.shape[0] * ratio)
    training_data = data[:offset]

    maximums, minimums, avgs = training_data.max(axis=0), training_data.min(axis=0), \
                                 training_data.sum(axis=0) / training_data.shape[0]
    
    global max_values
    global min_values
    global avg_values
    max_values = maximums
    min_values = minimums
    avg_values = avgs

    for i in range(feature_num):
        data[:, i] = (data[:, i] - avgs[i]) / (maximums[i] - minimums[i])

    training_data = data[:offset]
    test_data = data[offset:]
    return training_data, test_data

class Regressor(paddle.nn.Layer):

    def __init__(self):
        super(Regressor, self).__init__()
        
        self.fc = Linear(in_features=13, out_features=1)

    @paddle.jit.to_static
    def forward(self, inputs):
        x = self.fc(inputs)
        return x


if __name__ == "__main__":
    model = Regressor()
    model.train()
    training_data, test_data = load_data()

    opt = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters()) 

    EPOCH_NUM = 10  
    BATCH_SIZE = 10 

    for epoch_id in range(EPOCH_NUM):
        np.random.shuffle(training_data)
        mini_batches = [training_data[k:k+BATCH_SIZE] for k in range(0, len(training_data), BATCH_SIZE)]
        for iter_id, mini_batch in enumerate(mini_batches):
            x = np.array(mini_batch[:, :-1]) 
            y = np.array(mini_batch[:, -1:]) 
            house_features = paddle.to_tensor(x)
            prices = paddle.to_tensor(y)
            
            predicts = model(house_features)
            
            loss = F.square_error_cost(predicts, label=prices)
            avg_loss = paddle.mean(loss)
            if iter_id%20==0:
                print("epoch: {}, iter: {}, loss is: {}".format(epoch_id, iter_id, avg_loss.numpy()))
            
            avg_loss.backward()
            opt.step()
            opt.clear_grad()

    mlflow.log_param('learning_rate', 0.01)
    mlflow.paddle.log_model(model, "model")
    sk_path_dir = './test-out'
    mlflow.paddle.save_model(model, sk_path_dir)
    print("Model saved in run %s" % mlflow.active_run().info.run_uuid)

    # load model
    pd_model, params = mlflow.paddle.load_model('test-out')
    np_test_data = np.array(test_data).astype('float32')
    print(pd_model(np_test_data[:, :-1]))