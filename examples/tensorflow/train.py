# tensorflow 2.x core api
import tensorflow as tf
from sklearn.datasets import fetch_california_housing

import mlflow
from mlflow.models import infer_signature


class Normalize(tf.Module):
    """Data Normalization class"""

    def __init__(self, x):
        # Initialize the mean and standard deviation for normalization
        self.mean = tf.math.reduce_mean(x, axis=0)
        self.std = tf.math.reduce_std(x, axis=0)

    def norm(self, x):
        return (x - self.mean) / self.std

    def unnorm(self, x):
        return (x * self.std) + self.mean


class LinearRegression(tf.Module):
    """Linear Regression model class"""

    def __init__(self):
        self.built = False

    @tf.function
    def __call__(self, x):
        # Initialize the model parameters on the first call
        if not self.built:
            # Randomly generate the weight vector and bias term
            rand_w = tf.random.uniform(shape=[x.shape[-1], 1])
            rand_b = tf.random.uniform(shape=[])
            self.w = tf.Variable(rand_w)
            self.b = tf.Variable(rand_b)
            self.built = True
        y = tf.add(tf.matmul(x, self.w), self.b)
        return tf.squeeze(y, axis=1)


class ExportModule(tf.Module):
    """Exporting TF model"""

    def __init__(self, model, norm_x, norm_y):
        # Initialize pre and postprocessing functions
        self.model = model
        self.norm_x = norm_x
        self.norm_y = norm_y

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32)])
    def __call__(self, x):
        # Run the ExportModule for new data points
        x = self.norm_x.norm(x)
        y = self.model(x)
        y = self.norm_y.unnorm(y)
        return y


def mse_loss(y_pred, y):
    """Calculating Mean Square Error Loss function"""
    return tf.reduce_mean(tf.square(y_pred - y))


if __name__ == "__main__":
    # Set a random seed for reproducible results
    tf.random.set_seed(42)

    # Load dataset
    dataset = fetch_california_housing(as_frame=True)["frame"]
    # Drop missing values
    dataset = dataset.dropna()
    # using only 1500
    dataset = dataset[:1500]
    dataset_tf = tf.convert_to_tensor(dataset, dtype=tf.float32)

    # Split dataset into train and test
    dataset_shuffled = tf.random.shuffle(dataset_tf, seed=42)
    train_data = dataset_shuffled[100:]
    test_data = dataset_shuffled[:100]
    x_train = train_data[:, :-1]
    y_train = train_data[:, -1]
    x_test = test_data[:, :-1]
    y_test = test_data[:, -1]
    # Data normalization
    norm_x = Normalize(x_train)
    norm_y = Normalize(y_train)
    x_train_norm = norm_x.norm(x_train)
    y_train_norm = norm_y.norm(y_train)
    x_test_norm = norm_x.norm(x_test)
    y_test_norm = norm_y.norm(y_test)

    with mlflow.start_run():
        # Initialize linear regression model
        lin_reg = LinearRegression()

        # Use mini batches for memory efficiency and faster convergence
        batch_size = 32
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train_norm, y_train_norm))
        train_dataset = train_dataset.shuffle(buffer_size=x_train.shape[0]).batch(batch_size)
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test_norm, y_test_norm))
        test_dataset = test_dataset.shuffle(buffer_size=x_test.shape[0]).batch(batch_size)

        # Set training parameters
        epochs = 100
        learning_rate = 0.01
        train_losses = []
        test_losses = []

        # Format training loop
        for epoch in range(epochs):
            batch_losses_train = []
            batch_losses_test = []

            # Iterate through the training data
            for x_batch, y_batch in train_dataset:
                with tf.GradientTape() as tape:
                    y_pred_batch = lin_reg(x_batch)
                    batch_loss = mse_loss(y_pred_batch, y_batch)
                # Update parameters with respect to the gradient calculations
                grads = tape.gradient(batch_loss, lin_reg.variables)
                for g, v in zip(grads, lin_reg.variables):
                    v.assign_sub(learning_rate * g)
                # Keep track of batch-level training performance
                batch_losses_train.append(batch_loss)

            # Iterate through the testing data
            for x_batch, y_batch in test_dataset:
                y_pred_batch = lin_reg(x_batch)
                batch_loss = mse_loss(y_pred_batch, y_batch)
                # Keep track of batch-level testing performance
                batch_losses_test.append(batch_loss)

            # Keep track of epoch-level model performance
            train_loss = tf.reduce_mean(batch_losses_train)
            test_loss = tf.reduce_mean(batch_losses_test)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            if epoch % 10 == 0:
                mlflow.log_metric(key="train_losses", value=train_loss, step=epoch)
                mlflow.log_metric(key="test_losses", value=test_loss, step=epoch)
                print(f"Mean squared error for step {epoch}: {train_loss.numpy():0.3f}")

        # Log the parameters
        mlflow.log_params(
            {
                "epochs": epochs,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
            }
        )
        # Log the final metrics
        mlflow.log_metrics(
            {
                "final_train_loss": train_loss.numpy(),
                "final_test_loss": test_loss.numpy(),
            }
        )
        print(f"\nFinal train loss: {train_loss:0.3f}")
        print(f"Final test loss: {test_loss:0.3f}")

        # Export the tensorflow model
        lin_reg_export = ExportModule(model=lin_reg, norm_x=norm_x, norm_y=norm_y)

        # Infer model signature
        predictions = lin_reg_export(x_test)
        signature = infer_signature(x_test.numpy(), predictions.numpy())

        mlflow.tensorflow.log_model(lin_reg_export, name="model", signature=signature)
