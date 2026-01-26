from custom_model.transitive_test.transitive_dependency import some_function

from mlflow.pyfunc import PythonModel


class ModelWithTransitiveDependency(PythonModel):
    def predict(self, context, model_input, params=None):
        result = some_function()
        return [result] * len(model_input)
