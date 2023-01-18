import mlflow


class WrappedRecipeEstimator(object):
    def __init__(self, model, post_predict_fn):
        super().__init__()
        self._model = model
        self.post_predict_fn = post_predict_fn
        self.classes_ = model.classes_

    def predict(self, *args, **kwargs):
        # print("Predict_step", self.post_predict_fn(self._model.predict(*args, **kwargs)))
        return self.post_predict_fn(self._model.predict(*args, **kwargs))

    def predict_proba(self, *args, **kwargs):
        return self._model.predict_proba(*args, **kwargs)

    def __getattr__(self, attr):
        # see if this object has attr
        # NOTE do not use hasattr, it goes into
        # infinite recurrsion
        # print("directorrdasdasdrrr", dir(self))
        # print("current attr", attr)
        # print("current self", self)
        if attr in dir(self):
            # this object has it
            return object.__getattr__(self, attr)
        # proxy to the wrapped object
        return object.__getattr__(self._model, attr)

    # def __getattribute__(self, name):
    #     print("directorrrrr", self.__dict__)
    #     if attr in self.__dict__:
    #         return object.__getattribute__(self, name)

    #     print("self._model", self._model)
    #     return object.__getattribute__(self._model, name)
