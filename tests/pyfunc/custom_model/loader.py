import pickle

from custom_model.mod1 import mod2

__all__ = ["mod2"]


def _load_pyfunc(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")
