import importlib
import json
import os.path
import warnings
import numpy as np


def _label_encoder_to_json(le):
    """Returns a JSON compatible dictionary"""
    meta = {}
    for k, v in le.__dict__.items():
        if isinstance(v, np.ndarray):
            meta[k] = v.tolist()
        else:
            meta[k] = v
    return meta


def _label_encoder_from_json(doc):
    """Load the encoder back from a JSON compatible dict"""
    from lightgbm.compat import _LGBMLabelEncoder

    le = _LGBMLabelEncoder()
    meta = {}
    for k, v in doc.items():
        if k == "classes_":
            le.classes_ = np.array(v) if v is not None else None
            continue
        meta[k] = v
    le.__dict__.update(meta)
    return le


def _save_lgb_attr(model_dir, fname, attr_dict):
    with open(os.path.join(model_dir, "{}.json".format(fname)), "w") as f:
        json.dump(attr_dict, f)


def _load_lgb_attr(model_dir, fname):
    try:
        with open(os.path.join(model_dir, "{}.json".format(fname))) as f:
            attr = json.load(f)
        return attr
    except IOError:
        return None


def _save_lgb_model(lgb_model, model_path) -> None:
    import lightgbm as lgb

    model_dir = os.path.dirname(model_path)

    if not isinstance(lgb_model, lgb.Booster):
        meta = {}
        for k, v in lgb_model.__dict__.items():
            if k == "_le":
                meta["_le"] = _label_encoder_to_json(v) if v else None
                continue
            if k == "_Booster":
                continue
            if k == "_classes" and v is not None:
                meta["_classes"] = v.tolist()
                continue
            if k == "_class_map" and v:
                py_dict = {}
                for clazz, encoded in v.items():
                    py_dict[int(clazz)] = int(encoded)
                v = py_dict
            try:
                json.dumps({k: v})
                meta[k] = v
            except TypeError:
                warnings.warn(str(k) + " is not saved in Scikit-Learn meta.", UserWarning)
        _save_lgb_attr(model_dir, "scikit-learn", meta)
        lgb_model = lgb_model._Booster

    lgb_model.save_model(model_path)
    _save_lgb_attr(model_dir, "params", lgb_model.params)


def _load_lgb_model(lgb_model_class, model_path):
    import lightgbm as lgb

    module, cls = lgb_model_class.rsplit(".", maxsplit=1)
    model_dir = os.path.dirname(model_path)
    sk_attr = _load_lgb_attr(model_dir, "scikit-learn")
    bst_params = _load_lgb_attr(model_dir, "params")

    booster = lgb.Booster(model_file=model_path, params=bst_params)

    if sk_attr is None:
        warnings.warn("Loading a native LightGBM model with Scikit-Learn interface.")
        return booster

    sk_model = getattr(importlib.import_module(module), cls)()
    states = {}
    for k, v in sk_attr.items():
        if k == "_le":
            sk_model._le = _label_encoder_from_json(v)
            continue
        if k == "_classes":
            sk_model._classes = np.array(v)
            continue
        states[k] = v
    sk_model.__dict__.update(states)
    # Delete the attribute after load
    booster.set_attr(scikit_learn=None)
    sk_model._Booster = booster

    return sk_model
