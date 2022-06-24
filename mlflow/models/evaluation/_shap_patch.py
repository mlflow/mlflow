import shap
import numpy as np
import pickle
from shap._serializable import Serializer, Deserializer, Serializable


class _PatchedKernelExplainer(shap.KernelExplainer):
    @staticmethod
    def not_equal(i, j):
        # `shap.KernelExplainer.not_equal` method fails on some special types such as
        # timestamp, this breaks the kernel explainer routine.
        # `PatchedKernelExplainer` fixes this issue.
        # See https://github.com/slundberg/shap/pull/2586
        number_types = (int, float, np.number)
        if isinstance(i, number_types) and isinstance(j, number_types):
            return 0 if np.isclose(i, j, equal_nan=True) else 1
        else:
            return 0 if i == j else 1

    def save(self, out_file, model_saver=None, masker_saver=None):
        """
        This patched `save` method fix `KernelExplainer.save`.
        Issues in original `KernelExplainer.save`:
         - It saves model by calling model.save, but shap.utils._legacy.Model has no save method
         - It tries to save "masker", but there's no "masker" in KernelExplainer
         - It does not save "KernelExplainer.data" attribute, the attribute is required when
           loading back
        Note: `model_saver` and `masker_saver` are meaningless argument for `KernelExplainer.save`,
        the model in "KernelExplainer" is an instance of `shap.utils._legacy.Model`
        (it wraps the predict function), we can only use pickle to dump it.
        and no `masker` for KernelExplainer so `masker_saver` is meaningless.
        but I preserve the 2 argument for overridden API compatibility.
        """
        pickle.dump(type(self), out_file)
        with Serializer(out_file, "shap.Explainer", version=0) as s:
            s.save("model", self.model)
            s.save("link", self.link)
            s.save("data", self.data)

    @classmethod
    def load(cls, in_file, model_loader=None, masker_loader=None, instantiate=True):
        """
        This patched `load` method fix `KernelExplainer.load`.
        Issues in original KernelExplainer.load:
         - Use mismatched model loader to load model
         - Try to load non-existent "masker" attribute
         - Does not load "data" attribute and then cause calling " KernelExplainer"
           constructor lack of "data" argument.
        Note: `model_loader` and `masker_loader` are meaningless argument for
        `KernelExplainer.save`, because the `model` object is saved by pickle dump,
        we must use pickle load to load it.
        and no `masker` for KernelExplainer so `masker_loader` is meaningless.
        but I preserve the 2 argument for overridden API compatibility.
        """
        if instantiate:
            return cls._instantiated_load(in_file, model_loader=None, masker_loader=None)

        kwargs = Serializable.load(in_file, instantiate=False)
        with Deserializer(in_file, "shap.Explainer", min_version=0, max_version=0) as s:
            kwargs["model"] = s.load("model")
            kwargs["link"] = s.load("link")
            kwargs["data"] = s.load("data")
        return kwargs
