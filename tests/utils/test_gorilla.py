import pytest
from mlflow.utils import gorilla


class Delegator:
    def __init__(self, delegated_fn):
        self.delegated_fn = delegated_fn

    def __get__(self, instance, owner):
        return self.delegated_fn


def delegate(delegated_fn):
    return lambda fn: Delegator(delegated_fn)


def gen_class_A_B():
    class A:
        def f1(self):
            pass

        def f2(self):
            pass

        def delegated_f3(self):
            pass

        @delegate(delegated_f3)
        def f3(self):
            pass

    class B(A):
        def f1(self):
            pass

    return A, B


@pytest.fixture
def gorilla_setting():
    return gorilla.Settings(allow_hit=True, store_hit=True)


def test_basic_patch_for_class(gorilla_setting):
    A, B = gen_class_A_B()

    original_A_f1 = A.f1
    original_A_f2 = A.f2
    original_B_f1 = B.f1

    def patched_A_f1(self):  # pylint: disable=unused-argument
        pass

    def patched_A_f2(self):  # pylint: disable=unused-argument
        pass

    def patched_B_f1(self):  # pylint: disable=unused-argument
        pass

    patch_A_f1 = gorilla.Patch(A, "f1", patched_A_f1, gorilla_setting)
    patch_A_f2 = gorilla.Patch(A, "f2", patched_A_f2, gorilla_setting)
    patch_B_f1 = gorilla.Patch(B, "f1", patched_B_f1, gorilla_setting)

    assert gorilla.get_original_attribute(A, "f1") is original_A_f1
    assert gorilla.get_original_attribute(B, "f1") is original_B_f1
    assert gorilla.get_original_attribute(B, "f2") is original_A_f2

    gorilla.apply(patch_A_f1)
    assert A.f1 is patched_A_f1
    assert gorilla.get_original_attribute(A, "f1") is original_A_f1
    assert gorilla.get_original_attribute(B, "f1") is original_B_f1

    gorilla.apply(patch_B_f1)
    assert A.f1 is patched_A_f1
    assert B.f1 is patched_B_f1
    assert gorilla.get_original_attribute(A, "f1") is original_A_f1
    assert gorilla.get_original_attribute(B, "f1") is original_B_f1

    gorilla.apply(patch_A_f2)
    assert A.f2 is patched_A_f2
    assert B.f2 is patched_A_f2
    assert gorilla.get_original_attribute(A, "f2") is original_A_f2
    assert gorilla.get_original_attribute(B, "f2") is original_A_f2

    gorilla.revert(patch_A_f2)
    assert A.f2 is original_A_f2
    assert B.f2 is original_A_f2
    assert gorilla.get_original_attribute(A, "f2") == original_A_f2
    assert gorilla.get_original_attribute(B, "f2") == original_A_f2

    gorilla.revert(patch_B_f1)
    assert A.f1 is patched_A_f1
    assert B.f1 is original_B_f1
    assert gorilla.get_original_attribute(A, "f1") == original_A_f1
    assert gorilla.get_original_attribute(B, "f1") == original_B_f1

    gorilla.revert(patch_A_f1)
    assert A.f1 is original_A_f1
    assert B.f1 is original_B_f1
    assert gorilla.get_original_attribute(A, "f1") == original_A_f1
    assert gorilla.get_original_attribute(B, "f1") == original_B_f1


def test_patch_for_descriptor(gorilla_setting):
    A, _ = gen_class_A_B()

    original_A_f3_raw = object.__getattribute__(A, "f3")

    def patched_A_f3(self):  # pylint: disable=unused-argument
        pass

    patch_A_f3 = gorilla.Patch(A, "f3", patched_A_f3, gorilla_setting)

    assert gorilla.get_original_attribute(A, "f3") is A.delegated_f3
    assert (
        gorilla.get_original_attribute(A, "f3", bypass_descriptor_protocol=True)
        is original_A_f3_raw
    )

    gorilla.apply(patch_A_f3)
    assert A.f3 is patched_A_f3
    assert gorilla.get_original_attribute(A, "f3") is A.delegated_f3
    assert (
        gorilla.get_original_attribute(A, "f3", bypass_descriptor_protocol=True)
        is original_A_f3_raw
    )

    gorilla.revert(patch_A_f3)
    assert A.f3 is A.delegated_f3
    assert gorilla.get_original_attribute(A, "f3") is A.delegated_f3
    assert (
        gorilla.get_original_attribute(A, "f3", bypass_descriptor_protocol=True)
        is original_A_f3_raw
    )

    # test patch a descriptor
    @delegate(patched_A_f3)
    def new_patched_A_f3(self):  # pylint: disable=unused-argument
        pass

    new_patch_A_f3 = gorilla.Patch(A, "f3", new_patched_A_f3, gorilla_setting)
    gorilla.apply(new_patch_A_f3)
    assert A.f3 is patched_A_f3
    assert object.__getattribute__(A, "f3") is new_patched_A_f3
    assert gorilla.get_original_attribute(A, "f3") is A.delegated_f3
    assert (
        gorilla.get_original_attribute(A, "f3", bypass_descriptor_protocol=True)
        is original_A_f3_raw
    )


@pytest.mark.parametrize("store_hit", [True, False])
def test_patch_on_inherit_method(store_hit):
    A, B = gen_class_A_B()

    original_A_f2 = A.f2

    def patched_B_f2(self):  # pylint: disable=unused-argument
        pass

    gorilla_setting = gorilla.Settings(allow_hit=True, store_hit=store_hit)
    patch_B_f2 = gorilla.Patch(B, "f2", patched_B_f2, gorilla_setting)
    gorilla.apply(patch_B_f2)

    assert B.f2 is patched_B_f2

    assert gorilla.get_original_attribute(B, "f2") is original_A_f2

    gorilla.revert(patch_B_f2)
    assert B.f2 is original_A_f2
    assert gorilla.get_original_attribute(B, "f2") is original_A_f2
    assert "f2" not in B.__dict__  # assert no side effect after reverting


@pytest.mark.parametrize("store_hit", [True, False])
def test_patch_on_attribute_not_exist(store_hit):
    A, _ = gen_class_A_B()

    def patched_fx(self):  # pylint: disable=unused-argument
        return 101

    gorilla_setting = gorilla.Settings(allow_hit=True, store_hit=store_hit)
    fx_patch = gorilla.Patch(A, "fx", patched_fx, gorilla_setting)
    gorilla.apply(fx_patch)
    a1 = A()
    assert a1.fx() == 101
    gorilla.revert(fx_patch)
    assert not hasattr(A, "fx")
