"""
PEFT (Parameter-Efficient Fine-Tuning) is a library for efficiently adapting large pretrained
models without fine-tuning all of model parameters but only a small number of (extra) parameters.
Users can define a PEFT model that wraps a Transformer model to apply a thin adapter layer on
top of the base model. The PEFT model provides almost the same APIs as the original model such
as from_pretrained(), save_pretrained().
"""
_PEFT_ADAPTOR_DIR_NAME = "peft"


def is_peft_model(model) -> bool:
    try:
        from peft import PeftModel
    except ImportError:
        return False

    return isinstance(model, PeftModel)


def get_peft_base_model(model):
    """Extract the base model from a PEFT model."""
    peft_config = model.peft_config.get(model.active_adapter) if model.peft_config else None

    # PEFT usually wraps the base model with two additional classes, one is PeftModel class
    # and the other is the adaptor specific class, like LoraModel class, so the class hierarchy
    # looks like PeftModel -> LoraModel -> BaseModel
    # However, when the PEFT config is the one for "prompt learning", there is not adaptor class
    # and the PeftModel class directly wraps the base model.
    if peft_config and not peft_config.is_prompt_learning:
        return model.base_model.model

    return model.base_model


def get_model_with_peft_adapter(base_model, peft_adapter_path):
    """
    Apply the PEFT adapter to the base model to create a PEFT model.

    NB: The alternative way to load PEFT adapter is to use load_adapter API like
    `base_model.load_adapter(peft_adapter_path)`, as it injects the adapter weights
    into the model in-place hence reducing the memory footprint. However, doing so
    returns the base model class and not the PEFT model, loosing some properties
    such as peft_config. This is not preferable because load_model API should
    return the exact same object that was saved. Hence we construct the PEFT model
    instead of in-place injection, for consistency over the memory saving which
    should be small in most cases.
    """
    from peft import PeftModel

    return PeftModel.from_pretrained(base_model, peft_adapter_path)
