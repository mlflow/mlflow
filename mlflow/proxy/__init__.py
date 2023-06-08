from pydantic import BaseModel
import transformers


class Payload(BaseModel):  # TODO: we'll need different payload types for different endpoints
    text: str


def translator_en_to_fr(payload: Payload):  # TESTING ONLY
    translator = transformers.pipeline(
        task="translation_en_to_fr",
        model=transformers.T5ForConditionalGeneration.from_pretrained("t5-large"),
        tokenizer=transformers.T5TokenizerFast.from_pretrained("t5-large", model_max_length=200),
    )
    return translator(payload.text)


def translator_en_to_de(payload: Payload):  # TESTING ONLY
    translator = transformers.pipeline(
        task="translation_en_to_de",
        model=transformers.T5ForConditionalGeneration.from_pretrained("t5-large"),
        tokenizer=transformers.T5TokenizerFast.from_pretrained("t5-large", model_max_length=200),
    )
    return translator(payload.text)


def classifier(payload: Payload):  # TESTING ONLY
    classifier = transformers.pipeline(
        task="text-classification",
        tokenizer=transformers.AutoTokenizer.from_pretrained(
            "distilbert-base-uncased-finetuned-sst-2-english"
        ),
        model=transformers.AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased-finetuned-sst-2-english"
        ),
    )
    return classifier(payload.text)
