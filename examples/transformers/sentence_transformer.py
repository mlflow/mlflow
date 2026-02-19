import torch
from transformers import BertModel, BertTokenizerFast, pipeline

import mlflow

sentence_transformers_architecture = "sentence-transformers/all-MiniLM-L12-v2"
task = "feature-extraction"

model = BertModel.from_pretrained(sentence_transformers_architecture)
tokenizer = BertTokenizerFast.from_pretrained(sentence_transformers_architecture)

sentence_transformer_pipeline = pipeline(task=task, model=model, tokenizer=tokenizer)

with mlflow.start_run():
    model_info = mlflow.transformers.log_model(
        transformers_model=sentence_transformer_pipeline,
        name="sentence_transformer",
        framework="pt",
        torch_dtype=torch.bfloat16,
    )

loaded = mlflow.transformers.load_model(model_info.model_uri, return_type="components")


def pool_and_normalize_encodings(input_sentences, model, tokenizer, **kwargs):
    def pool(model_output, attention_mask):
        embeddings = model_output[0]
        expanded_mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        return torch.sum(embeddings * expanded_mask, 1) / torch.clamp(
            expanded_mask.sum(1), min=1e-9
        )

    encoded = tokenizer(
        input_sentences,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        model_output = model(**encoded)

    pooled = pool(model_output, encoded["attention_mask"])
    return torch.nn.functional.normalize(pooled, p=2, dim=1)


sentences = [
    "He said that he's sinking all of his investment budget into coconuts.",
    "No matter how deep you dig, there's going to be a point when it just gets too hot.",
    "She said that there isn't a noticeable difference between a 10 year and a 15 year whisky.",
]

encoded_sentences = pool_and_normalize_encodings(sentences, **loaded)

print(encoded_sentences)
