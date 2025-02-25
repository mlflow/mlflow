# Copyright 2020 The HuggingFace Evaluate Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" ROUGE metric from Google Research github repo. """

# The dependencies in https://github.com/google-research/google-research/blob/master/rouge/requirements.txt
import absl  # Here to have a nice missing dependency error message early on
import datasets
import nltk  # Here to have a nice missing dependency error message early on
import numpy  # Here to have a nice missing dependency error message early on
import six  # Here to have a nice missing dependency error message early on
from rouge_score import rouge_scorer, scoring

import evaluate


_CITATION = """\
@inproceedings{lin-2004-rouge,
    title = "{ROUGE}: A Package for Automatic Evaluation of Summaries",
    author = "Lin, Chin-Yew",
    booktitle = "Text Summarization Branches Out",
    month = jul,
    year = "2004",
    address = "Barcelona, Spain",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/W04-1013",
    pages = "74--81",
}
"""

_DESCRIPTION = """\
ROUGE, or Recall-Oriented Understudy for Gisting Evaluation, is a set of metrics and a software package used for
evaluating automatic summarization and machine translation software in natural language processing.
The metrics compare an automatically produced summary or translation against a reference or a set of references (human-produced) summary or translation.

Note that ROUGE is case insensitive, meaning that upper case letters are treated the same way as lower case letters.

This metrics is a wrapper around Google Research reimplementation of ROUGE:
https://github.com/google-research/google-research/tree/master/rouge
"""

_KWARGS_DESCRIPTION = """
Calculates average rouge scores for a list of hypotheses and references
Args:
    predictions: list of predictions to score. Each prediction
        should be a string with tokens separated by spaces.
    references: list of reference for each prediction. Each
        reference should be a string with tokens separated by spaces.
    rouge_types: A list of rouge types to calculate.
        Valid names:
        `"rouge{n}"` (e.g. `"rouge1"`, `"rouge2"`) where: {n} is the n-gram based scoring,
        `"rougeL"`: Longest common subsequence based scoring.
        `"rougeLsum"`: rougeLsum splits text using `"\n"`.
        See details in https://github.com/huggingface/datasets/issues/617
    use_stemmer: Bool indicating whether Porter stemmer should be used to strip word suffixes.
    use_aggregator: Return aggregates if this is set to True
Returns:
    rouge1: rouge_1 (f1),
    rouge2: rouge_2 (f1),
    rougeL: rouge_l (f1),
    rougeLsum: rouge_lsum (f1)
Examples:

    >>> rouge = evaluate.load('rouge')
    >>> predictions = ["hello there", "general kenobi"]
    >>> references = ["hello there", "general kenobi"]
    >>> results = rouge.compute(predictions=predictions, references=references)
    >>> print(results)
    {'rouge1': 1.0, 'rouge2': 1.0, 'rougeL': 1.0, 'rougeLsum': 1.0}
"""


class Tokenizer:
    """Helper class to wrap a callable into a class with a `tokenize` method as used by rouge-score."""

    def __init__(self, tokenizer_func):
        self.tokenizer_func = tokenizer_func

    def tokenize(self, text):
        return self.tokenizer_func(text)


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Rouge(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=[
                datasets.Features(
                    {
                        "predictions": datasets.Value("string", id="sequence"),
                        "references": datasets.Sequence(datasets.Value("string", id="sequence")),
                    }
                ),
                datasets.Features(
                    {
                        "predictions": datasets.Value("string", id="sequence"),
                        "references": datasets.Value("string", id="sequence"),
                    }
                ),
            ],
            codebase_urls=["https://github.com/google-research/google-research/tree/master/rouge"],
            reference_urls=[
                "https://en.wikipedia.org/wiki/ROUGE_(metric)",
                "https://github.com/google-research/google-research/tree/master/rouge",
            ],
        )

    def _compute(
        self, predictions, references, rouge_types=None, use_aggregator=True, use_stemmer=False, tokenizer=None
    ):
        if rouge_types is None:
            rouge_types = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

        multi_ref = isinstance(references[0], list)

        if tokenizer is not None:
            tokenizer = Tokenizer(tokenizer)

        scorer = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=use_stemmer, tokenizer=tokenizer)
        if use_aggregator:
            aggregator = scoring.BootstrapAggregator()
        else:
            scores = []

        for ref, pred in zip(references, predictions):
            if multi_ref:
                score = scorer.score_multi(ref, pred)
            else:
                score = scorer.score(ref, pred)
            if use_aggregator:
                aggregator.add_scores(score)
            else:
                scores.append(score)

        if use_aggregator:
            result = aggregator.aggregate()
            for key in result:
                result[key] = result[key].mid.fmeasure

        else:
            result = {}
            for key in scores[0]:
                result[key] = list(score[key].fmeasure for score in scores)

        return result
