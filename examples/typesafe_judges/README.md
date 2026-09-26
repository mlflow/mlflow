# Evaluate support answers with TypeSafe judges

This example evaluates two precomputed support answers with TypeSafe System One
models through MLflow's `make_judge()` API.

It demonstrates the two TypeSafe output shapes currently supported by MLflow
judges:

- `answer_helpfulness`: a `bool` judge backed by a TypeSafe Noul question.
- `request_category`: a finite `Literal` judge backed by a TypeSafe Choice question.

For a `bool` judge, write a yes-or-no question and explicitly instruct the model
to answer yes or no.

Set `TYPESAFE_API_KEY` to call TypeSafe directly:

```bash
export TYPESAFE_API_KEY=...
python evaluate.py
```

The example uses `model="typesafe:/jev-latest"`. Use a versioned model ID when
you need reproducible evaluations.

MLflow records TypeSafe probability data in assessment metadata. TypeSafe judges
do not return generated text rationales. AI Gateway TypeSafe passthrough
endpoints are raw System One endpoints and are not used as `gateway:/` models for
`make_judge()`.
