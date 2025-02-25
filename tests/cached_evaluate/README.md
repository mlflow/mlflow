# Cached evaluate

`evaluate.load` is highly unstable and often fails due to a network error or huggingface hub being down.
As a workaround, we fallback to loading from this directory while testing if the remote loading fails.

```sh
python tests/cached_evaluate/clone.py
```
