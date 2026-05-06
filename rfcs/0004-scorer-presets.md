---

## start_date: 2026-04-23

mlflow_issue: [https://github.com/mlflow/mlflow/issues/21445](https://github.com/mlflow/mlflow/issues/21445)
rfc_pr:

# Scorer Presets for Common Evaluation Patterns


| Author(s)              | Nehanth     |
| ---------------------- | ----------- |
| **Date Last Modified** | 2026-04-28  |
| **AI Assistant(s)**    | Claude Code |


# Summary

> **Note:** This RFC is based on [mlflow/mlflow#21445](https://github.com/mlflow/mlflow/issues/21445). The motivation, proposed presets, and API examples are derived from that issue, with additional design details and implementation specifics added here.

MLflow provides 21 built-in scorers for evaluating GenAI outputs, but users have no way to select a coherent subset for a specific evaluation pattern. Today, evaluating an agent requires importing and instantiating 9+ individual scorer classes -- boilerplate that gets copy-pasted across teams and templates.

This RFC proposes a `Preset` class that packages a named collection of scorers. MLflow ships built-in preset subclasses for common evaluation patterns (`Rag`, `Agent`, `ConversationalAgent`, `SafetyPreset`, `Quality`), and users can define their own. Presets can be passed directly in the `scorers` list alongside individual scorers, with automatic deduplication when presets overlap.

# Basic Example

```python
import mlflow
from mlflow.genai.scorers import Agent

# Use a built-in preset directly -- each call creates fresh scorer instances
result = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=predict_fn,
    scorers=[Agent()],
)
```

```python
# Mix presets and individual scorers
from mlflow.genai.scorers import Agent, Guidelines

result = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=predict_fn,
    scorers=[Agent(), Guidelines(name="tone", guidelines=["Respond professionally"])],
)
```

```python
# Combine presets -- duplicates are resolved automatically
from mlflow.genai.scorers import Agent, SafetyPreset

# Both contain Safety(); it runs once, not twice
result = mlflow.genai.evaluate(
    data=eval_dataset,
    scorers=[Agent(), SafetyPreset()],
)
```

```python
# Define a custom preset
from mlflow.genai.scorers import Preset, Safety, Fluency

my_preset = Preset("my_team_eval", scorers=[Safety(), Fluency(), my_custom_scorer])

result = mlflow.genai.evaluate(
    data=eval_dataset,
    scorers=[my_preset, another_scorer],
)
```

## Motivation

### The Problem

As described in [the original issue](https://github.com/mlflow/mlflow/issues/21445), the Databricks agent app template [evaluate_agent.py](https://github.com/databricks/app-templates/blob/main/agent-openai-agents-sdk/agent_server/evaluate_agent.py) imports and instantiates 9 separate scorers to evaluate a conversational agent:

```python
from mlflow.genai.scorers import (
    Completeness,
    ConversationalSafety,
    ConversationCompleteness,
    Fluency,
    KnowledgeRetention,
    RelevanceToQuery,
    Safety,
    ToolCallCorrectness,
    UserFrustration,
)

mlflow.genai.evaluate(
    data=simulator,
    predict_fn=predict_fn,
    scorers=[
        Completeness(),
        ConversationCompleteness(),
        ConversationalSafety(),
        KnowledgeRetention(),
        UserFrustration(),
        Fluency(),
        RelevanceToQuery(),
        Safety(),
        ToolCallCorrectness(),
    ],
)
```

Every team building agent evaluation follows this same pattern. This creates three problems (from the [original issue](https://github.com/mlflow/mlflow/issues/21445)):

1. **No built-in grouping.** `get_all_scorers()` returns all 19 default-constructible scorers. Users evaluating a RAG pipeline get `ToolCallCorrectness`; users evaluating an agent get `RetrievalGroundedness`. Each unnecessary scorer wastes an LLM API call.
2. **21 scorers to choose from.** Users must read documentation for each scorer to determine relevance. Session-level scorers (e.g., `KnowledgeRetention`) silently produce no results when passed to single-turn evaluation.
3. **Copy-paste problem.** The same scorer lists get duplicated across templates, notebooks, and tutorials. When new scorers are added, existing lists don't pick them up.

### Who Benefits

- **New users** get a curated starting point without reading all 21 scorer docs
- **Teams** can define and share custom presets, ensuring consistent evaluation across projects
- **Template authors** replace hardcoded scorer lists with a single preset
- **MLflow maintainers** gain a single place to update when new scorers are added

### Out of Scope

- **Parameterized presets.** Passing `model` or `inference_params` to all scorers in a preset. Users can iterate over the preset's scorers instead.
- **Third-party scorer presets.** Integrating presets for DeepEval, RAGAS, or TruLens scorers.
- **Preset registration/storage in the tracking server.** Presets are code-side only.

## Detailed Design

### The `Preset` Class

A `Preset` is a named, iterable container of scorers. It is **not** a `Scorer` subclass -- it is a grouping mechanism that gets flattened into individual scorers at validation time.

```python
class Preset:
    """A named, immutable collection of scorers for a common evaluation pattern.

    Presets can be passed in the ``scorers`` list alongside individual
    scorers. They are flattened and deduplicated during validation,
    so the evaluation loop only ever sees individual ``Scorer`` instances.

    Args:
        name: A descriptive name for this preset.
        scorers: The list of scorer instances in this preset.
    """

    def __init__(self, name: str, scorers: list[Scorer]):
        self._name = name
        self._scorers = tuple(self._deduplicate(scorers))

    @staticmethod
    def _deduplicate(scorers):
        seen = set()
        result = []
        for scorer in scorers:
            key = (type(scorer), scorer.name)
            if key not in seen:
                seen.add(key)
                result.append(scorer)
        return result

    @property
    def name(self) -> str:
        return self._name

    @property
    def scorers(self) -> tuple:
        return self._scorers

    def __iter__(self):
        return iter(self._scorers)

    def __len__(self):
        return len(self._scorers)

    def __add__(self, other):
        if isinstance(other, (Preset, list)):
            combined = list(self) + list(other)
            return self._deduplicate(combined)
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, list):
            combined = other + list(self)
            return self._deduplicate(combined)
        return NotImplemented

    def __repr__(self):
        scorer_names = [type(s).__name__ for s in self._scorers]
        return f"Preset('{self._name}', [{', '.join(scorer_names)}])"
```

**Key design decisions:**

- **Immutable and deduplicated.** Scorers are stored as a tuple and exposed via a read-only property. Deduplication happens in `__init__` and `__add__` using `(type, name)` as the key, so scorers of the same class with different names are preserved (e.g., two `Guidelines` with different rules).
- **Not a `Scorer` subclass.** A preset doesn't produce feedback -- it's a container. The evaluation loop assumes one scorer = one result column. Making `Preset` a scorer would require changes throughout the pipeline (aggregation, telemetry, serialization).
- **Iterable.** Supports `__iter__`, `__len__`, and `__add__`/`__radd__` so it composes naturally: `Agent() + [my_scorer]`, `[my_scorer] + Agent()`, or `Agent() + SafetyPreset()`.
- **Stores instances, not classes.** Users pass already-configured scorer instances.

### Built-in Presets as Subclasses

Each built-in preset is a subclass of `Preset` that hardcodes its scorer list. This means each call creates **fresh scorer instances** (no shared mutable singletons) and opens the door for preset-specific configuration and control flow in the future.

```python
class Agent(Preset):
    def __init__(self):
        super().__init__("agent", [
            ToolCallCorrectness(),
            ToolCallEfficiency(),
            RelevanceToQuery(),
            Safety(),
            Completeness(),
        ])

class Rag(Preset):
    def __init__(self):
        super().__init__("rag", [
            RetrievalRelevance(),
            RetrievalGroundedness(),
            RelevanceToQuery(),
            Safety(),
            Completeness(),
        ])

class ConversationalAgent(Preset):
    def __init__(self):
        super().__init__("conversational-agent", [
            ToolCallCorrectness(),
            ToolCallEfficiency(),
            RelevanceToQuery(),
            Safety(),
            Completeness(),
            UserFrustration(),
            ConversationCompleteness(),
            ConversationalSafety(),
            ConversationalToolCallEfficiency(),
            KnowledgeRetention(),
        ])

class SafetyPreset(Preset):
    def __init__(self):
        super().__init__("safety", [
            Safety(),
            ConversationalSafety(),
        ])

class Quality(Preset):
    def __init__(self):
        super().__init__("quality", [
            RelevanceToQuery(),
            Fluency(),
            Completeness(),
        ])
```

**Why subclasses over instances:**

- **Fresh instances every time.** `Agent()` creates new scorer instances on each call. No shared mutable state — the singleton problem is eliminated entirely.
- **Preset-specific configuration.** Each preset can accept its own parameters in the future (e.g., `Agent(model="openai:/gpt-4o")` to set the judge model for all scorers).
- **Type checking.** `isinstance(preset, Agent)` works — code can distinguish which preset is being used.
- **Custom control flow.** Each preset can override methods for preset-specific validation or behavior.

### Deduplication

When multiple presets are combined, the same scorer type can appear more than once. For example, `Agent()` and `SafetyPreset()` both contain `Safety()`. Running the same scorer twice wastes LLM API calls and produces duplicate result columns.

Deduplication happens in two places:

- **In the `Preset` class** — both `__init__` and `__add__` deduplicate using `(type(scorer), scorer.name)` as the key, so the preset is always clean whenever scorers are added or combined.
- **In `validate_scorers()`** — when multiple presets are passed directly in a list (e.g., `scorers=[Agent(), SafetyPreset()]`) without using `+`, `__add__` is never called. `validate_scorers()` flattens and deduplicates as a safety net:

```python
def validate_scorers(scorers: list[Any]) -> list[Scorer]:
    from mlflow.genai.scorers.presets import Preset

    # 1. Flatten presets into individual scorers
    flat = []
    for item in scorers:
        if isinstance(item, Preset):
            flat.extend(item)
        else:
            flat.append(item)

    # 2. Deduplicate by (type, name)
    flat = Preset._deduplicate(flat)

    # 3. Existing validation on the flattened list
    ...
```

Scorers of the same class with different names are preserved (e.g., two `Guidelines` with different rules). Only true duplicates — same class and same name — are removed.

`evaluate()` itself does not change. By the time scorers reach the evaluation loop, they are all individual `Scorer` instances.

### Built-in Preset Summary

MLflow ships five built-in preset subclasses. Each call creates fresh scorer instances.

| Preset                 | Scorers                                                                                                                                 | Use Case                                                 |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| `Rag()`                | RetrievalRelevance, RetrievalGroundedness, RelevanceToQuery, Safety, Completeness                                                       | Retrieval-augmented generation pipelines                 |
| `Agent()`              | ToolCallCorrectness, ToolCallEfficiency, RelevanceToQuery, Safety, Completeness                                                         | Single-turn tool-calling agents                          |
| `ConversationalAgent()`| All of `Agent` + UserFrustration, ConversationCompleteness, ConversationalSafety, ConversationalToolCallEfficiency, KnowledgeRetention  | Multi-turn conversational agents                         |
| `SafetyPreset()`       | Safety, ConversationalSafety                                                                                                            | Safety-focused evaluation (composable with other presets) |
| `Quality()`            | RelevanceToQuery, Fluency, Completeness                                                                                                 | Architecture-independent output quality                  |


#### Design Rationale

- **Safety is in `Rag` and `Agent`** because these presets aim to be complete starting points. Most users want safety checks without composing two presets.
- **Fluency is excluded from `Agent`** because agent evaluation emphasizes tool usage and task completion. Users who need it can compose: `Agent() + [Fluency()]`.
- **`ConversationalAgent` excludes `ConversationalRoleAdherence`** because it requires a defined persona in the system prompt, which not all agents have.
- **`RetrievalSufficiency` is excluded from `Rag`** because it requires `expected_response` or `expected_facts` (ground truth). Users who have expectations data can add it manually: `Rag() + [RetrievalSufficiency()]`.
- **`Correctness` is excluded from all presets** because it requires `expectations` (ground truth) data. Users who have ground truth can add it manually: `Quality() + [Correctness()]`.
- **`Guidelines` and `ConversationalGuidelines` are excluded from all presets** because both require a `guidelines` constructor argument.

## Drawbacks

1. **New class in the API.** Adds `Preset` to the public surface. Mitigation: it's a simple container with no complex behavior.
2. **Opinionated defaults.** Not everyone will agree on which scorers belong in which preset. Mitigation: presets are extensible via `+`, and users can define their own.
3. **Implicit behavior changes on upgrade.** A new scorer added to a built-in preset means different evaluation results after upgrading. Consistent with how `get_all_scorers()` already behaves.

# Alternatives

### 1. `get_preset()` function (no class)

Instead of a `Preset` class, provide a simple function that returns a plain list:

```python
from typing import Literal

from mlflow.exceptions import MlflowException
from mlflow.genai.scorers.builtin_scorers import (
    Completeness,
    ConversationalSafety,
    ConversationalToolCallEfficiency,
    ConversationCompleteness,
    Correctness,
    Fluency,
    KnowledgeRetention,
    RelevanceToQuery,
    RetrievalGroundedness,
    RetrievalRelevance,
    RetrievalSufficiency,
    Safety,
    ToolCallCorrectness,
    ToolCallEfficiency,
    UserFrustration,
)

_PRESETS: dict[str, list[type]] = {
    "rag": [
        RetrievalRelevance,
        RetrievalSufficiency,
        RetrievalGroundedness,
        RelevanceToQuery,
        Safety,
        Completeness,
    ],
    "agent": [
        ToolCallCorrectness,
        ToolCallEfficiency,
        RelevanceToQuery,
        Safety,
        Completeness,
    ],
    "conversational-agent": [
        ToolCallCorrectness,
        ToolCallEfficiency,
        RelevanceToQuery,
        Safety,
        Completeness,
        UserFrustration,
        ConversationCompleteness,
        ConversationalSafety,
        ConversationalToolCallEfficiency,
        KnowledgeRetention,
    ],
    "safety": [
        Safety,
        ConversationalSafety,
    ],
    "quality": [
        RelevanceToQuery,
        Fluency,
        Completeness,
        Correctness,
    ],
}

_VALID_PRESET_NAMES = ", ".join(sorted(_PRESETS.keys()))
PresetName = Literal["rag", "agent", "conversational-agent", "safety", "quality"]


def get_preset(name: PresetName) -> list:
    if name not in _PRESETS:
        raise MlflowException.invalid_parameter_value(
            f"Unknown preset '{name}'. Valid presets are: {_VALID_PRESET_NAMES}"
        )
    return [scorer_class() for scorer_class in _PRESETS[name]]


def list_presets() -> dict[str, list[str]]:
    return {
        name: [cls.__name__ for cls in classes]
        for name, classes in _PRESETS.items()
    }
```

Usage:

```python
from mlflow.genai.scorers import get_preset

# Simple usage
result = mlflow.genai.evaluate(scorers=get_preset("agent"))

# Extending a preset
scorers = get_preset("agent") + [Guidelines(name="tone", guidelines=["Be professional"])]
result = mlflow.genai.evaluate(scorers=scorers)
```

**Pros:** Simpler (~30 lines). No validation changes needed. Returns fresh instances each call (no mutable singleton concern). `Literal` type gives IDE autocompletion. Going from function to class later is non-breaking.

**Cons:** No user-defined presets. Composition requires `+` with list concatenation. The preset concept disappears immediately -- it's just a list. No deduplication when combining presets.

This is a viable first step if the class approach is deemed too heavy. The class can be added later as a non-breaking extension.

### 2. Tag-based filtering

Add `categories` to each scorer class and provide `get_scorers(categories=["rag"])`. More flexible but over-engineered for 21 scorers and requires modifying every existing class.

### 3. Enum-based API

`ScorerPreset.RAG.get_scorers()`. Type-safe but heavier API surface. The `Literal` type on a function already provides IDE autocompletion.

### 4. Do nothing

Users keep copy-pasting scorer lists. Does not scale as the scorer count grows.

# Adoption Strategy

This is an **additive, non-breaking change**. Existing code continues to work unchanged.

- Update documentation and templates to show `Preset` usage alongside the manual import pattern.
- Update the `validate_scorers()` error message to mention presets for discoverability.
- Databricks agent templates can simplify from 9 imports + 9 instantiations to `scorers=[ConversationalAgent()]`.

# Open Questions

1. **Should `ConversationalRoleAdherence` be in `ConversationalAgent`?** Currently excluded because it requires a defined persona. **Open for discussion.**
2. **Should `Correctness` be in `Agent` or `Rag`?** Currently excluded from all presets because it requires `expectations` data. **Open for discussion.**
3. **Should there be an `All` preset?** `get_all_scorers()` already serves this role. **Recommendation:** Do not add.
4. **Deduplication key.** Should deduplication use `type(scorer)` alone, or `(type(scorer), scorer.name)`? The latter preserves multiple instances of the same class with different names (e.g., two `Guidelines` with different rules).
5. **Future: parameterized presets?** e.g., `Agent(model="openai:/gpt-4o")` to set the judge model for all scorers in the preset. Can be a future addition.

