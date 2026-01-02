# CLAUDE.md - Experiment Scorers Directory

This document provides guidance for working with the experiment-scorers feature, which manages LLM judges and custom code scorers for evaluating ML traces.

## Component Hierarchy Diagram

```
Data source legend:  [scorer] = ScheduledScorer    [form] = form data via control

ExperimentScorersPage
  │  (Feature gate, Error boundary, Trace prefetch)
  │
  └─► ExperimentScorersContentContainer
        │  Data: useGetScheduledScorers() → ScheduledScorer[]
        │
        ├─► ScorerEmptyStateRenderer [scorer] (if no scorers)
        │
        ├─► ScorerCardContainer[] [scorer → form init]
        │     │
        │     ├─► ScorerCardRenderer [scorer + form] ⚠️ see note below
        │     │     └─► LLMScorerFormRenderer [form]
        │     │         or CustomCodeScorerFormRenderer [form]
        │     │
        │     ├─► ScorerModalRenderer (mode=EDIT)
        │     │     └─► ScorerFormEditContainer [scorer → form init]
        │     │           └─► ScorerFormRenderer [form]
        │     │
        │     └─► DeleteScorerModalRenderer [scorer]
        │
        └─► ScorerModalRenderer (mode=CREATE)
              └─► ScorerFormCreateContainer [form only]
                    └─► ScorerFormRenderer [form]


ScorerFormRenderer [form]
  │
  ├─► Judge Type Radio [form] (CREATE only)
  │
  ├─► LLMScorerFormRenderer [form]
  │     ├─► LLMTemplateSection [form]
  │     ├─► NameSection [form]
  │     ├─► GuidelinesSection [form]
  │     ├─► InstructionsSection [form]
  │     ├─► ModelSectionRenderer [form]
  │     │     └─► EndpointSelector (reusable, self-contained)
  │     │           Data: useEndpointsQuery() → Endpoint[]
  │     └─► EvaluateTracesSectionRenderer [form]
  │
  ├─► CustomCodeScorerFormRenderer [form]
  │
  └─► SampleScorerOutputPanelContainer [form]
        └─► SampleScorerOutputPanelRenderer

⚠️ ScorerCardRenderer currently receives both scorer (for header) and control
   (for expanded form). Ideally should only take scorer, with form initialized
   in a child container. See principle #2.
```

## Data Flow

```
Transform Layer (scorerTransformUtils.ts)
─────────────────────────────────────────

  ScorerConfig ────────► ScheduledScorer ────────► ScorerFormData
  (API shape)            (UI domain model)         (form state)
                         e.g. guidelines: []       e.g. guidelines: ""

  ScorerConfig ◄──────── ScheduledScorer ◄──────── ScorerFormData
```

## Key Design Principles

**Important: New features should be designed keeping these principles in mind**

### 1. Data Fetching / Rendering Separation

Containers own data fetching, mutations, and state. Renderers just display what they're given.

```
Container (data)                   Renderer (presentation)
────────────────                   ───────────────────────
ScorerCardContainer        ────►   ScorerCardRenderer
ScorerFormCreateContainer  ────►   ScorerFormRenderer
ScorerFormEditContainer    ────►   ScorerFormRenderer
SampleScorerOutputPanel-   ────►   SampleScorerOutputPanelRenderer
  Container
```

Renderers should NOT call `useQuery`, `useMutation`, or manage complex state. They receive data via props and render it.

**Note**: In some cases if the component is simple you can just define a component like ModelSection that does state management + rendering on its own without container/renderer separation. But any complex components should be split up.

### 2. Data Source: `scorer` vs Form Data

**When to use each data source:**

| Use `scorer` (ScheduledScorer)         | Use form data (via `control`)                |
| -------------------------------------- | -------------------------------------------- |
| Displaying persisted/saved state       | Editing or previewing editable fields        |
| Card header, collapsed summary, badges | Expanded card view, edit modal, create modal |

**Key insight:** The expanded card view is a **read-only form preview** - it shows the same form the user would edit, just not editable yet. This is why it uses form data, not `scorer`.

**Component rules:**

- Components should only receive **one data source** (either `scorer` or `control`)
- **Exception**: Container components that need `scorer` to initialize the form via `getFormValuesFromScorer()`
- If a component needs both for other reasons, consider splitting it

**Current state:** `ScorerCardRenderer` receives both `scorer` (for header) and `control` (for expanded form area). Ideally it should only take in `scorer`, and the form should be initialized in a child container component. Future components should keep this distinction clear.

### 3. Type Hierarchy: ScorerConfig vs ScheduledScorer vs ScorerFormData

Three types represent a scorer at different layers:

| Type                | Purpose                            | Shaped by                            |
| ------------------- | ---------------------------------- | ------------------------------------ |
| **ScorerConfig**    | Wire format (API request/response) | Backend API contract                 |
| **ScheduledScorer** | Central UI domain model            | What a scorer semantically IS        |
| **ScorerFormData**  | Form state for react-hook-form     | What the user sees/types in the form |

**When to add a field to each type:**

- **ScorerConfig**: Maps 1:1 to backend API. Only add fields that exist in the API response or are needed for API requests.

- **ScheduledScorer**: The canonical frontend representation of "what is a scorer". Add fields here when they represent the true semantic value. Not all fields need to be exposed in the form, and the shape may differ from form representation (e.g., `guidelines: string[]` as an array for business logic).

- **ScorerFormData**: Must be 1:1 with form controls the user interacts with. If there's a textarea, there's a string field. If there's a checkbox, there's a boolean. Shape is dictated by UI controls, not domain semantics.

**Example: `guidelines` field**

```
ScorerConfig.builtin_scorer_pydantic_data.guidelines: string[]  (API)
       ↓ transformScorerConfig()
ScheduledScorer.guidelines: string[]  (domain - array for iteration/logic)
       ↓ getFormValuesFromScorer()
ScorerFormData.guidelines: string  (form - newline-joined for textarea)
       ↓ convertFormDataToScheduledScorer()
ScheduledScorer.guidelines: string[]  (split by '\n', trimmed, filtered)
```

In most cases, `ScheduledScorer` and `ScorerFormData` will be similar, but they can diverge when the form representation differs from the domain representation.
