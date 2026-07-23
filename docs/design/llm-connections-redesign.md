# LLM Connections Redesign — Design Plan

> Status: Design draft (branch `design/llm-connections-general`)
> Design system: **Data-Dense Dashboard** (Databricks Design System / DuBois) — clean, dense, functional, light-mode default, WCAG AA, full light+dark.

## 1. Problem & Goals

**Today**
- "LLM Connections" is buried as a *separate* settings tab (`/settings/llm-connections`), one of three peers: General · LLM Connections · Webhooks. It embeds the gateway API-keys manager.
- A connection = `provider` + encrypted API key + `auth_config`. No first-class notion of a *preferred / allowlisted model* attached to a key.
- LLM-powered surfaces (Detect Issues → `IssueDetectionModal`/`GenAIModelSelection`, Assistant, scorers) force the user to re-pick **provider + model + key** inline *every single time*. High-friction, error-prone, duplicative.

**Goals**
1. **Surface connections front-and-center** — fold the manager into the **General** settings tab so it's the first thing a user sees, not a tab they have to discover.
2. **Allowlist models per key** — when adding a provider key, the user also picks one or more preferred models to pair with it (e.g. Anthropic key → Claude Sonnet + Claude Opus). Backed by the gateway.
3. **One-pick consumption** — every LLM surface shows a **simple dropdown of allowlisted `provider · model` pairs**. No inline provider/model/key entry.

---

## 2. Information Architecture

### Before
```
Settings
 ├─ General          (misc org settings)
 ├─ LLM Connections  ← manager lives here (separate tab)
 └─ Webhooks
```

### After
```
Settings
 ├─ General
 │    ├─ (existing general settings)
 │    └─ ── LLM Connections ─────────────────  ← manager promoted into General
 │           • Connection list / cards
 │           • "Add connection" → provider + key + model allowlist
 └─ Webhooks
```
- The standalone `/settings/llm-connections` route **redirects** to `/settings/general#llm-connections` (anchor-scroll) so existing deep links and the Detect-Issues "Add a connection" shortcut don't break.
- Within General, "LLM Connections" is a clearly-titled **section** (H2 + short helper text), not a nested tab — matches `nav-hierarchy` (primary nav = settings tabs; this is section-level content).

---

## 3. Connections Section in General Settings

### Layout — populated state
A **section header** + **connection list**. Each connection is a row/card (DuBois `Table` row or `Card`) showing provider, masked key, and its allowlisted models as chips.

```
┌─ LLM Connections ─────────────────────────────────────────────────────┐
│ Store provider keys and choose which models MLflow features can use.   │
│                                                    [ + Add connection ] │
│                                                                         │
│ ┌─────────────────────────────────────────────────────────────────┐   │
│ │ ◆ Anthropic          sk-ant-…x9f2        Updated 2d ago   ⋯      │   │
│ │   Allowed models:  [Claude Sonnet ✕] [Claude Opus ✕]  [+ model]  │   │
│ ├─────────────────────────────────────────────────────────────────┤   │
│ │ ◆ OpenAI             sk-…a71b            Updated 5d ago   ⋯      │   │
│ │   Allowed models:  [gpt-5 ✕] [gpt-5-mini ✕]           [+ model]  │   │
│ └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

- **Provider glyph + name** — one consistent icon family (no emoji; SVG per `no-emoji-icons`). Reuse existing `ProviderSelect` iconography.
- **Masked key** — tabular figures (`number-tabular`), never the full secret. Matches existing `masked_values`.
- **Allowed-models chips** — each model an inline removable chip; `[+ model]` opens the model allowlist picker (reuses `ModelSelectorModal` filtered to that provider). This is the *new* first-class concept.
- **Row overflow `⋯`** — Edit key · Rotate key · Remove connection (destructive, red, spatially separated per `destructive-emphasis`).
- **Density** — minimal padding, row-highlight on hover, consistent 8px rhythm. `virtualize-lists` only if >50 connections (unlikely; skip).

### Empty state (`empty-states`)
Never a blank box — a helpful prompt with a single primary CTA:
```
┌─ LLM Connections ─────────────────────────────────────────────────────┐
│               ◆  No connections yet                                     │
│   Add a provider key so features like Detect Issues and the            │
│   Assistant can call an LLM on your behalf.                            │
│                        [ + Add your first connection ]                  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Error / degraded states
- **Key invalid / auth failed** — inline status pill on the row (`● Auth failed`, destructive color + icon, not color-alone per `color-not-only`) with a "Re-enter key" action.
- **Gateway unreachable** — section-level inline alert with retry (`error-recovery`), list still renders last-known connections read-only.

---

## 4. Add-Connection + Model-Allowlist Flow

A single modal with **progressive disclosure** (`progressive-disclosure`) — the model step only becomes meaningful after a provider + key exist.

```
Add connection  (modal)
 Step 1 · Provider & key
   Provider   [ Anthropic ▾ ]        ← ProviderSelect (existing)
   API key    [ ••••••••••••••• 👁 ] ← password-toggle, autocomplete off
              helper: "Stored encrypted. Never shown again after saving."
   (+ provider-specific auth_config fields when required — Azure region, etc.)

 Step 2 · Allowed models
   "Choose which models this key can be used with."
   [ + add model ]  → ModelSelectorModal (filtered to Anthropic, capability tags)
   Selected:  [Claude Sonnet ✕]  [Claude Opus ✕]
              helper: "These appear as options wherever MLflow needs a model."

              [ Cancel ]                        [ Save connection ]
```

- **Validation** — key validated on blur / on save against the gateway (`inline-validation`, `submit-feedback`): button → loading → success toast or inline error with cause + fix (`error-clarity`).
- **At least one model** required to save (a key with zero allowlisted models can't be consumed anywhere → dead config). Enforce with a clear inline message, not a silent disabled button.
- **Reuses**: `ProviderSelect`, `SecretFormFields`, `ModelSelectorModal` / `ModelSelect`. The only genuinely new UI is the *multi-model chip collector* in Step 2 — a thin wrapper around `ModelSelectorModal` that accumulates selections.
- **Edit flow** — same modal, key field shows masked + "Replace key" affordance (`read-only-distinction`); model allowlist fully editable.

---

## 5. Allowlisted-Pair Dropdown on Consuming Surfaces

Replace the inline provider/model/key blocks in `GenAIModelSelection` (used by `IssueDetectionModal`, and to be used by Assistant / scorers) with **one dropdown**.

```
Model   [ Anthropic · Claude Sonnet          ▾ ]
        ┌──────────────────────────────────────┐
        │ Anthropic · Claude Sonnet             │
        │ Anthropic · Claude Opus               │
        │ OpenAI · gpt-5                        │
        │ OpenAI · gpt-5-mini                   │
        │ ──────────────────────────────────── │
        │ + Manage connections →                │  → deep-links to General#llm-connections
        └──────────────────────────────────────┘
```

- **Options = the flat list of every allowlisted `provider · model` pair** across all connections. Label format `Provider · Model`; optional capability tags (Tools/Reasoning) as trailing muted chips for disambiguation.
- **No inline key/provider/model entry** on the surface anymore — that responsibility moved entirely to the connections manager. Single source of truth.
- **Empty state inside the dropdown** — if no connections exist: a disabled explanatory row + "＋ Add a connection →" that deep-links to General settings (`empty-nav-state` — explain, don't silently hide).
- **Active/selected** pair clearly marked (`nav-state-active`). Default selection = last-used pair (persisted), else first available.
- Backend contract unchanged: the surface still ultimately sends `provider` + `model` + `secret_id` — it just derives them from the chosen pair instead of collecting them inline.

---

## 6. Visual & Interaction Guidelines (applied)

| Area | Rule | Application |
|------|------|-------------|
| Style | Data-Dense Dashboard | Minimal padding, table/row density, DuBois tokens, light default + full dark. |
| Icons | `no-emoji-icons`, `icon-style-consistent` | One SVG provider-icon set; overflow `⋯`, chips use consistent glyphs. |
| Forms | `input-labels`, `password-toggle`, `inline-validation`, `submit-feedback` | Visible labels, masked key w/ toggle, blur validation, loading→success/error. |
| Feedback | `empty-states`, `error-recovery`, `error-clarity`, `undo-support` | Guided empties; retry on gateway errors; "Undo" toast on connection remove. |
| Destructive | `destructive-emphasis`, `confirmation-dialogs` | Remove connection = red, separated, confirm dialog (invalidates dependent surfaces). |
| Color | `color-not-only`, `color-accessible-pairs` | Status pills pair icon+text; 4.5:1 contrast both themes. |
| A11y | `focus-states`, `form-labels`, `keyboard-nav` | Visible focus rings, labelled chips (removable via keyboard), dropdown keyboard-navigable. |
| Motion | `duration-timing`, `reduced-motion` | 150–300ms row hover / modal transitions; respect reduced-motion. |
| Nav | `deep-linking`, `nav-state-active`, `empty-nav-state` | Old route redirects; "Manage connections" deep-links; active pair highlighted. |

### Anti-patterns to avoid
- ❌ Keeping connections as a discoverable-only separate tab.
- ❌ Re-collecting provider/model/key on every surface (the current pain).
- ❌ Allowing a saved key with no allowlisted models (dead config).
- ❌ Color-only status; blank empty states; unlabelled icon-only controls.

---

## 7. Screen Inventory (for the flow board)

1. **Settings › General (empty)** — connections section with empty state.
2. **Settings › General (populated)** — connection list with model chips.
3. **Add connection — Step 1** — provider + key.
4. **Add connection — Step 2** — model allowlist collector.
5. **Connection saved** — success toast, new row in list.
6. **Consuming surface (Detect Issues) — old** → **new** allowlisted-pair dropdown.
7. **Dropdown open** — list of `provider · model` pairs + "Manage connections".
8. **Dropdown empty** — no connections → deep-link CTA.

Transitions feed directly into `/ui-flow-board`.
