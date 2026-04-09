# Trace View Deeplinks Design Spec

**Date:** 2026-04-09
**Status:** Draft

## Overview

Add support for inline deeplinks within `SpanRange.description` text that navigate to specific spans in the trace timeline. This enables trace view summaries to reference focal points — e.g., "The agent had an issue with tool calls [here](spans/span-abc123)" — where clicking "here" scrolls to and highlights the referenced span.

## Deeplink Syntax

Deeplinks use standard markdown link syntax with a path following the existing information architecture:

**Fully-qualified:**
```
[link text](/experiments/{experiment_id}/traces/{trace_id}/spans/{span_id})
```

**Relative (resolved against current trace context):**
```
[link text](spans/{span_id})
```

The fully-qualified form supports cross-trace linking. The relative form is a convenience for the common case of linking within the same trace.

## Data Model

No changes. `SpanRange.description` is already a `Text` field that accepts arbitrary content. Deeplinks are a rendering convention, not a schema concern. The backend does not parse or validate deeplink syntax — it stores and returns the description as-is.

## Frontend Changes

### Description Renderer

Today, `SpanRange.description` is rendered as plain text in:
- `ModelTraceExplorerRangesView` (summary view, position 0 overview + step cards)
- `ModelTraceExplorerRangeDetailView` (single range detail)

**Change:** Replace plain text rendering with a component that parses markdown-style links (`[text](path)`) and renders them as clickable elements. Non-link text remains plain text — no need for full markdown support.

**Parsing rules:**
- Match `[link text](path)` patterns in the description string
- A path is a deeplink if it matches:
  - `/experiments/{id}/traces/{id}/spans/{id}` (fully-qualified)
  - `spans/{id}` (relative)
- Non-matching links are rendered as plain text (not clickable)

### Click Handler

When a deeplink is clicked:

1. **Parse** the path into segments: `experimentId`, `traceId`, `spanId`
2. **Resolve** relative paths (`spans/{spanId}`) using the current trace/experiment context
3. **Route:**
   - **Same trace:** Call `setSelectedNode` via `ModelTraceExplorerViewStateContext` to select the span, then scroll it into view in the timeline tree
   - **Different trace:** Navigate via React Router to the target trace with a `focusSpan` query parameter (see below), which handles span selection on mount

### `focusSpan` Query Parameter

A new URL query parameter following the existing `selectedTraceId` pattern.

- **Name:** `focusSpan`
- **Value:** Span ID to focus on mount
- **Defined in:** `constants.ts` alongside `SELECTED_TRACE_ID_QUERY_PARAM`
- **Consumed by:** The component that instantiates `ModelTraceExplorer`, passing the value as the `selectedSpanId` prop
- **Behavior:** Feeds into the existing `selectedSpanIdOnRender` → `searchTreeBySpanId()` machinery to select the span when the trace explorer mounts

### Scroll Into View

When a span is programmatically selected (via deeplink click or `focusSpan` param), the corresponding timeline tree node should scroll into view.

- Use `scrollIntoView({ behavior: 'smooth', block: 'center' })` on the timeline tree node's DOM element
- Trigger after the selected node state update, using a `useEffect` that watches `selectedNode`
- Only scroll when selection is driven programmatically (deeplink or query param), not on normal user clicks in the timeline tree (the user is already looking at what they clicked)

To distinguish programmatic selection from user clicks, use a ref flag (e.g., `programmaticSelectionRef`) set to `true` before calling `setSelectedNode` from a deeplink/query param path, and checked/cleared in the scroll effect.

## Programmatic Authoring

No backend or API changes are needed. Deeplinks are just text in the description field. AI-generated views and programmatic `create_view()` calls can emit deeplink syntax naturally since they have access to span IDs.

For example, an LLM analyzing a trace might produce:
```python
SpanRange(
    label="Tool Call Failure",
    description="The agent's search tool failed due to a malformed query [here](spans/span-abc). "
                "It retried with corrected parameters [here](spans/span-def) and succeeded.",
    from_selector=SpanSelector(span_id="span-abc"),
    to_selector=SpanSelector(span_id="span-def"),
    position=1,
)
```

## Scope Summary

| Area | Change Required |
|------|----------------|
| Database schema | None |
| Backend API | None |
| Python data model | None |
| Description rendering (2 components) | Parse markdown links, render as clickable |
| Click handler | New — resolve path, navigate or select span |
| `focusSpan` query param | New — wire into existing `selectedSpanId` prop |
| Scroll into view | New — `scrollIntoView` on programmatic selection |

## Out of Scope

- Full markdown rendering in descriptions (only `[text](path)` links are parsed)
- Selector-based deeplinks (e.g., matching by span name/type instead of ID)
- Deeplink validation on the backend
- Editing deeplinks in the view editor UI
