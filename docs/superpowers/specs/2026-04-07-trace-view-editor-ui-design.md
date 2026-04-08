# Trace View Editor UI Design

Create and edit trace views directly from the MLflow UI. Users can define which spans to include via drag-to-select on the timeline, configure which JSON fields to extract via checkbox trees on span inputs/outputs, and save views with names and descriptions.

## Motivation

Trace views currently can only be created via the Python API or CLI. The UI only supports read-only rendering of existing views via the `TraceViewSelector` dropdown and `ModelTraceExplorerRangesView`. Users need a way to create and edit views visually — selecting spans, configuring JSON field extraction, and managing view metadata — without leaving the browser.

The interaction model is inspired by LangSmith's configure input/output preview, which presents trace data as an expandable tree with checkboxes for field selection.

## Scope

- **In scope:** Creating and editing trace-scoped views from the UI
- **Out of scope:** Experiment-scoped views (templates), bulk operations, programmatic view creation from UI

## Design

### Entry Points

The `TraceViewSelector` dropdown gains two new affordances:

1. **"+ Create View" menu item** at the bottom of the dropdown. Clicking it enters edit mode with an empty view (no ranges, blank name).
2. **Edit icon** next to the active view name when a view is selected. Clicking it enters edit mode with the existing view's ranges pre-loaded.

### Edit Mode Lifecycle

Entering edit mode:
- A top toolbar appears with: editable name field, Cancel button, Save button
- The detail/timeline view and summary view both enter "range selection mode" — checkboxes appear on every span row
- Spans belonging to existing ranges are pre-checked and color-coded by range
- Unselected spans are dimmed (reduced opacity)

Exiting edit mode:
- **Save** calls `createTraceView` (new) or `updateTraceView` (existing) via the REST API, then exits edit mode
- **Cancel** discards all draft changes and exits edit mode, reverting to the previously selected view (or no view if creating)

### Range Creation & Management

**Creating ranges via drag:**
- In edit mode, the user drags vertically across span rows in the timeline or summary view
- The top span becomes `from_selector` (matched by `span_id`), bottom span becomes `to_selector`
- All spans between them in tree traversal order are included
- A new range is created with an auto-generated label ("Range 1", "Range 2", etc.) and a unique color from a fixed palette

**Single-span ranges:**
- Clicking a span's checkbox toggles it into a new single-span range (`from_selector` only, `to_selector` is null)

**Range badges:**
- Each range gets a colored badge displayed inline above its first span in the list
- Badge content: label, delete icon (in edit mode only)
- Colors are assigned from a fixed palette (blue, purple, green, amber, red, cyan) cycling by range index

**Range containment logic:**
- `isSpanInRange()` in `useTraceViewFiltering.ts` is the shared function for determining if a span falls within a range
- Uses `spanMatchesSelector()` for boundary matching (supports span_id, span_name, span_type, attributes)
- Both edit mode selection and view mode highlighting use this same function
- Tree flattening uses `getTimelineTreeNodesList()` from `TimelineTree.utils.ts` — no duplicated traversal logic

**Deleting a range:**
- Click the delete icon on the range badge
- Removes the range from the draft, unchecks its spans, removes highlighting

### JSON Field Selection

**Accessing the JSON tree:**
- In edit mode, each checked span row displays a gear (⚙) icon on the right
- Clicking it expands an inline panel beneath that span row
- The panel shows two side-by-side trees: "Input Fields" and "Output Fields"

**The checkbox tree (`JsonFieldSelector`):**
- Each property or array item in the JSON is a tree node with:
  - Expand/collapse arrow (for objects and arrays)
  - Checkbox
  - Property name or array index
  - Dimmed preview of the value (truncated strings, item counts for arrays, `{...}` for objects)
- Checking a leaf node selects that specific field
- The generated JSONPath is displayed below each tree as a read-only text field

**Per-range scope:**
- JSON field configuration applies to the entire range, not individual spans
- The gear icon on any span in the range opens the same configuration panel (using that span's actual data as the example tree structure)
- A note clarifies: "These paths apply to all spans in this range"

### View Mode (Non-Edit)

When a saved trace view is selected (not in edit mode):

**Timeline tree highlighting:**
- Spans within ranges are shown with colored left borders and backgrounds matching their range color
- Spans outside ranges are dimmed (reduced opacity)
- Range badges appear above the first span of each range (without delete button)

**Selectable range badges:**
- Clicking a range badge in view mode selects it, showing a selected border state
- The right pane switches from span tabs to a **range detail view** showing:
  - Range label and description
  - Input/output JSON paths (if configured)
  - Filtered inputs/outputs of the currently selected span
- Clicking a span in the tree clears the range selection and returns to normal span tab view

### Visual States

**Span row states in edit mode:**

| State | Appearance |
|-------|-----------|
| Unselected | Dimmed (reduced opacity), empty checkbox |
| Selected (in range) | Full opacity, checked checkbox, colored left border and background matching range color |
| Drag hover | Light blue highlight background indicating pending selection |
| Range boundary (from) | Range badge displayed above this row |

**Span row states in view mode:**

| State | Appearance |
|-------|-----------|
| Not in any range | Dimmed (reduced opacity) |
| In range | Full opacity, colored left border and background matching range color |
| Range boundary (from) | Clickable range badge displayed above this row |

## Components

### New Components

**`TraceViewEditToolbar`** (`edit-mode/TraceViewEditToolbar.tsx`)
- Renders above the span list when in edit mode
- Contains: name text input, Cancel button, Save button
- Save button is disabled when name is empty or no ranges exist
- Rendered in both the detail view and summary view

**`useSpanRangeSelection`** (`edit-mode/useSpanRangeSelection.ts`)
- Hook that manages drag-to-select, checkbox clicks, and per-node edit state
- Takes a flat node list, draft ranges, and add/remove callbacks
- Returns `getNodeEditState(flatIndex)` for rendering, pointer event handlers, and span key index
- Uses `isSpanInRange()` from `useTraceViewFiltering` for range containment (no duplicated logic)

**`RangeBadge`** (`edit-mode/RangeBadge.tsx`)
- Colored pill showing range label
- Optional `onDelete` for edit mode (renders close icon)
- Optional `onClick` and `isSelected` for view mode (clickable with selected border state)

**`JsonFieldSelector`** (`edit-mode/JsonFieldSelector.tsx`)
- Inline expandable panel beneath a span row
- Takes a parsed JSON object and renders it as a tree with checkboxes
- Two instances per expansion: one for input, one for output
- Emits JSONPath string on checkbox change
- Props: `data: unknown`, `selectedPath: string | null`, `onPathChange: (path: string | null) => void`, `label: string`

**`ModelTraceExplorerRangeDetailView`** (`right-pane/ModelTraceExplorerRangeDetailView.tsx`)
- Shows range details when a range badge is clicked in view mode
- Displays: range label, description, input/output paths, filtered span inputs/outputs
- Uses `ModelTraceExplorerCodeSnippet` and `ModelTraceExplorerCollapsibleSection` for consistency with the existing content tab

**`rangeColors`** (`edit-mode/rangeColors.ts`)
- Fixed palette of 6 colors (blue, purple, green, amber, red, cyan) with primary and background variants
- `getRangeColor(index)` cycles through the palette

### New/Updated Hooks

**`useTraceViewEditMode`** (`hooks/useTraceViewEditMode.ts`)
- Manages the draft `TraceView` object during editing
- Colocated with `ModelTraceExplorerViewStateContext`
- Exposes: `isEditMode`, `draftView`, `enterEditMode`, `exitEditMode`, `addRange`, `removeRange`, `updateRange`, `reorderRanges`, `setName`

**`useTraceViewMutations`** (`hooks/useTraceViewMutations.ts`)
- `useCreateTraceView(traceId)` — POST mutation
- `useUpdateTraceView(traceId)` — PATCH mutation
- `useDeleteTraceView(traceId)` — DELETE mutation
- All invalidate the trace views query cache on success

### Updated Shared Logic

**`useTraceViewFiltering`** (`hooks/useTraceViewFiltering.ts`)
- `isSpanInRange(node, flatNodes, range)` — shared function used by both edit mode and view mode
- `spanMatchesSelector(span, selector)` — used by `isSpanInRange` for boundary matching
- `useTraceViewSpanMatches(allNodes, activeView)` — returns `{ matchedKeys, rangeMap }` where `rangeMap` provides per-span range info (`rangeIdx`, `isFirstInRange`)
- Tree flattening uses `getTimelineTreeNodesList()` from `TimelineTree.utils.ts`

### Modified Components

**`TraceViewSelector`**
- Added "+ Create View" menu item at the bottom of the dropdown
- Added edit icon (pencil) next to the active view name

**`ModelTraceExplorerViewStateContext`**
- Added `editMode` (from `useTraceViewEditMode`) to context
- Added `selectedViewRangeIdx` / `setSelectedViewRangeIdx` for range badge selection in view mode

**`ModelTraceExplorerDetailView`**
- Renders `TraceViewEditToolbar` in edit mode
- Clears `selectedViewRangeIdx` when a span is clicked
- Removed the "Viewing: {name}" banner bar (replaced by range badges)

**`ModelTraceExplorerRightPaneTabs`**
- When `selectedViewRangeIdx` is set, renders `ModelTraceExplorerRangeDetailView` instead of normal span tabs

**`TimelineTree` / `TimelineTreeNode`**
- In edit mode: renders checkboxes, handles drag-to-select via `useSpanRangeSelection`, shows edit badges with delete
- In view mode: renders range badges (clickable), colored borders/backgrounds via `viewRangeMap`
- Passes `viewRangeMap`, `viewRanges` props through the recursive tree

**`ModelTraceExplorerSummarySpans` / `ModelTraceExplorerSummaryIntermediateNode`**
- In edit mode: renders checkboxes and drag-to-select on intermediate nodes
- Uses `getTimelineTreeNodesList()` for flattening (no duplicated traversal)

### Backend

No backend changes required. The existing REST API provides all necessary operations:

- `POST /ajax-api/2.0/mlflow/traces/{trace_id}/views` — create view
- `PATCH /ajax-api/2.0/mlflow/traces/{trace_id}/views/{view_id}` — update view
- `GET /ajax-api/2.0/mlflow/traces/{trace_id}/views` — list views (already used by `useTraceViews`)
- `DELETE /ajax-api/2.0/mlflow/traces/{trace_id}/views/{view_id}` — delete view

## Data Flow

```
User drags across spans (edit mode)
  → useSpanRangeSelection captures pointer events
  → Identifies from/to spans by their span_id
  → Calls addRange() on useTraceViewEditMode
  → draftView.ranges updated with new SpanRange
  → Span list re-renders with checkboxes, colors, badges

User clicks ⚙ on a span (edit mode)
  → JsonFieldSelector expands inline beneath span row
  → Renders input/output JSON as checkbox trees
  → User checks fields → JSONPath generated
  → Calls updateRange() with new input_path/output_path
  → draftView.ranges updated

User clicks Save (edit mode)
  → POST or PATCH to REST API with draftView data
  → useInvalidateTraceViews() invalidates cache
  → Edit mode exits

User clicks range badge (view mode)
  → setSelectedViewRangeIdx(rangeIdx) in context
  → Right pane switches to ModelTraceExplorerRangeDetailView
  → Shows range label, description, paths, filtered inputs/outputs

User clicks a span (view mode, with range selected)
  → setSelectedViewRangeIdx(null) clears range selection
  → Right pane returns to normal span tab view
```

## Testing Strategy

- **Unit tests for `useTraceViewFiltering`:** verify `spanMatchesSelector`, `isSpanInRange`, `applyJsonPath`, `applyJsonPathToObject`
- **Unit tests for `useTraceViewEditMode`:** verify add/remove/update/reorder range operations, enter/exit lifecycle
- **Unit tests for `TraceViewSelector`:** verify "+ Create View" and edit icon trigger edit mode, view selection works
- **Integration tests for `TimelineTreeNode`:** verify edit mode checkboxes, drag events, view mode badges
- **Integration tests for `ModelTraceExplorerRightPane`:** verify tab rendering with spans
