# Trace View Editor UI Implementation Plan

**Goal:** Enable users to create and edit trace views from the MLflow UI, with drag-to-select span ranges, checkbox-based JSON field extraction, and inline range management.

**Architecture:** Edit mode layered onto the existing trace explorer detail/timeline view. A `useTraceViewEditMode` hook manages draft state; `useSpanRangeSelection` handles drag selection and visual feedback; `JsonFieldSelector` renders checkbox trees for input/output path configuration. Mutations use the existing REST API via React Query `useMutation`. View mode shows range badges and highlighting using shared `isSpanInRange` logic.

**Tech Stack:** React 18, TypeScript, @databricks/design-system, @tanstack/react-query, jsonpath-plus

---

## Completed Tasks

### Task 1: API Mutation Hooks ✅
- Created `hooks/useTraceViewMutations.ts` with `useCreateTraceView`, `useUpdateTraceView`, `useDeleteTraceView`
- All invalidate query cache on success via `useInvalidateTraceViews`

### Task 2: Range Color Palette ✅
- Created `edit-mode/rangeColors.ts` with 6-color palette and `getRangeColor(index)`

### Task 3: useTraceViewEditMode Hook ✅
- Created `hooks/useTraceViewEditMode.ts` managing draft view state
- Exposes: `isEditMode`, `draftView`, `enterEditMode`, `exitEditMode`, `addRange`, `removeRange`, `updateRange`, `reorderRanges`, `setName`

### Task 4: JsonFieldSelector Component ✅
- Created `edit-mode/JsonFieldSelector.tsx` with checkbox tree and JSONPath generation
- Supports nested objects, arrays, leaf previews, and pre-checked state

### Task 5: TraceViewEditToolbar Component ✅
- Created `edit-mode/TraceViewEditToolbar.tsx` with name input, Cancel, Save buttons
- Save disabled when name empty or no ranges

### Task 6: SpanRangeOverlay → useSpanRangeSelection ✅
- Originally planned as `SpanRangeOverlay` component, implemented as `useSpanRangeSelection` hook instead
- Provides drag-to-select, checkbox toggle, per-node edit state (`SpanEditState`)
- Created `edit-mode/RangeBadge.tsx` for range labels with optional delete/click/selected states

### Task 7: Edit Mode Context Integration ✅
- Added `editMode` to `ModelTraceExplorerViewStateContext`
- Added `selectedViewRangeIdx` / `setSelectedViewRangeIdx` for view-mode range selection

### Task 8: TraceViewSelector Entry Points ✅
- Added "+ Create View" menu item to dropdown
- Added edit icon (pencil) on active view

### Task 9: Timeline Tree Edit Mode Integration ✅
- `TimelineTree` passes `editModeProps` to `TimelineTreeNode`
- `TimelineTreeNode` renders checkboxes, drag events, gear icon, RangeBadge in edit mode
- Edit toolbar rendered in both detail view and summary view

### Task 10: Summary View Edit Mode Integration ✅
- `ModelTraceExplorerSummarySpans` renders edit toolbar and checkboxes on intermediate nodes
- `ModelTraceExplorerSummaryIntermediateNode` supports edit state, drag events, gear icon

### Task 11: Refactor — Shared Range Matching Logic ✅
- Extracted `isSpanInRange()` into `useTraceViewFiltering.ts` as shared function
- `findRangeForSpan` in `useSpanRangeSelection` delegates to `isSpanInRange`
- `useTraceViewSpanMatches` uses `isSpanInRange` for full from→to range matching (was only matching `from_selector`)
- All span matching uses `spanMatchesSelector()` (supports span_id, name, type, attributes)
- Tree flattening uses `getTimelineTreeNodesList()` — no duplicated traversal logic

### Task 12: View Mode — Range Badges & Highlighting in Timeline ✅
- `useTraceViewSpanMatches` returns `{ matchedKeys, rangeMap }` with per-span range info (`rangeIdx`, `isFirstInRange`)
- `TimelineTreeNode` renders colored borders, backgrounds, and `RangeBadge` labels in view mode
- `RangeBadge` `onDelete` made optional (no delete button in view mode)
- New props `viewRangeMap` and `viewRanges` passed through recursive tree

### Task 13: View Mode — Selectable Range Badges & Right Pane Detail ✅
- Removed "Viewing: {name}" banner bar from detail view
- `RangeBadge` gained `onClick` and `isSelected` props for view mode interactivity
- `selectedViewRangeIdx` in context tracks which range badge is selected
- Created `right-pane/ModelTraceExplorerRangeDetailView.tsx` showing range label, description, input/output paths, and filtered span data
- `ModelTraceExplorerRightPaneTabs` renders range detail view when a range is selected
- Clicking a span clears range selection, returning to normal tab view

---

## File Inventory

### New Files
| File | Status |
|------|--------|
| `hooks/useTraceViewMutations.ts` | ✅ |
| `hooks/useTraceViewEditMode.ts` | ✅ |
| `edit-mode/TraceViewEditToolbar.tsx` | ✅ |
| `edit-mode/useSpanRangeSelection.ts` | ✅ |
| `edit-mode/RangeBadge.tsx` | ✅ |
| `edit-mode/JsonFieldSelector.tsx` | ✅ |
| `edit-mode/rangeColors.ts` | ✅ |
| `right-pane/ModelTraceExplorerRangeDetailView.tsx` | ✅ |

### Modified Files
| File | Status |
|------|--------|
| `TraceViewSelector.tsx` | ✅ |
| `ModelTraceExplorerViewStateContext.tsx` | ✅ |
| `ModelTraceExplorerDetailView.tsx` | ✅ |
| `timeline-tree/TimelineTree.tsx` | ✅ |
| `timeline-tree/TimelineTreeNode.tsx` | ✅ |
| `summary-view/ModelTraceExplorerSummarySpans.tsx` | ✅ |
| `summary-view/ModelTraceExplorerSummaryIntermediateNode.tsx` | ✅ |
| `summary-view/ModelTraceExplorerCompactSummaryView.tsx` | ✅ |
| `summary-view/ModelTraceExplorerSummaryView.tsx` | ✅ |
| `hooks/useTraceViewFiltering.ts` | ✅ |
| `right-pane/ModelTraceExplorerRightPaneTabs.tsx` | ✅ |

---

## Commit History

| Commit | Description |
|--------|-------------|
| `f12184b82` | feat: add TraceViewEditToolbar with name input and save/cancel buttons |
| `4a3897259` | feat: add SpanRangeOverlay with drag-to-select, checkboxes, range badges, and inline JSON config |
| `8a277dd12` | feat: add edit mode state to trace explorer view context |
| `a9abbb3c4` | feat: add Create View and Edit entry points to TraceViewSelector |
| `fda72fb62` | feat: integrate trace view edit mode into summary spans view |
| `73b67bd55` | refactor: integrate edit mode as overlay into existing trace views |
| `8bf9f54f1` | refactor: reuse spanMatchesSelector and tree utils in edit mode selection |
| `beed2fcb7` | feat: show span range badges and highlighting in timeline view mode |
| `17c19f73e` | feat: selectable range badges with detail view in right pane |
