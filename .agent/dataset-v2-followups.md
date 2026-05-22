# V2 Datasets UI port — follow-ups

Tracks gaps surfaced while smoke-testing the initial port on branch `dataset-v2-port`. Items
marked `[x]` are addressed in this PR; the rest are scoped to follow-up PRs.

## In this PR

- [x] **#4** Docs link `See SDK docs to add records programmatically` points at the wrong
  URL. Current: `https://mlflow.org/docs/latest/genai/eval-monitor/build-eval-dataset`.
  Should be: `https://mlflow.org/docs/latest/genai/datasets`.
- [x] **#6** Refresh button is redundant in OSS — the OSS create / upsert / delete hooks
  already invalidate the list cache, and OSS's list endpoint isn't rate-limited the way
  universe's managed-evals endpoint is. Remove the explicit refresh button from both
  toolbars (`DatasetsListToolbar`, `DatasetRecordsToolbar`).
- [x] **#13** Hide the `source` column by default in OSS — the underlying field isn't
  populated, so the column renders empty for every row. Removed from
  `DEFAULT_VISIBLE_RECORD_COLUMNS`; users can still toggle it in via the column selector.

## Follow-up PRs

- [ ] **#1** Creating a new dataset does not refresh the list — the row appears only on
  manual refresh. Root cause: `CreateDatasetButton` (our OSS adapter) wraps OSS's
  `CreateEvaluationDatasetModal`, which uses `useCreateEvaluationDatasetMutation` —
  that hook invalidates `SEARCH_EVALUATION_DATASETS_QUERY_KEY` (OSS v1 cache), not the
  v2 cache keys `['v2ListDatasetsPage', ...]` or `['listDatasets', experimentId]`. Fix:
  invalidate v2 keys after the create resolves (either in the adapter, or by replacing
  the wrapped modal with a v2-native create flow that uses `useCreateDatasetMutation`
  from `useDatasetsQueries.tsx`).
- [ ] **#2** After creating a dataset, the UI should open it automatically. Same root
  cause as #1 — the create flow doesn't return the newly-created dataset to the v2
  layer, so the list page has nothing to navigate to. Fix likely lands in the same PR
  as #1: have the create flow return the new dataset, then `navigate()` to its detail
  route.
- [ ] **#3** Empty-state alignment on a single dataset's detail page is too far down
  the viewport. The records empty-state container needs to vertically center within
  the available content area, not just hug the table top.
- [x] **#5** Record `inputs` / `expectations` columns stay stuck at `Loading…` /
  empty. Root cause: the adapter was `JSON.parse`-ing fields that the OSS server
  already returns as dicts (`DatasetRecord.to_dict()` on the server `json.loads`es
  each field before re-serializing the outer envelope). Fixed by switching the
  `OssDatasetRecord` shape from JSON-string fields to dict fields and passing them
  through unchanged. Same fix closes the "Attribute 'inputs' does not accept
  objects of type str" upsert error — write path now sends dicts too.
- [ ] **#7** Multi-format input editor: support JSON / Text / YAML / Pretty (read-only),
  auto-detected from input shape. Today the side panel only does JSON via the Monaco
  editor. Likely involves picking the editor mode based on `inputs.messages` shape
  (chat) vs object (structured) vs string (raw).
- [ ] **#8** Scrollbar layout dirt: on the left pane, the scrollbar floats above the
  bottom of the window with a visible gap. Probably a container with a hard-coded
  bottom padding or an `overflow: auto` on the wrong element. Audit the side panel's
  flex tree.
- [ ] **#9** Records-table rows are visually too tight. Add vertical padding inside
  each cell (likely a Du Bois `Table` density override or per-cell `paddingY`).
- [ ] **#10** Border radius on the side-panel `Inputs` / `Expectations` editor boxes
  is too small. Bump to a larger token (`borderRadiusMd` instead of the current `Sm`).
- [ ] **#11** The `Inputs` / `Expectations` editors are too tall — `Tags` section is
  off-screen on first open of the side panel. Cap the editor's content-driven growth at
  the visible viewport, or fall back to fixed `maxHeight`. The current
  `onDidContentSizeChange` listener in `JsonRecordEditor` grows the wrapper to fit any
  document, which is the immediate culprit.
- [ ] **#12** Tag entry would be much easier in YAML format (`key: value` lines) than
  the current one-tag-at-a-time form. Likely depends on #7 (multi-format editor) since
  YAML support is the same plumbing.
- [ ] **#14** Drop the explicit Save / Discard buttons in the record side panel and
  auto-save instead. Today the flow is: click `Add record` → type → click `Save` →
  drawer closes → click `Add record` again → repeat. The proposed flow: click
  `Add record` once, drawer stays open across record creations, edits debounce-save
  in the background, and clicking `Add record` again just finalizes the current
  record and resets the form to a fresh empty one. Touchpoints: rewrite
  `useRecordCreateState.ts` and `useRecordSaveState.ts` to drop the draft/commit
  boundary; replace the footer buttons with a small saved-indicator; redo the
  invalid-JSON path (today the Save button blocks; auto-save needs to defer or
  show inline). Behavior for switching to a different existing record while one
  has invalid JSON also needs a call.
- [ ] **#15** Records table columns aren't user-resizable. Du Bois `Table` supports
  column resizing via `react-table`'s `enableColumnResizing` + drag handles in the
  header; today every column has a fixed width that ignores content. Add resize
  handles to `DatasetRecordsTable.tsx` and persist widths alongside visibility in
  `usePersistedTableColumns` (or in a sibling hook keyed on the same storage
  version).
- [ ] **#16** Polish the records-table cell hover effect. The hover state itself is
  good, but the show-up transition feels abrupt. Likely tweak the `transition`
  timing / easing on the cell wrapper in `DatasetRecordCell.tsx` (currently no
  explicit transition — relying on default Du Bois hover state).
- [ ] **#17** UI is text-heavy. Replace some button labels / inline text with
  icons (or icon-+-tooltip) where the action is unambiguous: page-level overflow
  actions already use icons, but the records toolbar (`Add record` button), the
  side panel's `Save` / `Discard` (related to #14), and the bulk-action group all
  carry redundant text. Audit which are icon-friendly and which need to stay
  labeled for accessibility.
