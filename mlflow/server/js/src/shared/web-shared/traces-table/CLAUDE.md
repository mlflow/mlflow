# traces-table (`@databricks/web-shared/traces-table`)

A **dumb, fully-controlled, presentational** traces table + controls. It owns **no data, no URL
state, and no product coupling**. Every piece of state (search text, filter model, sort, visible
columns, column sizing, bulk selection, pagination) is owned by the consumer and passed in with
change callbacks. This is deliberately the opposite of `genai-traces-table`, which is feature-rich
but tightly coupled.

## What lives here vs. the consumer

- **Here (presentational):** `TracesTable` (the dumb TanStack table), `TracesTableToolbar`
  (layout shell + built-in search), `TracesPaginationBar`, `TraceFilterButton`,
  `TraceColumnSelector`, the state components (`TracesEmptyState`, `TracesNoResultsState`, …), and
  the `TracesTableView` convenience wrapper that composes them by a consumer-computed `viewState`.
- **Consumer-owned:** data fetching, URL/query state, which page is current, the filter *model* and
  its compilation into a server filter string, SQL-warehouse / monitoring logic, delete/bulk
  mutations, the trace drawer, and time-range/refresh semantics.

## Opt-in data layer (`data/`)

Most consumers fetch traces the same way (cursor-paginated `ajax-api/4.0/mlflow/traces/search`), so
that logic ships as an **opt-in** layer: `useTracesPageQuery` + `useTraceTokenCache`. The
presentational files import **nothing** from `data/` — that separation is the structural guarantee
the table stays reusable with any backend. The hook takes an opaque, consumer-built `filter` string
+ an optional `sqlWarehouseId` + an `enabled` flag and returns data only; it never compiles filters,
owns URL state, or knows product concepts.

## Conventions & gotchas

- **Module-scope columns.** `STANDARD_COLUMNS` is defined once at module scope and never rebuilt —
  this is what keeps `React.memo` + `useReactTableWithDeepMemo` effective. Per-render params (`intl`,
  `onTraceSelected`, `getSessionHref`, `onFilterByTag`) reach cells via table `meta`, never baked
  into the column defs. Per-trace closures are built *inside* cells.
- **componentId is static.** The `@databricks/no-dynamic-property-value` lint rule requires every DS
  `componentId` to be statically determinable, so there is **no runtime `componentIdPrefix`** — each
  file uses a module-local `const COMPONENT_ID = 'web-shared.traces-table'`. (The migrated MLflow tab
  therefore re-keys its analytics ids from `mlflow.traces-v4.*` to `web-shared.traces-table.*`.)
- **Adding a column.** Extend `TRACE_COLUMN_IDS` + `COLUMN_SIZES` + `STANDARD_COLUMNS`, then add a
  cell in `TraceCell.tsx`. Product-specific columns go through `extraColumns` on `TracesTable`
  instead (appended after the standard columns). Note: an `extraColumns` column can't join the typed
  `visibleColumns` union — it's always-on or gated by the consumer.
- **Session link.** The only product coupling in the presentational layer is the session cell's
  *URL*. Pass `getSessionHref?: ({trace, sessionId}) => To | undefined`; when it returns a `To` the
  shared cell wraps its `Tag` in a `Link`, otherwise it renders plain text. There is no render-prop
  escape hatch — the visual is fixed on purpose.
- **Filter builder.** `TraceFilterButton` is parameterized by `fields: FilterFieldDef[]` and owns the
  draft/apply UX + the neutral `TraceFilterModel` AST. Server-clause compilation is the consumer's.
- **Column persistence hooks** (`useTraceColumnVisibility`, `useTraceColumnSizing`) take a
  `storageKey` + `version` so the consumer controls the localStorage namespace and reset semantics.
- **Optional seams degrade silently.** `getSessionHref`, `onFilterByTag`, and `getErrorDescription`
  are all optional and *load-bearing*: omit them and the surface still renders but quietly loses
  behavior — a plain-text (non-linked) session cell, non-clickable tags, and a generic load-error
  message with no backend-specific hint (e.g. MLflow's SQL-warehouse-timeout CTA). MLflow wires all
  three; a new consumer that skips one gets the degraded path with no type error.
