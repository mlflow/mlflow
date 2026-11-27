# traces-table/data — opt-in fetch layer

The reusable, cursor-paginated trace-fetch mechanism (`useTracesPageQuery` + `useTraceTokenCache`,
built on `searchTracesLongRunningPage`). It is **opt-in and decoupled**: the presentational layer
(`TracesTable`, toolbar, states, `TracesTableView`) imports **nothing** from `data/`. A consumer with
a different backend ignores these hooks and feeds the table from its own source.

## The contract (keep it, or the layer stops being reusable)

`useTracesPageQuery` is a fetch *mechanism*, not a controller. It must stay this way:

- **Opaque filter.** `identity.filter` is a server-clause string the consumer builds and compiles.
  The hook never parses, builds, or compiles it.
- **No product concepts.** `identity.sqlWarehouseId` is forwarded verbatim when present; the hook
  doesn't know what a SQL warehouse is. Don't add warehouse/monitoring/experiment logic here.
- **No URL/page ownership.** The consumer owns which page is current and passes `pageIndex` +
  `onPageIndexChange`. The hook returns data + cursor affordances only.
- **Consumer owns `enabled`.** e.g. MLflow disables the query when no warehouse is selected.

The risk with a shared fetch hook is that it slowly absorbs product logic and stops being reusable.
The structural guardrail is that the presentational files import nothing from here; the behavioral
guardrail is the contract above.

## Transport

Three transports return the same `{ trace_infos, next_page_token }` shape, built on web-shared's own
`fetchAPI`/`getAjaxUrl` (`model-trace-explorer`) so the standard endpoints need no injection:

- synchronous `search` and async initiate→poll `search-long-running` — chosen by
  `shouldUseLongRunningTracesAPI` (`searchTracesLongRunningPage.ts`).
- **progressive** (`search-progressive` + `.../operations`, `searchTracesProgressivePage.ts`) — used
  when the consumer passes `useProgressiveSearch: true`. Eligibility (a V2 trace table) is **passed
  in, not derived here** — keeping the schema-versioning decision product-side. It loops
  initiate→poll, accumulating partial batches until the page is full or the search is exhausted; the
  hook still sees one logical page. There is **no cancel endpoint**, so abort is cooperative (stop
  looping; the statement is abandoned server-side).

Shared poll/delay/error-mapping helpers live in `longRunningOperation.ts`.
