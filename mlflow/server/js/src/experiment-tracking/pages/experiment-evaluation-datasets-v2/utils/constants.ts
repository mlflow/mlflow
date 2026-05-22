/** Search-input debounce. 250ms balances snappy feedback with cutting wasted backend calls. */
export const SEARCH_DEBOUNCE_MS = 250;

/** Datasets list page size. Cap is 50 — managed-evals ListDatasets caps page_size at 50. */
export const DEFAULT_DATASET_PAGE_SIZE = 20;

/** Records page size. ListDatasetRecords supports up to 100. */
export const DEFAULT_RECORD_PAGE_SIZE = 25;

/** Default order for the datasets list (most recently created first). */
export const DEFAULT_DATASETS_ORDER_BY = 'created_time DESC';

/**
 * Every column the records table can surface. Order doubles as the canonical render order
 * — `usePersistedTableColumns` preserves it when toggling.
 *
 * Adding a column id here is safe without bumping `COLUMN_STORAGE_VERSION`: the persistence
 * hook filters unknown ids at read time, so older stored visibility sets keep working and
 * new ids are simply unselected until the user toggles them.
 */
export const RECORD_COLUMN_IDS = [
  'dataset_record_id',
  'inputs',
  'expectations',
  'create_time',
  'created_by',
  'source',
  'last_updated',
  'last_updated_by',
  'tags',
] as const;
export type RecordColumnId = (typeof RECORD_COLUMN_IDS)[number];

/** Columns visible by default. Users can toggle the rest in via the column selector. */
export const DEFAULT_VISIBLE_RECORD_COLUMNS: readonly RecordColumnId[] = [
  'inputs',
  'expectations',
  'source',
  'last_updated',
  'tags',
];

export type SortDirection = 'asc' | 'desc';

/**
 * Columns the user can sort by. JSON, tag-map, and source-struct columns are excluded —
 * there is no canonical primitive comparator for structured data. Keeping this in one
 * place so the URL state, query, and table render layers can't disagree about what's
 * sortable.
 */
export const SORTABLE_RECORD_COLUMNS: readonly RecordColumnId[] = [
  'dataset_record_id',
  'create_time',
  'created_by',
  'last_updated',
  'last_updated_by',
];

/** Default sort matches V1's behavior: most recently edited first. */
export const DEFAULT_SORT_COLUMN: RecordColumnId = 'last_updated';
export const DEFAULT_SORT_DIR: SortDirection = 'desc';
