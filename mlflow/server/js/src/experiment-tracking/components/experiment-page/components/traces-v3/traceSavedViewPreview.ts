import { useCallback, useMemo } from 'react';
import {
  type EvaluationsOverviewTableSort,
  type TracesTableColumn,
  type TracesTableColumnType,
} from '@databricks/web-shared/genai-traces-table';

const COLUMNS_SEPARATOR = ',';
const SORT_SEPARATOR = '::';

/**
 * Decode the comma-joined column-id list stored in a saved traces view (the `selectedColumns` URL
 * param wire format) into the column objects the table renders. Ids that no longer resolve to a
 * known column are dropped rather than throwing, so a view saved against an older column set still
 * opens. Returns undefined when the value is absent/empty.
 */
export const decodePreviewColumns = (
  raw: string | undefined,
  allColumns: TracesTableColumn[],
): TracesTableColumn[] | undefined => {
  if (!raw) {
    return undefined;
  }
  const byId = new Map(allColumns.map((col) => [col.id, col]));
  const resolved = raw
    .split(COLUMNS_SEPARATOR)
    .filter(Boolean)
    .map((id) => byId.get(id))
    .filter((col): col is TracesTableColumn => Boolean(col));
  // Return undefined (not []) when nothing resolves — an empty set would hide every column. This
  // covers a view saved against an older column schema AND the transient first render where
  // `allColumns` is still empty during async metadata load; the caller falls back to the user's
  // own columns in both cases.
  return resolved.length > 0 ? resolved : undefined;
};

/**
 * Decode the `key::type::asc` sort wire format stored in a saved traces view. Returns undefined when
 * the value is absent or not a well-formed triple.
 */
export const decodePreviewSort = (raw: string | undefined): EvaluationsOverviewTableSort | undefined => {
  if (!raw) {
    return undefined;
  }
  const [key, type, ascStr] = raw.split(SORT_SEPARATOR);
  if (!key || !type || !ascStr) {
    return undefined;
  }
  return { key, type: type as TracesTableColumnType, asc: ascStr === 'true' };
};

/**
 * Preview-override layer for opening a saved traces view.
 *
 * When a saved view is open (`rawColumns`/`rawSort` are the values captured in the view's stored
 * URL state), the decoded columns/sort are held here as the "preview" — the table renders them
 * INSTEAD of the user's own state, but nothing is written to local storage. Only an explicit
 * {@link override} adopts the preview into the user's real state (the sole persistence write);
 * {@link discard} drops the preview and leaves the user's state untouched. This mirrors the runs
 * shared-view Override/Discard model without ever silently clobbering the user's saved columns.
 *
 * `active` is true whenever a view is open; `columns`/`sort` are the decoded preview values the
 * caller should render in place of its own when `active`.
 */
export const useSavedViewPreview = ({
  active,
  rawColumns,
  rawSort,
  allColumns,
  setSelectedColumns,
  setTableSort,
  exitPreview,
}: {
  active: boolean;
  rawColumns: string | undefined;
  rawSort: string | undefined;
  allColumns: TracesTableColumn[];
  setSelectedColumns: (columns: TracesTableColumn[]) => void;
  setTableSort: (sort: EvaluationsOverviewTableSort | undefined) => void;
  exitPreview: () => void;
}) => {
  const columns = useMemo(
    () => (active ? decodePreviewColumns(rawColumns, allColumns) : undefined),
    [active, rawColumns, allColumns],
  );
  const sort = useMemo(() => (active ? decodePreviewSort(rawSort) : undefined), [active, rawSort]);

  // Adopt the preview into the user's own persisted state — the ONLY local-storage write in this
  // flow — then leave preview mode.
  const override = useCallback(() => {
    if (columns) {
      setSelectedColumns(columns);
    }
    if (sort) {
      setTableSort(sort);
    }
    exitPreview();
  }, [columns, sort, setSelectedColumns, setTableSort, exitPreview]);

  // Drop the preview without writing anything; the user's own state resurfaces untouched.
  const discard = useCallback(() => {
    exitPreview();
  }, [exitPreview]);

  return { active, columns, sort, override, discard };
};
