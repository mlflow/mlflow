import {
  Pagination,
  Table,
  TableCell,
  TableHeader,
  TableRow,
  TableRowSelectCell,
  TableSkeleton,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import type { DatasetRecord } from '../hooks/useDatasetsQueries';
import { getCoreRowModel, useReactTable } from '@tanstack/react-table';
import type { ColumnDef, ColumnSizingState, OnChangeFn } from '@tanstack/react-table';
import {
  CreatedByCell,
  CreateTimeCell,
  truncateCss,
  JsonPreviewCell,
  LastUpdatedByCell,
  LastUpdatedCell,
  RecordIdCell,
  SourceCell,
  TagsInlinePreviewBody,
  TagsPreviewCell,
} from './DatasetRecordCell';
import { SORTABLE_RECORD_COLUMNS, type RecordColumnId, type SortDirection } from '../utils/constants';
import type { PendingNewRecord } from '../hooks/useRecordCreateState';

interface DatasetRecordsTableProps {
  records: DatasetRecord[];
  totalRecords: number;
  pageIndex: number;
  pageSize: number;
  onPageChange: (pageIndex: number) => void;
  /**
   * True only during the initial load (no prior data in cache). Background refetches keep
   * the previous rows visible — the underlying query has `keepPreviousData: true`. Skeleton
   * is reserved for the genuine no-data state.
   */
  isLoading: boolean;
  /**
   * True while the underlying records query is fetching (initial load OR refetch). Drives
   * `aria-busy` on the table region so screen readers announce the in-flight refresh while
   * the prior rows stay visible.
   */
  isFetching: boolean;
  onRecordSelected: (record: DatasetRecord) => void;
  selectedRecordId?: string;
  visibleColumns: RecordColumnId[];
  columnSizing: ColumnSizingState;
  setColumnsSizing: OnChangeFn<ColumnSizingState>;
  /** Set of record IDs currently checked for bulk delete. */
  selectedForBulk: Set<string>;
  /** True iff every record currently rendered is in `selectedForBulk`. */
  isAllOnPageSelected: boolean;
  /** True if some but not all rendered records are in `selectedForBulk`. */
  isSomeOnPageSelected: boolean;
  onToggleBulkRow: (recordId: string) => void;
  onToggleBulkAll: () => void;
  /** Active sort column id. */
  sort: RecordColumnId;
  dir: SortDirection;
  onSort: (column: RecordColumnId, direction: SortDirection) => void;
  /**
   * In-progress new record. When non-null, the table renders a synthetic row at the top
   * (key="__new__") that reflects the side panel's live edits. The row is excluded from
   * `selectedForBulk`, pagination math, and sort — it's a UI-only preview. `inputsText` and
   * `expectationsText` carry the raw editor text so the row updates per keystroke even while
   * JSON is partial/invalid; the parsed objects (when valid) are preferred for display so
   * pretty-printed source collapses to compact JSON.
   */
  pendingNewRecord?: PendingNewRecord | null;
}

const SORTABLE_COLUMN_SET = new Set<RecordColumnId>(SORTABLE_RECORD_COLUMNS);

// Default starting widths (px) by content class. Columns are user-resizable, so these
// are only the initial sizes TanStack seeds `columnSizing` with.
const COLUMN_WIDTH = {
  narrow: 120, // source, timestamps
  normal: 160, // ids, usernames, tags
  wide: 400, // inputs / expectations JSON blobs
} as const;

// Fallback when a column id has no entry in the table model — effectively unreachable
// since every column is defined, but `getColumn` is typed as possibly-undefined.
const DEFAULT_COLUMN_WIDTH = COLUMN_WIDTH.normal;

// Base cell styles — column width is supplied separately via flexStyleForColumn.
const cellStyles = { verticalAlign: 'middle' as const };
const stopPropagationProps = {
  // Row click opens the drawer; inner interactive elements (tag pills, etc.) must opt out.
  onClick: (event: React.MouseEvent) => event.stopPropagation(),
};
// TableRowSelectCell ships with vertical-padding tokens and `alignItems: 'start'`
// baked into its internal styles and exposes no prop to flip them. We target the
// design-system's stable `.table-row-select-cell` class from the parent row.
// Rows are single-line, so centering the checkbox vertically lines up with the
// cell text on both the header and the body.
const selectCellHeaderAlign = {
  '.table-row-select-cell': { alignItems: 'center' },
} as const;
const selectCellBodyAlign = {
  '.table-row-select-cell': { alignItems: 'center' },
} as const;
const SKELETON_ROW_COUNT = 5;

export const DatasetRecordsTable = ({
  records,
  totalRecords,
  pageIndex,
  pageSize,
  onPageChange,
  isLoading,
  isFetching,
  onRecordSelected,
  selectedRecordId,
  visibleColumns,
  columnSizing,
  setColumnsSizing,
  selectedForBulk,
  isAllOnPageSelected,
  isSomeOnPageSelected,
  onToggleBulkRow,
  onToggleBulkAll,
  sort,
  dir,
  onSort,
  pendingNewRecord,
}: DatasetRecordsTableProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const isColumnVisible = (id: RecordColumnId) => visibleColumns.includes(id);

  // Sort cycle is 2-state: clicking the active column flips direction, clicking any other
  // sortable column resets to DESC. There's always exactly one active sort.
  const headerSortProps = (id: RecordColumnId) => {
    if (!SORTABLE_COLUMN_SET.has(id)) return {};
    const isActive = sort === id;
    const sortDirection: SortDirection | 'none' = isActive ? dir : 'none';
    return {
      sortable: true,
      sortDirection,
      onToggleSort: () => onSort(id, isActive && dir === 'desc' ? 'asc' : 'desc'),
    };
  };

  const columns: ColumnDef<DatasetRecord>[] = [
    { id: 'dataset_record_id', accessorKey: 'dataset_record_id', header: 'Record ID', size: COLUMN_WIDTH.normal },
    { id: 'inputs', accessorKey: 'inputs', header: 'Inputs', size: COLUMN_WIDTH.wide },
    { id: 'expectations', accessorKey: 'expectations', header: 'Expectations', size: COLUMN_WIDTH.wide },
    { id: 'create_time', accessorKey: 'create_time', header: 'Created', size: COLUMN_WIDTH.narrow },
    { id: 'created_by', accessorKey: 'created_by', header: 'Created by', size: COLUMN_WIDTH.normal },
    { id: 'source', accessorKey: 'source', header: 'Source', size: COLUMN_WIDTH.narrow },
    { id: 'last_updated', accessorKey: 'last_updated', header: 'Last updated', size: COLUMN_WIDTH.narrow },
    { id: 'last_updated_by', accessorKey: 'last_updated_by', header: 'Last updated by', size: COLUMN_WIDTH.normal },
    { id: 'tags', accessorKey: 'tags', header: 'Tags', size: COLUMN_WIDTH.normal },
  ];

  const table = useReactTable({
    data: records,
    columns: columns,
    getCoreRowModel: getCoreRowModel(),
    enableColumnResizing: true,
    columnResizeMode: 'onChange',
    state: {
      columnSizing,
    },
    onColumnSizingChange: setColumnsSizing,
  });

  const headersById = Object.fromEntries(table.getHeaderGroups()[0].headers.map((h) => [h.column.id, h]));

  const flexStyleForColumn = (id: RecordColumnId): React.CSSProperties => {
    const size = table.getColumn(id)?.getSize() ?? DEFAULT_COLUMN_WIDTH;
    return { flex: `0 0 ${size}px` };
  };

  return (
    <div
      css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}
      role="region"
      aria-busy={isFetching}
      aria-label={intl.formatMessage({
        defaultMessage: 'Dataset records',
        description: 'Region label for the V2 dataset records table — wraps the table and pagination',
      })}
    >
      {/* `someRowsSelected` controls the table's `--row-checkbox-opacity` token,
       * which is what keeps per-row checkboxes visible. The strict
       * `isSomeOnPageSelected` flips to false when *every* row is selected, so
       * we OR it with `isAllOnPageSelected` to keep the visual ticked through
       * the "all selected" state. The strict prop is still passed below for
       * the header's indeterminate UI. */}
      <Table someRowsSelected={isAllOnPageSelected || isSomeOnPageSelected}>
        <TableRow isHeader css={selectCellHeaderAlign}>
          <TableRowSelectCell
            componentId="mlflow.eval-datasets-v2.records.row-select-all"
            checked={isAllOnPageSelected}
            indeterminate={isSomeOnPageSelected && !isAllOnPageSelected}
            onChange={onToggleBulkAll}
            checkboxLabel={intl.formatMessage({
              defaultMessage: 'Select all rows on this page',
              description: 'Aria label for the select-all checkbox in the V2 dataset records table header',
            })}
          />
          {isColumnVisible('dataset_record_id') && (
            <TableHeader
              componentId="mlflow.eval-datasets-v2.records.header.record-id"
              style={flexStyleForColumn('dataset_record_id')}
              {...headerSortProps('dataset_record_id')}
              header={headersById['dataset_record_id']}
              column={headersById['dataset_record_id']?.column}
              setColumnSizing={table.setColumnSizing}
            >
              <FormattedMessage defaultMessage="Record ID" description="Header for the dataset record id column" />
            </TableHeader>
          )}
          {isColumnVisible('inputs') && (
            <TableHeader
              componentId="mlflow.eval-datasets-v2.records.header.inputs"
              style={flexStyleForColumn('inputs')}
              header={headersById['inputs']}
              column={headersById['inputs']?.column}
              setColumnSizing={table.setColumnSizing}
            >
              <FormattedMessage defaultMessage="Inputs" description="Header for the dataset record inputs column" />
            </TableHeader>
          )}
          {isColumnVisible('expectations') && (
            <TableHeader
              componentId="mlflow.eval-datasets-v2.records.header.expectations"
              style={flexStyleForColumn('expectations')}
              header={headersById['expectations']}
              column={headersById['expectations']?.column}
              setColumnSizing={table.setColumnSizing}
            >
              <FormattedMessage
                defaultMessage="Expectations"
                description="Header for the dataset record expectations column"
              />
            </TableHeader>
          )}
          {isColumnVisible('create_time') && (
            <TableHeader
              componentId="mlflow.eval-datasets-v2.records.header.create-time"
              style={flexStyleForColumn('create_time')}
              {...headerSortProps('create_time')}
              header={headersById['create_time']}
              column={headersById['create_time']?.column}
              setColumnSizing={table.setColumnSizing}
            >
              <FormattedMessage
                defaultMessage="Created"
                description="Header for the dataset record create-time column"
              />
            </TableHeader>
          )}
          {isColumnVisible('created_by') && (
            <TableHeader
              componentId="mlflow.eval-datasets-v2.records.header.created-by"
              style={flexStyleForColumn('created_by')}
              {...headerSortProps('created_by')}
              header={headersById['created_by']}
              column={headersById['created_by']?.column}
              setColumnSizing={table.setColumnSizing}
            >
              <FormattedMessage
                defaultMessage="Created by"
                description="Header for the dataset record created-by column"
              />
            </TableHeader>
          )}
          {isColumnVisible('source') && (
            <TableHeader
              componentId="mlflow.eval-datasets-v2.records.header.source"
              style={flexStyleForColumn('source')}
              header={headersById['source']}
              column={headersById['source']?.column}
              setColumnSizing={table.setColumnSizing}
            >
              <FormattedMessage defaultMessage="Source" description="Header for the dataset record source column" />
            </TableHeader>
          )}
          {isColumnVisible('last_updated') && (
            <TableHeader
              componentId="mlflow.eval-datasets-v2.records.header.last-updated"
              style={flexStyleForColumn('last_updated')}
              {...headerSortProps('last_updated')}
              header={headersById['last_updated']}
              column={headersById['last_updated']?.column}
              setColumnSizing={table.setColumnSizing}
            >
              <FormattedMessage
                defaultMessage="Last updated"
                description="Header for the dataset record last-updated column"
              />
            </TableHeader>
          )}
          {isColumnVisible('last_updated_by') && (
            <TableHeader
              componentId="mlflow.eval-datasets-v2.records.header.last-updated-by"
              style={flexStyleForColumn('last_updated_by')}
              {...headerSortProps('last_updated_by')}
              header={headersById['last_updated_by']}
              column={headersById['last_updated_by']?.column}
              setColumnSizing={table.setColumnSizing}
            >
              <FormattedMessage
                defaultMessage="Last updated by"
                description="Header for the dataset record last-updated-by column"
              />
            </TableHeader>
          )}
          {isColumnVisible('tags') && (
            <TableHeader
              componentId="mlflow.eval-datasets-v2.records.header.tags"
              style={flexStyleForColumn('tags')}
              header={headersById['tags']}
              column={headersById['tags']?.column}
              setColumnSizing={table.setColumnSizing}
            >
              <FormattedMessage defaultMessage="Tags" description="Header for the dataset record tags column" />
            </TableHeader>
          )}
        </TableRow>
        {isLoading
          ? Array.from({ length: SKELETON_ROW_COUNT }, (_, i) => (
              <TableRow key={`skeleton-${i}`} css={selectCellBodyAlign}>
                <TableCell css={cellStyles}>
                  <TableSkeleton seed={`records-select-${i}`} />
                </TableCell>
                {visibleColumns.map((col) => (
                  <TableCell key={col} css={cellStyles} style={flexStyleForColumn(col)}>
                    <TableSkeleton seed={`records-${col}-${i}`} />
                  </TableCell>
                ))}
              </TableRow>
            ))
          : null}
        {!isLoading && pendingNewRecord && (
          // Synthetic "new record" row. Lives outside `records` so it's invisible to
          // pagination, sort, and bulk selection — pure UI preview wired to the side
          // panel's editor state. Uses `<TableRowSelectCell noCheckbox>` for the leading
          // cell so its `flex: 0` and fixed checkbox-column width match the data rows;
          // an empty `<TableCell>` here would inherit `flex: 1` and shift every column
          // out of alignment with the rows above and below.
          <TableRow
            key="__new__"
            css={{
              backgroundColor: theme.colors.tableBackgroundUnselectedHover,
              ...selectCellBodyAlign,
            }}
          >
            <TableRowSelectCell componentId="mlflow.eval-datasets-v2.records.row-select.phantom" noCheckbox />
            {visibleColumns.map((col) => {
              // Phantom row only. `wrapContent={false}` skips the DS TableCell's auto
              // Typography.Text wrapper (we render our own), and the cell wrapper below
              // forces `text-align: left` + `justify-items: start` so the short
              // "(empty)" / "-" placeholders hug the left edge of the cell instead of
              // floating in the middle. Data rows wrap their content in inline-block
              // activator components that already render flush-left, so they don't need
              // this override.
              const colCellStyles = {
                ...cellStyles,
                textAlign: 'left' as const,
                justifyItems: 'start' as const,
              };
              if (col === 'inputs' || col === 'expectations') {
                const value = col === 'inputs' ? pendingNewRecord.inputs : pendingNewRecord.expectations;
                const text = col === 'inputs' ? pendingNewRecord.inputsText : pendingNewRecord.expectationsText;
                const hasValidContent = value !== undefined && Object.keys(value).length > 0;
                const trimmedText = text.trim();
                const isEmpty = !hasValidContent && trimmedText === '';
                // Prefer compact JSON for valid input (collapses pretty-printed source onto one
                // line); otherwise echo the raw text so each keystroke shows up immediately.
                const display = hasValidContent ? JSON.stringify(value) : trimmedText;
                return (
                  <TableCell
                    key={col}
                    css={colCellStyles}
                    style={flexStyleForColumn(col)}
                    align="left"
                    wrapContent={false}
                  >
                    {isEmpty ? (
                      <Typography.Text color="secondary" size="sm">
                        <FormattedMessage defaultMessage="(empty)" description="Placeholder for an empty JSON cell" />
                      </Typography.Text>
                    ) : (
                      <span
                        css={[
                          truncateCss,
                          { fontFamily: 'monospace', fontSize: theme.typography.fontSizeSm, display: 'block' },
                        ]}
                      >
                        {display}
                      </span>
                    )}
                  </TableCell>
                );
              }
              if (col === 'tags') {
                // Read-only preview of the draft tags. Phantom row has no record yet, so it
                // skips the saved-row activator + tooltip and renders the same inline
                // pill + "+N more" body as `TagsPreviewCell` directly.
                const tagEntries = Object.entries(pendingNewRecord.tags);
                if (tagEntries.length === 0) {
                  return (
                    <TableCell
                      key={col}
                      css={colCellStyles}
                      style={flexStyleForColumn(col)}
                      align="left"
                      wrapContent={false}
                    >
                      <Typography.Text color="secondary">-</Typography.Text>
                    </TableCell>
                  );
                }
                return (
                  <TableCell
                    key={col}
                    css={colCellStyles}
                    style={flexStyleForColumn(col)}
                    align="left"
                    wrapContent={false}
                  >
                    <TagsInlinePreviewBody
                      entries={tagEntries}
                      componentId="mlflow.eval-datasets-v2.records.tag.phantom-pill"
                    />
                  </TableCell>
                );
              }
              return (
                <TableCell
                  key={col}
                  css={colCellStyles}
                  style={flexStyleForColumn(col)}
                  align="left"
                  wrapContent={false}
                >
                  <Typography.Text color="secondary">-</Typography.Text>
                </TableCell>
              );
            })}
          </TableRow>
        )}
        {!isLoading &&
          records.map((record) => {
            const isSelected = record.dataset_record_id === selectedRecordId;
            const isBulkChecked = selectedForBulk.has(record.dataset_record_id);
            const openDrawer = () => onRecordSelected(record);
            const recordIdLabel = intl.formatMessage(
              {
                defaultMessage: 'Open dataset record {recordId}',
                description: 'Aria label for the record-id activator in the V2 dataset records table',
              },
              { recordId: record.dataset_record_id },
            );
            const inputsLabel = intl.formatMessage(
              {
                defaultMessage: 'Open dataset record {recordId} — inputs',
                description: 'Aria label for the inputs JSON-preview cell that opens the record drawer',
              },
              { recordId: record.dataset_record_id },
            );
            const expectationsLabel = intl.formatMessage(
              {
                defaultMessage: 'Open dataset record {recordId} — expectations',
                description: 'Aria label for the expectations JSON-preview cell that opens the record drawer',
              },
              { recordId: record.dataset_record_id },
            );
            const tagsLabel = intl.formatMessage(
              {
                defaultMessage: 'Open dataset record {recordId} — tags',
                description: 'Aria label for the tags-preview cell that opens the record drawer',
              },
              { recordId: record.dataset_record_id },
            );
            return (
              <TableRow
                key={record.dataset_record_id}
                onClick={openDrawer}
                css={{
                  cursor: 'pointer',
                  backgroundColor: isSelected ? theme.colors.tableBackgroundUnselectedHover : undefined,
                  ...selectCellBodyAlign,
                }}
              >
                <TableRowSelectCell
                  componentId="mlflow.eval-datasets-v2.records.row-select"
                  checked={isBulkChecked}
                  onChange={() => onToggleBulkRow(record.dataset_record_id)}
                  checkboxLabel={intl.formatMessage(
                    {
                      defaultMessage: 'Select record {recordId}',
                      description: 'Aria label for the per-row select checkbox in the V2 dataset records table',
                    },
                    { recordId: record.dataset_record_id },
                  )}
                  {...stopPropagationProps}
                />
                {isColumnVisible('dataset_record_id') && (
                  <TableCell css={cellStyles} style={flexStyleForColumn('dataset_record_id')}>
                    <RecordIdCell record={record} onActivate={openDrawer} accessibleLabel={recordIdLabel} />
                  </TableCell>
                )}
                {isColumnVisible('inputs') && (
                  <TableCell css={cellStyles} style={flexStyleForColumn('inputs')}>
                    <JsonPreviewCell
                      value={record.inputs}
                      emptyLabel={
                        <FormattedMessage defaultMessage="(empty)" description="Placeholder for an empty JSON cell" />
                      }
                      onActivate={openDrawer}
                      accessibleLabel={inputsLabel}
                    />
                  </TableCell>
                )}
                {isColumnVisible('expectations') && (
                  <TableCell css={cellStyles} style={flexStyleForColumn('expectations')}>
                    <JsonPreviewCell
                      value={record.expectations}
                      emptyLabel={
                        <FormattedMessage defaultMessage="(empty)" description="Placeholder for an empty JSON cell" />
                      }
                      onActivate={openDrawer}
                      accessibleLabel={expectationsLabel}
                    />
                  </TableCell>
                )}
                {isColumnVisible('create_time') && (
                  <TableCell css={cellStyles} style={flexStyleForColumn('create_time')}>
                    <CreateTimeCell record={record} />
                  </TableCell>
                )}
                {isColumnVisible('created_by') && (
                  <TableCell css={cellStyles} style={flexStyleForColumn('created_by')}>
                    <CreatedByCell record={record} />
                  </TableCell>
                )}
                {isColumnVisible('source') && (
                  <TableCell css={cellStyles} style={flexStyleForColumn('source')}>
                    <SourceCell record={record} />
                  </TableCell>
                )}
                {isColumnVisible('last_updated') && (
                  <TableCell css={cellStyles} style={flexStyleForColumn('last_updated')}>
                    <LastUpdatedCell record={record} />
                  </TableCell>
                )}
                {isColumnVisible('last_updated_by') && (
                  <TableCell css={cellStyles} style={flexStyleForColumn('last_updated_by')}>
                    <LastUpdatedByCell record={record} />
                  </TableCell>
                )}
                {isColumnVisible('tags') && (
                  <TableCell css={cellStyles} style={flexStyleForColumn('tags')}>
                    <TagsPreviewCell tags={record.tags} onActivate={openDrawer} accessibleLabel={tagsLabel} />
                  </TableCell>
                )}
              </TableRow>
            );
          })}
      </Table>
      {totalRecords > pageSize && (
        <div css={{ display: 'flex', justifyContent: 'flex-end' }}>
          <Pagination
            componentId="mlflow.eval-datasets-v2.records.pagination"
            currentPageIndex={pageIndex}
            pageSize={pageSize}
            numTotal={totalRecords}
            onChange={(nextPage) => onPageChange(nextPage)}
          />
        </div>
      )}
    </div>
  );
};
