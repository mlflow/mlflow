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
import {
  CreatedByCell,
  CreateTimeCell,
  JsonPreviewCell,
  LastUpdatedByCell,
  LastUpdatedCell,
  RecordIdCell,
  SourceCell,
  TagsPreviewCell,
} from './DatasetRecordCell';
import { SORTABLE_RECORD_COLUMNS, type RecordColumnId, type SortDirection } from '../utils/constants';

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
}

const SORTABLE_COLUMN_SET = new Set<RecordColumnId>(SORTABLE_RECORD_COLUMNS);

const cellStylesForColumn = (id: RecordColumnId) => {
  if (id === 'inputs' || id === 'expectations') return wideCellStyles;
  if (id === 'source') return narrowCellStyles;
  if (id === 'tags') return tagsCellStyles;
  return cellStyles;
};

const cellStyles = { verticalAlign: 'middle' as const };
// Inputs/expectations carry the bulk of each record's payload — give their
// cells (and headers) more horizontal room than the metadata columns. Source
// shrinks because it renders a single short Tag. Tags sits slightly above
// the default to fit the first-tag pill plus a "+N more" suffix without
// crowding. Applied symmetrically to header and cell so columns line up.
const wideCellStyles = { ...cellStyles, flex: 2.5 };
const WIDE_HEADER_STYLE = { flex: 2.5 } as const;
const narrowCellStyles = { ...cellStyles, flex: 0.5 };
const NARROW_HEADER_STYLE = { flex: 0.5 } as const;
const tagsCellStyles = { ...cellStyles, flex: 1.5 };
const TAGS_HEADER_STYLE = { flex: 1.5 } as const;
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
  selectedForBulk,
  isAllOnPageSelected,
  isSomeOnPageSelected,
  onToggleBulkRow,
  onToggleBulkAll,
  sort,
  dir,
  onSort,
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
              {...headerSortProps('dataset_record_id')}
            >
              <FormattedMessage defaultMessage="Record ID" description="Header for the dataset record id column" />
            </TableHeader>
          )}
          {isColumnVisible('inputs') && (
            <TableHeader componentId="mlflow.eval-datasets-v2.records.header.inputs" style={WIDE_HEADER_STYLE}>
              <FormattedMessage defaultMessage="Inputs" description="Header for the dataset record inputs column" />
            </TableHeader>
          )}
          {isColumnVisible('expectations') && (
            <TableHeader componentId="mlflow.eval-datasets-v2.records.header.expectations" style={WIDE_HEADER_STYLE}>
              <FormattedMessage
                defaultMessage="Expectations"
                description="Header for the dataset record expectations column"
              />
            </TableHeader>
          )}
          {isColumnVisible('create_time') && (
            <TableHeader
              componentId="mlflow.eval-datasets-v2.records.header.create-time"
              {...headerSortProps('create_time')}
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
              {...headerSortProps('created_by')}
            >
              <FormattedMessage
                defaultMessage="Created by"
                description="Header for the dataset record created-by column"
              />
            </TableHeader>
          )}
          {isColumnVisible('source') && (
            <TableHeader componentId="mlflow.eval-datasets-v2.records.header.source" style={NARROW_HEADER_STYLE}>
              <FormattedMessage defaultMessage="Source" description="Header for the dataset record source column" />
            </TableHeader>
          )}
          {isColumnVisible('last_updated') && (
            <TableHeader
              componentId="mlflow.eval-datasets-v2.records.header.last-updated"
              {...headerSortProps('last_updated')}
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
              {...headerSortProps('last_updated_by')}
            >
              <FormattedMessage
                defaultMessage="Last updated by"
                description="Header for the dataset record last-updated-by column"
              />
            </TableHeader>
          )}
          {isColumnVisible('tags') && (
            <TableHeader componentId="mlflow.eval-datasets-v2.records.header.tags" style={TAGS_HEADER_STYLE}>
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
                  <TableCell key={col} css={cellStylesForColumn(col)}>
                    <TableSkeleton seed={`records-${col}-${i}`} />
                  </TableCell>
                ))}
              </TableRow>
            ))
          : null}
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
                  <TableCell css={cellStylesForColumn('dataset_record_id')}>
                    <RecordIdCell record={record} onActivate={openDrawer} accessibleLabel={recordIdLabel} />
                  </TableCell>
                )}
                {isColumnVisible('inputs') && (
                  <TableCell css={cellStylesForColumn('inputs')}>
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
                  <TableCell css={cellStylesForColumn('expectations')}>
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
                  <TableCell css={cellStylesForColumn('create_time')}>
                    <CreateTimeCell record={record} />
                  </TableCell>
                )}
                {isColumnVisible('created_by') && (
                  <TableCell css={cellStylesForColumn('created_by')}>
                    <CreatedByCell record={record} />
                  </TableCell>
                )}
                {isColumnVisible('source') && (
                  <TableCell css={cellStylesForColumn('source')}>
                    <SourceCell record={record} />
                  </TableCell>
                )}
                {isColumnVisible('last_updated') && (
                  <TableCell css={cellStylesForColumn('last_updated')}>
                    <LastUpdatedCell record={record} />
                  </TableCell>
                )}
                {isColumnVisible('last_updated_by') && (
                  <TableCell css={cellStylesForColumn('last_updated_by')}>
                    <LastUpdatedByCell record={record} />
                  </TableCell>
                )}
                {isColumnVisible('tags') && (
                  <TableCell css={cellStylesForColumn('tags')}>
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
