import {
  Button,
  CheckIcon,
  ChevronDownIcon,
  DropdownMenu,
  FilterIcon,
  SortAscendingIcon,
  SortDescendingIcon,
  VisibleOffIcon,
} from '@databricks/design-system';
import type { Column } from '@tanstack/react-table';
import { useRef } from 'react';
import { FormattedMessage, useIntl, type IntlShape } from 'react-intl';
import type { RunEntityOrGroupData } from './ExperimentEvaluationRunsPage.utils';
import { parseEvalRunsTableKeyedColumnKey } from './ExperimentEvaluationRunsTable.utils';
import { EVAL_RUNS_COLUMN_LABELS, type EvalRunsTableColumnId } from './ExperimentEvaluationRunsTable.constants';

// Screen-reader label differentiates per-column controls: the header text for fixed columns, the key for keyed ones.
const getColumnLabel = (intl: IntlShape, columnId: string): string => {
  const labelDescriptor = EVAL_RUNS_COLUMN_LABELS[columnId as EvalRunsTableColumnId];
  return labelDescriptor
    ? intl.formatMessage(labelDescriptor)
    : (parseEvalRunsTableKeyedColumnKey(columnId)?.key ?? columnId);
};

/** Sort-direction toggle shown only on actively-sorted columns; clicking flips asc↔desc. */
export const EvalRunsHeaderSortToggle = ({ column }: { column: Column<RunEntityOrGroupData, unknown> }) => {
  const intl = useIntl();
  const sorted = column.getIsSorted();
  if (!sorted) {
    return null;
  }
  const columnLabel = getColumnLabel(intl, column.id);
  return (
    <Button
      componentId="mlflow.eval-runs.column-header-menu.sort-toggle"
      size="small"
      icon={sorted === 'asc' ? <SortAscendingIcon /> : <SortDescendingIcon />}
      aria-label={intl.formatMessage(
        {
          defaultMessage: 'Toggle sort direction for {columnLabel}',
          description: 'Accessible label for the clickable sort-direction icon in the evaluation runs table header',
        },
        { columnLabel },
      )}
      onClick={(e) => {
        e.stopPropagation();
        column.toggleSorting(sorted === 'asc');
      }}
    />
  );
};

export interface EvalRunsColumnHeaderMenuProps {
  column: Column<RunEntityOrGroupData, unknown>;
  /** Whether the column may be hidden (false for the run-name / icon columns that must always show). */
  hideable: boolean;
  selectedColumns: { [key: string]: boolean };
  setSelectedColumns: (columns: { [key: string]: boolean }) => void;
  /** SearchRuns field represented by the column, when column-header filtering is enabled. */
  filterField?: string;
  onFilterByColumn?: (field: string) => void;
}

/** Per-column dropdown: sort asc/desc and hide. "Hide" flips off selectedColumns; Columns button restores it. */
export const EvalRunsColumnHeaderMenu = ({
  column,
  hideable,
  selectedColumns,
  setSelectedColumns,
  filterField,
  onFilterByColumn,
}: EvalRunsColumnHeaderMenuProps) => {
  const intl = useIntl();
  const pendingFilterFieldRef = useRef<string>();
  const canSort = column.getCanSort();
  const canFilter = Boolean(filterField && onFilterByColumn);
  const handleFilterByColumn = () => {
    if (filterField && onFilterByColumn) {
      // Wait until Radix has dismissed this menu before opening the sibling filter popover. Calling
      // the sibling action from the item click races the menu's outside-interaction cleanup.
      pendingFilterFieldRef.current = filterField;
    }
  };
  const handleCloseAutoFocus = () => {
    const field = pendingFilterFieldRef.current;
    pendingFilterFieldRef.current = undefined;
    if (field && onFilterByColumn) {
      onFilterByColumn(field);
    }
  };
  if (!canSort && !hideable && !canFilter) {
    return null;
  }
  const sorted = column.getIsSorted();
  const columnLabel = getColumnLabel(intl, column.id);

  return (
    <DropdownMenu.Root modal={false}>
      <DropdownMenu.Trigger asChild>
        <Button
          componentId="mlflow.eval-runs.column-header-menu.trigger"
          className="eval-runs-header-menu-trigger"
          size="small"
          icon={<ChevronDownIcon />}
          aria-label={intl.formatMessage(
            {
              defaultMessage: 'Options for {columnLabel} column',
              description: 'Accessible label for the per-column options menu in the evaluation runs table header',
            },
            { columnLabel },
          )}
          onClick={(e) => e.stopPropagation()}
        />
      </DropdownMenu.Trigger>
      <DropdownMenu.Content align="start" onCloseAutoFocus={handleCloseAutoFocus}>
        {canSort && (
          <>
            <DropdownMenu.Item
              componentId="mlflow.eval-runs.column-header-menu.sort-ascending"
              onClick={() => column.toggleSorting(false)}
            >
              <DropdownMenu.IconWrapper>
                <SortAscendingIcon />
              </DropdownMenu.IconWrapper>
              <FormattedMessage
                defaultMessage="Sort ascending"
                description="Menu option to sort a column ascending in the evaluation runs table header"
              />
              {sorted === 'asc' && (
                <DropdownMenu.HintColumn>
                  <CheckIcon />
                </DropdownMenu.HintColumn>
              )}
            </DropdownMenu.Item>
            <DropdownMenu.Item
              componentId="mlflow.eval-runs.column-header-menu.sort-descending"
              onClick={() => column.toggleSorting(true)}
            >
              <DropdownMenu.IconWrapper>
                <SortDescendingIcon />
              </DropdownMenu.IconWrapper>
              <FormattedMessage
                defaultMessage="Sort descending"
                description="Menu option to sort a column descending in the evaluation runs table header"
              />
              {sorted === 'desc' && (
                <DropdownMenu.HintColumn>
                  <CheckIcon />
                </DropdownMenu.HintColumn>
              )}
            </DropdownMenu.Item>
          </>
        )}
        {canSort && (hideable || canFilter) && <DropdownMenu.Separator />}
        {canFilter && (
          <DropdownMenu.Item
            componentId="mlflow.eval-runs.column-header-menu.filter-by-column"
            onClick={handleFilterByColumn}
          >
            <DropdownMenu.IconWrapper>
              <FilterIcon />
            </DropdownMenu.IconWrapper>
            <FormattedMessage
              defaultMessage="Filter by this column"
              description="Menu option to open the evaluation runs filters for the selected column"
            />
          </DropdownMenu.Item>
        )}
        {canFilter && hideable && <DropdownMenu.Separator />}
        {hideable && (
          <DropdownMenu.Item
            componentId="mlflow.eval-runs.column-header-menu.hide-column"
            onClick={() => setSelectedColumns({ ...selectedColumns, [column.id]: false })}
          >
            <DropdownMenu.IconWrapper>
              <VisibleOffIcon />
            </DropdownMenu.IconWrapper>
            <FormattedMessage
              defaultMessage="Hide column"
              description="Menu option to hide a column in the evaluation runs table header"
            />
          </DropdownMenu.Item>
        )}
      </DropdownMenu.Content>
    </DropdownMenu.Root>
  );
};
