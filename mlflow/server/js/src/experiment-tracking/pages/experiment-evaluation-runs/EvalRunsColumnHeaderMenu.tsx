import {
  Button,
  CheckIcon,
  ChevronDownIcon,
  DropdownMenu,
  SortAscendingIcon,
  SortDescendingIcon,
  VisibleOffIcon,
} from '@databricks/design-system';
import type { Column } from '@tanstack/react-table';
import { FormattedMessage, useIntl } from 'react-intl';
import type { RunEntityOrGroupData } from './ExperimentEvaluationRunsPage.utils';
import { parseEvalRunsTableKeyedColumnKey } from './ExperimentEvaluationRunsTable.utils';

/** Sort-direction toggle shown only on actively-sorted columns; clicking flips asc↔desc. */
export const EvalRunsHeaderSortToggle = ({ column }: { column: Column<RunEntityOrGroupData, unknown> }) => {
  const intl = useIntl();
  const sorted = column.getIsSorted();
  if (!sorted) {
    return null;
  }
  // Screen-reader label differentiates per-column sort toggles.
  const columnLabel = parseEvalRunsTableKeyedColumnKey(column.id)?.key ?? column.id;
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
}

/** Per-column dropdown: sort asc/desc and hide. "Hide" flips off selectedColumns; Columns button restores it. */
export const EvalRunsColumnHeaderMenu = ({
  column,
  hideable,
  selectedColumns,
  setSelectedColumns,
}: EvalRunsColumnHeaderMenuProps) => {
  const intl = useIntl();
  const canSort = column.getCanSort();
  if (!canSort && !hideable) {
    return null;
  }
  const sorted = column.getIsSorted();
  // Screen-reader label differentiates per-column option menus.
  const columnLabel = parseEvalRunsTableKeyedColumnKey(column.id)?.key ?? column.id;

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
      <DropdownMenu.Content align="start">
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
        {canSort && hideable && <DropdownMenu.Separator />}
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
