import { Fragment, useMemo } from 'react';
import {
  Button,
  ColumnsIcon,
  DropdownMenu,
  Tooltip,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { useIntl } from 'react-intl';
import { groupBy } from 'lodash';
import type { EvalRunsTableColumnId } from './ExperimentEvaluationRunsTable.constants';
import {
  EVAL_RUNS_COLUMN_LABELS,
  EVAL_RUNS_COLUMN_TYPE_LABELS,
  EVAL_RUNS_TOOLBAR_ICON_SIZE,
  EVAL_RUNS_UNSELECTABLE_COLUMNS,
} from './ExperimentEvaluationRunsTable.constants';
import { EvalRunsTableKeyedColumnPrefix } from './ExperimentEvaluationRunsTable.constants';
import { parseEvalRunsTableKeyedColumnKey } from './ExperimentEvaluationRunsTable.utils';

const COMPONENT_ID = 'mlflow.eval-runs.columns';

export interface EvalRunsColumnsButtonProps {
  /** Full column-visibility map (base attributes + dynamic metric/param/tag keys), as the page owns it. */
  selectedColumns: { [key: string]: boolean };
  setSelectedColumns: (columns: { [key: string]: boolean }) => void;
}

export interface EvalRunsColumnsMenuProps {
  /** Full column-visibility map (base attributes + dynamic metric/param/tag keys), as the page owns it. */
  selectedColumns: { [key: string]: boolean };
  setSelectedColumns: (columns: { [key: string]: boolean }) => void;
}

/**
 * Icon-only column-visibility control for the eval-runs V2 toolbar. Groups toggleable columns by type;
 * each multi-column group header is a checkbox that shows/hides the whole group. Mirrors Traces V4 Display.
 */
export const EvalRunsColumnsButton = ({ selectedColumns, setSelectedColumns }: EvalRunsColumnsButtonProps) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();

  // Active / total over toggleable columns only; always-on base columns excluded from both counts.
  const { activeColumnCount, totalColumnCount } = useMemo(() => {
    const toggleable = Object.entries(selectedColumns).filter(
      ([column]) => !EVAL_RUNS_UNSELECTABLE_COLUMNS.has(column),
    );
    return {
      activeColumnCount: toggleable.filter(([, selected]) => selected).length,
      totalColumnCount: toggleable.length,
    };
  }, [selectedColumns]);

  const columnsLabel = intl.formatMessage({
    defaultMessage: 'Columns',
    description: 'Aria label for the eval-runs columns button',
  });
  // Tooltip shows active/total count (e.g. "Columns (3/8)") so the button stays icon-only.
  const columnsTooltip = intl.formatMessage(
    {
      defaultMessage: 'Columns ({active}/{total})',
      description: 'Tooltip for the eval-runs columns button showing active vs total column counts',
    },
    { active: activeColumnCount, total: totalColumnCount },
  );

  return (
    // Non-modal so the open menu doesn't aria-hide / scroll-lock the table behind it.
    <DropdownMenu.Root modal={false}>
      <Tooltip componentId={`${COMPONENT_ID}.trigger.tooltip`} content={columnsTooltip}>
        <DropdownMenu.Trigger asChild>
          {/* Icon as a child (not the `icon` prop) so the Button keeps its grey border; icon-only renders borderless. */}
          <Button
            componentId={`${COMPONENT_ID}.trigger`}
            aria-label={columnsLabel}
            // `&&&` + !important beats dbui's importantified padding.
            css={{
              flexShrink: 0,
              '&&&': {
                padding: `${theme.spacing.xs}px ${theme.spacing.sm}px !important`,
              },
            }}
          >
            <ColumnsIcon css={{ fontSize: EVAL_RUNS_TOOLBAR_ICON_SIZE, color: theme.colors.textPlaceholder }} />
          </Button>
        </DropdownMenu.Trigger>
      </Tooltip>
      <DropdownMenu.Content align="end">
        <EvalRunsColumnsMenu selectedColumns={selectedColumns} setSelectedColumns={setSelectedColumns} />
      </DropdownMenu.Content>
    </DropdownMenu.Root>
  );
};

/** Column visibility menu content shared by the standalone legacy button and Display's submenu. */
export const EvalRunsColumnsMenu = ({ selectedColumns, setSelectedColumns }: EvalRunsColumnsMenuProps) => {
  const intl = useIntl();

  const getColumnLabel = (column: string): string => {
    const labelDescriptor = EVAL_RUNS_COLUMN_LABELS[column as EvalRunsTableColumnId];
    return labelDescriptor
      ? intl.formatMessage(labelDescriptor)
      : (parseEvalRunsTableKeyedColumnKey(column)?.key ?? column);
  };

  const getGroupLabel = (columnType: string): string => {
    const headerLabelDescriptor = EVAL_RUNS_COLUMN_TYPE_LABELS[columnType as EvalRunsTableKeyedColumnPrefix];
    return headerLabelDescriptor ? intl.formatMessage(headerLabelDescriptor) : columnType;
  };

  // Group the toggleable columns by type (Attributes / Metrics / Params / Tags). Empty groups drop out.
  const columnGroups = Object.entries(
    groupBy(
      Object.entries(selectedColumns),
      ([columnId]) =>
        parseEvalRunsTableKeyedColumnKey(columnId)?.columnType ?? EvalRunsTableKeyedColumnPrefix.ATTRIBUTE,
    ),
  )
    .map(([columnType, columns]) => ({
      columnType,
      columns: columns.filter(([column]) => !EVAL_RUNS_UNSELECTABLE_COLUMNS.has(column)),
    }))
    .filter((group) => group.columns.length > 0);

  const toggleColumn = (column: string, selected: boolean) =>
    setSelectedColumns({ ...selectedColumns, [column]: !selected });

  // Show/hide every column in a group at once (backs the group header's show/hide-all toggle).
  const setAllInGroup = (groupColumns: [string, boolean][], nextSelected: boolean) => {
    const next = { ...selectedColumns };
    groupColumns.forEach(([column]) => {
      next[column] = nextSelected;
    });
    setSelectedColumns(next);
  };

  return (
    <>
      {columnGroups.map(({ columnType, columns }, groupIndex) => {
        const groupLabel = getGroupLabel(columnType);
        const allSelectedInGroup = columns.every(([, selected]) => selected);
        return (
          <Fragment key={columnType}>
            {groupIndex > 0 && <DropdownMenu.Separator />}
            {columns.length > 1 ? (
              // Group header doubles as a show/hide-all toggle; bold text marks it as a section header.
              <DropdownMenu.CheckboxItem
                componentId={`${COMPONENT_ID}.group-toggle-all`}
                checked={allSelectedInGroup}
                onSelect={(event) => {
                  event.preventDefault();
                  setAllInGroup(columns, !allSelectedInGroup);
                }}
              >
                <DropdownMenu.ItemIndicator />
                <Typography.Text bold>{groupLabel}</Typography.Text>
              </DropdownMenu.CheckboxItem>
            ) : (
              <DropdownMenu.Label>{groupLabel}</DropdownMenu.Label>
            )}
            {columns.map(([column, selected]) => (
              <DropdownMenu.CheckboxItem
                key={column}
                // Static shared id (dynamic per-column ids are disallowed by the lint rule).
                componentId={`${COMPONENT_ID}.column-toggle`}
                checked={selected}
                // Toggle in onSelect + preventDefault so the menu stays open across changes.
                onSelect={(event) => {
                  event.preventDefault();
                  toggleColumn(column, selected);
                }}
              >
                <DropdownMenu.ItemIndicator />
                {getColumnLabel(column)}
              </DropdownMenu.CheckboxItem>
            ))}
          </Fragment>
        );
      })}
    </>
  );
};
