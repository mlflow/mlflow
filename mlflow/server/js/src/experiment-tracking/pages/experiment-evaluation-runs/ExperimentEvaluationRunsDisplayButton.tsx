import { Fragment, useMemo } from 'react';
import {
  BarChartIcon,
  Button,
  ChevronDownIcon,
  ColumnsIcon,
  DropdownMenu,
  SchemaIcon,
  SlidersIcon,
  SortAscendingIcon,
  TableIcon,
  useDesignSystemTheme,
} from '@databricks/design-system';
import type { OnChangeFn, SortingState } from '@tanstack/react-table';
import { FormattedMessage, useIntl, type IntlShape } from 'react-intl';
import type { ReactNode } from 'react';
import { ToolbarCollapsibleLabel } from '@databricks/web-shared/traces-table';
import type { RunEntity } from '../../types';
import type { RunsGroupByConfig } from '../../components/experiment-page/utils/experimentPage.group-row-utils';
import { isGroupedBy } from '../../components/experiment-page/utils/experimentPage.group-row-utils';
import {
  RunGroupingAggregateFunction,
  RunGroupingMode,
} from '../../components/experiment-page/utils/experimentPage.row-types';
import { isUserFacingTag } from '@mlflow/mlflow/src/common/utils/TagUtils';
import {
  EVAL_RUN_AGENT_FUNCTION_TAG,
  EVAL_RUN_AGENT_GROUP_BY_KEY,
  EVAL_RUN_AGENT_URI_TAG,
  EVAL_RUN_SCORERS_DIGEST_TAG,
  EVAL_RUN_SCORERS_GROUP_BY_KEY,
} from '../eval-run-details/evalRunLineage';
import { EVAL_RUNS_COLUMN_LABELS, EvalRunsTableColumnId } from './ExperimentEvaluationRunsTable.constants';
import { parseEvalRunsTableKeyedColumnKey } from './ExperimentEvaluationRunsTable.utils';
import { EvalRunsColumnsMenu } from './EvalRunsColumnsButton';
import { ExperimentEvaluationRunsPageMode } from './hooks/useExperimentEvaluationRunsPageMode';
import { useEvalRunsSorting } from './hooks/useEvalRunsTableUrlState';

const COMPONENT_ID = 'mlflow.eval-runs.display';
const SORT_VALUE_SEPARATOR = ':';

export interface ExperimentEvaluationRunsDisplayButtonProps {
  runs: RunEntity[];
  selectedColumns: { [key: string]: boolean };
  setSelectedColumns: (columns: { [key: string]: boolean }) => void;
  groupByConfig: RunsGroupByConfig | null;
  setGroupByConfig: (groupBy: RunsGroupByConfig | null) => void;
  /** Optional controlled sorting state. The table URL state is used when omitted. */
  sorting?: SortingState;
  setSorting?: OnChangeFn<SortingState>;
  viewMode?: ExperimentEvaluationRunsPageMode;
  setViewMode?: (mode: ExperimentEvaluationRunsPageMode) => void;
}

export const parseSortValue = (value: string): [string, boolean] => {
  const separatorIndex = value.lastIndexOf(SORT_VALUE_SEPARATOR);
  return [value.slice(0, separatorIndex), value.slice(separatorIndex + SORT_VALUE_SEPARATOR.length) === 'desc'];
};

const getColumnLabel = (intl: IntlShape, column: string): string => {
  const labelDescriptor = EVAL_RUNS_COLUMN_LABELS[column as EvalRunsTableColumnId];
  return labelDescriptor
    ? intl.formatMessage(labelDescriptor)
    : (parseEvalRunsTableKeyedColumnKey(column)?.key ?? column);
};

const getSortValue = (sorting: SortingState): string => {
  const activeSort = sorting[0];
  return activeSort ? `${activeSort.id}${SORT_VALUE_SEPARATOR}${activeSort.desc ? 'desc' : 'asc'}` : 'none';
};

/**
 * Consolidated display menu for the evaluation-runs table. This menu owns column visibility,
 * grouping, sorting, and the legacy chart view.
 */
export const ExperimentEvaluationRunsDisplayButton = ({
  runs,
  selectedColumns,
  setSelectedColumns,
  groupByConfig,
  setGroupByConfig,
  sorting: controlledSorting,
  setSorting: controlledSetSorting,
  viewMode,
  setViewMode,
}: ExperimentEvaluationRunsDisplayButtonProps) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  const [urlSorting, setUrlSorting] = useEvalRunsSorting();
  const sorting = controlledSorting ?? urlSorting;
  const setSorting = controlledSetSorting ?? setUrlSorting;

  const { uniqueParams, uniqueTags, hasDatasets, hasScorersDigest, hasAgent } = useMemo(() => {
    const uniqueParams = new Set<string>();
    const uniqueTags = new Set<string>();
    let hasDatasets = false;
    let hasScorersDigest = false;
    let hasAgent = false;

    for (const run of runs) {
      hasDatasets ||= (run.inputs?.datasetInputs?.length ?? 0) > 0;
      for (const param of run.data?.params ?? []) {
        uniqueParams.add(param.key);
      }
      for (const tag of run.data?.tags ?? []) {
        if (isUserFacingTag(tag.key)) {
          uniqueTags.add(tag.key);
        }
        hasScorersDigest ||= tag.key === EVAL_RUN_SCORERS_DIGEST_TAG;
        hasAgent ||= tag.key === EVAL_RUN_AGENT_URI_TAG || tag.key === EVAL_RUN_AGENT_FUNCTION_TAG;
      }
    }

    return { uniqueParams, uniqueTags, hasDatasets, hasScorersDigest, hasAgent };
  }, [runs]);
  const hasAttributes = hasDatasets || hasScorersDigest || hasAgent;

  const sortOptions = useMemo(() => {
    const knownSortableColumns = [
      EvalRunsTableColumnId.runName,
      EvalRunsTableColumnId.status,
      EvalRunsTableColumnId.createdAt,
      EvalRunsTableColumnId.dataset,
      EvalRunsTableColumnId.datasetVersion,
    ];
    const dynamicColumns = Object.keys(selectedColumns).filter((column) =>
      Boolean(parseEvalRunsTableKeyedColumnKey(column)),
    );
    return [...knownSortableColumns, ...dynamicColumns]
      .filter((column, index, columns) => columns.indexOf(column) === index)
      .filter((column) => selectedColumns[column])
      .map((column) => ({ id: column, label: getColumnLabel(intl, column) }));
  }, [intl, selectedColumns]);

  const activeSort = sorting[0];
  const activeSortLabel = activeSort ? getColumnLabel(intl, activeSort.id) : undefined;
  const activeGroupCount = groupByConfig?.groupByKeys?.length ?? 0;
  const hasGroupableKeys = hasAttributes || uniqueParams.size > 0 || uniqueTags.size > 0;
  const displayLabel = intl.formatMessage({
    defaultMessage: 'Display',
    description: 'Label for the evaluation-runs display options toolbar button',
  });

  const toggleGroupBy = (mode: RunGroupingMode, columnName: string) => {
    const groupByKeys = groupByConfig?.groupByKeys ?? [];
    const newGroupByKeys = [...groupByKeys];
    const existingIndex = newGroupByKeys.findIndex((key) => key.mode === mode && key.groupByData === columnName);
    if (existingIndex >= 0) {
      newGroupByKeys.splice(existingIndex, 1);
    } else {
      newGroupByKeys.push({ mode, groupByData: columnName });
    }

    setGroupByConfig(
      newGroupByKeys.length === 0
        ? null
        : {
            aggregateFunction: groupByConfig?.aggregateFunction ?? RunGroupingAggregateFunction.Average,
            groupByKeys: newGroupByKeys,
          },
    );
  };

  const renderGroupByOption = (mode: RunGroupingMode, value: string, label: ReactNode) => (
    <DropdownMenu.CheckboxItem
      key={`${mode}:${value}`}
      componentId={`${COMPONENT_ID}.group-by-option`}
      checked={isGroupedBy(groupByConfig, mode, value)}
      onSelect={(event) => {
        event.preventDefault();
        toggleGroupBy(mode, value);
      }}
    >
      <DropdownMenu.ItemIndicator />
      {label}
    </DropdownMenu.CheckboxItem>
  );

  const handleSortChange = (value: string) => {
    if (value === 'none') {
      setSorting([]);
      return;
    }
    const [column, desc] = parseSortValue(value);
    setSorting([{ id: column, desc }]);
  };

  const isChartsView = viewMode === ExperimentEvaluationRunsPageMode.CHARTS;

  return (
    <DropdownMenu.Root modal={false}>
      <DropdownMenu.Trigger asChild>
        <Button
          componentId={`${COMPONENT_ID}.trigger`}
          icon={<SlidersIcon css={{ color: theme.colors.textSecondary }} />}
          endIcon={<ChevronDownIcon />}
          aria-label={displayLabel}
        >
          <ToolbarCollapsibleLabel>
            <FormattedMessage
              defaultMessage="Display"
              description="Label for the evaluation-runs display options toolbar button"
            />
          </ToolbarCollapsibleLabel>
        </Button>
      </DropdownMenu.Trigger>
      <DropdownMenu.Content align="end">
        <DropdownMenu.Sub>
          <DropdownMenu.SubTrigger>
            <DropdownMenu.IconWrapper>
              <ColumnsIcon />
            </DropdownMenu.IconWrapper>
            <FormattedMessage
              defaultMessage="Columns"
              description="Display menu item opening the evaluation-runs column-visibility submenu"
            />
          </DropdownMenu.SubTrigger>
          <DropdownMenu.SubContent>
            <EvalRunsColumnsMenu selectedColumns={selectedColumns} setSelectedColumns={setSelectedColumns} />
          </DropdownMenu.SubContent>
        </DropdownMenu.Sub>

        <DropdownMenu.Sub>
          <DropdownMenu.SubTrigger>
            <DropdownMenu.IconWrapper>
              <SortAscendingIcon />
            </DropdownMenu.IconWrapper>
            <FormattedMessage
              defaultMessage="Sort"
              description="Display menu item opening the evaluation-runs sort submenu"
            />
            <DropdownMenu.HintColumn
              css={{
                flexGrow: 1,
                textAlign: 'right',
                marginLeft: theme.spacing.sm,
              }}
            >
              {activeSortLabel}
            </DropdownMenu.HintColumn>
          </DropdownMenu.SubTrigger>
          <DropdownMenu.SubContent>
            <DropdownMenu.RadioGroup
              componentId={`${COMPONENT_ID}.sort`}
              value={getSortValue(sorting)}
              onValueChange={handleSortChange}
            >
              <DropdownMenu.RadioItem value="none">
                <DropdownMenu.ItemIndicator />
                <FormattedMessage defaultMessage="None" description="Evaluation-runs sort option clearing sorting" />
              </DropdownMenu.RadioItem>
              {sortOptions.map(({ id, label }) => (
                <Fragment key={id}>
                  <DropdownMenu.RadioItem value={`${id}${SORT_VALUE_SEPARATOR}desc`}>
                    <DropdownMenu.ItemIndicator />
                    <FormattedMessage
                      defaultMessage="{column} (descending)"
                      description="Evaluation-runs sort option in descending order"
                      values={{ column: label }}
                    />
                  </DropdownMenu.RadioItem>
                  <DropdownMenu.RadioItem value={`${id}${SORT_VALUE_SEPARATOR}asc`}>
                    <DropdownMenu.ItemIndicator />
                    <FormattedMessage
                      defaultMessage="{column} (ascending)"
                      description="Evaluation-runs sort option in ascending order"
                      values={{ column: label }}
                    />
                  </DropdownMenu.RadioItem>
                </Fragment>
              ))}
            </DropdownMenu.RadioGroup>
          </DropdownMenu.SubContent>
        </DropdownMenu.Sub>

        <DropdownMenu.Sub>
          <DropdownMenu.SubTrigger>
            <DropdownMenu.IconWrapper>
              <SchemaIcon />
            </DropdownMenu.IconWrapper>
            <FormattedMessage
              defaultMessage="Group by"
              description="Display menu item opening the evaluation-runs group-by submenu"
            />
            <DropdownMenu.HintColumn
              css={{
                flexGrow: 1,
                textAlign: 'right',
                marginLeft: theme.spacing.sm,
              }}
            >
              {activeGroupCount > 0 ? activeGroupCount : undefined}
            </DropdownMenu.HintColumn>
          </DropdownMenu.SubTrigger>
          <DropdownMenu.SubContent>
            {!hasGroupableKeys && (
              <DropdownMenu.Item componentId={`${COMPONENT_ID}.no-groupable-fields`} disabled>
                <FormattedMessage
                  defaultMessage="No fields to group by yet"
                  description="Empty evaluation-runs group-by menu message"
                />
              </DropdownMenu.Item>
            )}
            {hasAttributes && (
              <>
                <DropdownMenu.Label>
                  <FormattedMessage defaultMessage="Attributes" description="Evaluation-runs group-by attributes" />
                </DropdownMenu.Label>
                {hasDatasets &&
                  renderGroupByOption(
                    RunGroupingMode.Dataset,
                    'dataset',
                    <FormattedMessage defaultMessage="Dataset" description="Group evaluation runs by dataset" />,
                  )}
                {hasScorersDigest &&
                  renderGroupByOption(
                    RunGroupingMode.Tag,
                    EVAL_RUN_SCORERS_GROUP_BY_KEY,
                    <FormattedMessage
                      defaultMessage="Scorers"
                      description="Group evaluation runs by the scorer definitions they used"
                    />,
                  )}
                {hasAgent &&
                  renderGroupByOption(
                    RunGroupingMode.Tag,
                    EVAL_RUN_AGENT_GROUP_BY_KEY,
                    <FormattedMessage
                      defaultMessage="Agent"
                      description="Group evaluation runs by the evaluated agent"
                    />,
                  )}
              </>
            )}
            {uniqueParams.size > 0 && (
              <>
                {hasAttributes && <DropdownMenu.Separator />}
                <DropdownMenu.Label>
                  <FormattedMessage defaultMessage="Parameters" description="Evaluation-runs group-by parameters" />
                </DropdownMenu.Label>
                {Array.from(uniqueParams).map((param) => renderGroupByOption(RunGroupingMode.Param, param, param))}
              </>
            )}
            {uniqueTags.size > 0 && (
              <>
                {(hasAttributes || uniqueParams.size > 0) && <DropdownMenu.Separator />}
                <DropdownMenu.Label>
                  <FormattedMessage defaultMessage="Tags" description="Evaluation-runs group-by tags" />
                </DropdownMenu.Label>
                {Array.from(uniqueTags).map((tag) => renderGroupByOption(RunGroupingMode.Tag, tag, tag))}
              </>
            )}
          </DropdownMenu.SubContent>
        </DropdownMenu.Sub>

        <DropdownMenu.Item
          componentId={`${COMPONENT_ID}.view`}
          disabled={!setViewMode}
          onClick={() =>
            setViewMode?.(
              isChartsView ? ExperimentEvaluationRunsPageMode.TRACES : ExperimentEvaluationRunsPageMode.CHARTS,
            )
          }
        >
          <DropdownMenu.IconWrapper>{isChartsView ? <TableIcon /> : <BarChartIcon />}</DropdownMenu.IconWrapper>
          {isChartsView ? (
            <FormattedMessage
              defaultMessage="View table"
              description="Evaluation-runs menu option opening the table view"
            />
          ) : (
            <FormattedMessage
              defaultMessage="View legacy charts"
              description="Evaluation-runs menu option opening the legacy charts view"
            />
          )}
        </DropdownMenu.Item>
      </DropdownMenu.Content>
    </DropdownMenu.Root>
  );
};
