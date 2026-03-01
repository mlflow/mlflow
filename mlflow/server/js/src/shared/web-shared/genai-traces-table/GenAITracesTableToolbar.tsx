import { isNil } from 'lodash';
import React, { useCallback } from 'react';

import {
  Typography,
  useDesignSystemTheme,
  TableFilterLayout,
  Tooltip,
  Spinner,
  WarningIcon,
  Button,
  RefreshIcon,
  ToggleButton,
  SparkleIcon,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';

import { GenAITracesTableActions } from './GenAITracesTableActions';
import { GenAiTracesTableFilter } from './GenAiTracesTableFilter';
import { GenAiTracesTableSearchInput } from './GenAiTracesTableSearchInput';
import { EvaluationsOverviewColumnSelectorGrouped } from './components/EvaluationsOverviewColumnSelectorGrouped';
import { EvaluationsOverviewSortDropdown } from './components/EvaluationsOverviewSortDropdown';
import type {
  EvaluationsOverviewTableSort,
  TraceActions,
  AssessmentInfo,
  TracesTableColumn,
  TableFilter,
  TableFilterOptions,
} from './types';
import { shouldEnableSessionGrouping, shouldEnableTagGrouping } from './utils/FeatureUtils';
import { shouldEnableIssueDetection } from '../../../common/utils/FeatureUtils';
import type { ModelTraceInfoV3 } from '../model-trace-explorer/ModelTrace.types';

interface CountInfo {
  currentCount?: number;
  totalCount: number;
  maxAllowedCount: number;
  logCountLoading: boolean;
}

interface GenAITracesTableToolbarProps {
  // Experiment metadata
  experimentId?: string;

  // Table metadata
  allColumns: TracesTableColumn[];
  assessmentInfos: AssessmentInfo[];

  // Table data
  traceInfos: ModelTraceInfoV3[] | undefined;

  // Filters
  searchQuery: string;
  setSearchQuery: (query: string) => void;
  filters: TableFilter[];
  setFilters: (newFilters: TableFilter[] | undefined, replace?: boolean) => void;
  tableSort: EvaluationsOverviewTableSort | undefined;
  setTableSort: (sort: EvaluationsOverviewTableSort | undefined) => void;
  selectedColumns: TracesTableColumn[];
  toggleColumns: (newColumns: TracesTableColumn[]) => void;
  setSelectedColumns: (nextSelected: TracesTableColumn[]) => void;

  // Actions
  traceActions?: TraceActions;

  // Stats
  countInfo: CountInfo;

  // Table filter options
  tableFilterOptions: TableFilterOptions;

  // Loading state
  isMetadataLoading?: boolean;

  // Error state
  metadataError?: Error | null;

  // whether or not the toolbar show show additional search options only
  // available in the new APIs. this param is somewhat confusingly named
  // in OSS, since the "new APIs" still use the v3 prefixes
  usesV4APIs?: boolean;
  onRefresh?: () => void;
  isRefreshing?: boolean;

  // Session grouping
  isGroupedBySession?: boolean;
  forceGroupBySession?: boolean;
  onToggleSessionGrouping?: () => void;

  // Issue detection
  onDetectIssues?: () => void;

  // Additional elements to render in the toolbar
  addons?: React.ReactNode;
}

export const GenAITracesTableToolbar: React.FC<React.PropsWithChildren<GenAITracesTableToolbarProps>> = React.memo(
  // eslint-disable-next-line react-component-name/react-component-name -- TODO(FEINF-4716)
  (props: GenAITracesTableToolbarProps) => {
    const {
      searchQuery,
      setSearchQuery,
      filters,
      setFilters,
      tableSort,
      setTableSort,
      selectedColumns,
      toggleColumns,
      setSelectedColumns,
      assessmentInfos,
      experimentId,
      traceInfos,
      tableFilterOptions,
      traceActions,
      allColumns,
      countInfo,
      isMetadataLoading,
      usesV4APIs,
      metadataError,
      onRefresh,
      isRefreshing,
      isGroupedBySession,
      forceGroupBySession,
      onToggleSessionGrouping,
      onDetectIssues,
      addons,
    } = props;
    const { theme } = useDesignSystemTheme();
    const intl = useIntl();

    const onSortChange = useCallback(
      (sortOption, orderByAsc) => {
        setTableSort({ key: sortOption.key, type: sortOption.type, asc: orderByAsc });
      },
      [setTableSort],
    );

    // When using V4 APIs, we want users to be able to change filters while the traces are being loaded or there is an error
    const shouldDisplayErrorState = Boolean(metadataError && !usesV4APIs);
    const shouldDisplayLoadingState = isMetadataLoading && !usesV4APIs;

    return (
      <div
        css={{
          display: 'flex',
          width: '100%',
          alignItems: 'flex-end',
          gap: theme.spacing.sm,
          paddingBottom: `${theme.spacing.xs}px`,
        }}
      >
        <TableFilterLayout
          css={{
            marginBottom: 0,
            flex: 1,
          }}
        >
          <GenAiTracesTableSearchInput searchQuery={searchQuery} setSearchQuery={setSearchQuery} />
          <GenAiTracesTableFilter
            filters={filters}
            setFilters={setFilters}
            assessmentInfos={assessmentInfos}
            experimentId={experimentId}
            tableFilterOptions={tableFilterOptions}
            allColumns={allColumns}
            isLoading={shouldDisplayLoadingState}
            isError={shouldDisplayErrorState}
            usesV4APIs={usesV4APIs}
          />
          <EvaluationsOverviewSortDropdown
            tableSort={tableSort}
            columns={selectedColumns}
            onChange={onSortChange}
            enableGrouping={shouldEnableTagGrouping()}
            isLoading={shouldDisplayLoadingState}
            isError={shouldDisplayErrorState}
          />

          <EvaluationsOverviewColumnSelectorGrouped
            columns={allColumns}
            selectedColumns={selectedColumns}
            toggleColumns={toggleColumns}
            setSelectedColumns={setSelectedColumns}
            isLoading={shouldDisplayLoadingState}
            isError={shouldDisplayErrorState}
          />
          {traceActions && experimentId && (
            <GenAITracesTableActions
              experimentId={experimentId}
              traceActions={traceActions}
              traceInfos={traceInfos}
              // prettier-ignore
            />
          )}
          {shouldEnableSessionGrouping() && onToggleSessionGrouping && !forceGroupBySession && (
            <Tooltip
              componentId="mlflow.traces-table.group-by-session-button.tooltip"
              content={intl.formatMessage({
                defaultMessage: 'Toggle session grouping',
                description: 'Tooltip for the group by session button in the traces table toolbar',
              })}
            >
              <ToggleButton
                componentId="mlflow.traces-table.group-by-session-button"
                onPressedChange={onToggleSessionGrouping}
                pressed={isGroupedBySession}
                aria-label={intl.formatMessage({
                  defaultMessage: 'Toggle session grouping',
                  description: 'Aria label for the group by session button in the traces table toolbar',
                })}
              >
                <FormattedMessage
                  defaultMessage="Group by session"
                  description="Label for the group by session button in the traces table toolbar"
                />
              </ToggleButton>
            </Tooltip>
          )}
          {shouldEnableIssueDetection() && onDetectIssues && (
            <Tooltip
              componentId="mlflow.traces-table.detect-issues-button.tooltip"
              content={intl.formatMessage({
                defaultMessage: 'Detect issues in traces',
                description: 'Tooltip for the detect issues button in the traces table toolbar',
              })}
            >
              <Button
                componentId="mlflow.traces-table.detect-issues-button"
                onClick={onDetectIssues}
                aria-label={intl.formatMessage({
                  defaultMessage: 'Detect issues in traces',
                  description: 'Aria label for the detect issues button in the traces table toolbar',
                })}
              >
                <SparkleIcon color="ai" css={{ marginRight: theme.spacing.xs }} />
                <FormattedMessage
                  defaultMessage="Detect Issues"
                  description="Label for the detect issues button in the traces table toolbar"
                />
              </Button>
            </Tooltip>
          )}
          {onRefresh && (
            <Tooltip
              componentId="mlflow.traces-table.refresh-button.tooltip"
              content={intl.formatMessage({
                defaultMessage: 'Refresh traces',
                description: 'Tooltip for the refresh traces button in the traces table toolbar',
              })}
            >
              <Button
                componentId="mlflow.traces-table.refresh-button"
                icon={<RefreshIcon />}
                onClick={onRefresh}
                loading={isRefreshing}
                aria-label={intl.formatMessage({
                  defaultMessage: 'Refresh traces',
                  description: 'Aria label for the refresh traces button in the traces table toolbar',
                })}
              />
            </Tooltip>
          )}
          {addons}
        </TableFilterLayout>
        <SampledInfoBadge countInfo={countInfo} />
      </div>
    );
  },
);

const SampledInfoBadge = (props: { countInfo: CountInfo }) => {
  const { countInfo } = props;
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();

  if (countInfo.logCountLoading || isNil(countInfo.currentCount)) {
    return <Spinner size="small" />;
  }

  return (
    <div
      css={{
        display: 'flex',
        alignItems: 'center',
        gap: theme.spacing.sm,
      }}
    >
      {countInfo.currentCount >= countInfo.maxAllowedCount && (
        <Tooltip
          componentId="mlflow.experiment_list_view.max_traces.tooltip"
          content={intl.formatMessage(
            {
              defaultMessage: 'Only the top {evalResultsCount} results are shown',
              description: 'Evaluation review > evaluations list > sample info tooltip',
            },
            {
              evalResultsCount: countInfo.maxAllowedCount,
            },
          )}
        >
          <WarningIcon color="warning" />
        </Tooltip>
      )}
      <Typography.Hint>
        {intl.formatMessage(
          {
            defaultMessage: '{numFilteredEvals} of {numEvals}',
            description: 'Text displayed when showing a filtered subset evaluations in the evaluation review page',
          },
          {
            // Sometimes the api returns more than the max allowed count. To avoid confusion, we show the max allowed count.
            numFilteredEvals:
              countInfo.currentCount >= countInfo.maxAllowedCount ? countInfo.maxAllowedCount : countInfo.currentCount,
            numEvals:
              countInfo.totalCount >= countInfo.maxAllowedCount
                ? `${countInfo.maxAllowedCount}+`
                : countInfo.totalCount,
          },
        )}
      </Typography.Hint>
    </div>
  );
};
