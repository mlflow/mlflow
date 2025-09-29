import { isNil } from 'lodash';
import React, { useCallback } from 'react';

import {
  Typography,
  useDesignSystemTheme,
  TableFilterLayout,
  Tooltip,
  Spinner,
  WarningIcon,
} from '@databricks/design-system';
import { useIntl } from '@databricks/i18n';

import { GenAITracesTableActions } from './GenAITracesTableActions';
import { GenAiTracesTableFilter } from './GenAiTracesTableFilter';
import { GenAiTracesTableSearchInput } from './GenAiTracesTableSearchInput';
import { EvaluationsOverviewColumnSelectorGrouped } from './components/EvaluationsOverviewColumnSelectorGrouped';
import { EvaluationsOverviewSortDropdown } from './components/EvaluationsOverviewSortDropdown';
import type {
  TraceInfoV3,
  EvaluationsOverviewTableSort,
  TraceActions,
  AssessmentInfo,
  TracesTableColumn,
  TableFilter,
  TableFilterOptions,
} from './types';
import { shouldEnableTagGrouping } from './utils/FeatureUtils';

interface CountInfo {
  currentCount?: number;
  totalCount: number;
  maxAllowedCount: number;
  logCountLoading: boolean;
}

interface GenAITracesTableToolbarProps {
  // Experiment metadata
  experimentId: string;

  // Table metadata
  allColumns: TracesTableColumn[];
  assessmentInfos: AssessmentInfo[];

  // Table data
  traceInfos: TraceInfoV3[] | undefined;

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
}

export const GenAITracesTableToolbar: React.FC<React.PropsWithChildren<GenAITracesTableToolbarProps>> = React.memo(
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
      metadataError,
    } = props;
    const { theme } = useDesignSystemTheme();

    const onSortChange = useCallback(
      (sortOption, orderByAsc) => {
        setTableSort({ key: sortOption.key, type: sortOption.type, asc: orderByAsc });
      },
      [setTableSort],
    );

    return (
      <div
        css={{
          display: 'flex',
          width: '100%',
          alignItems: 'flex-end',
          justifyContent: 'space-between',
          paddingBottom: `${theme.spacing.xs}px`,
        }}
      >
        <TableFilterLayout
          css={{
            marginBottom: 0,
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
            isMetadataLoading={isMetadataLoading}
            metadataError={metadataError}
          />
          <EvaluationsOverviewSortDropdown
            tableSort={tableSort}
            columns={selectedColumns}
            onChange={onSortChange}
            enableGrouping={shouldEnableTagGrouping()}
            isMetadataLoading={isMetadataLoading}
            metadataError={metadataError}
          />

          <EvaluationsOverviewColumnSelectorGrouped
            columns={allColumns}
            selectedColumns={selectedColumns}
            toggleColumns={toggleColumns}
            setSelectedColumns={setSelectedColumns}
            isMetadataLoading={isMetadataLoading}
            metadataError={metadataError}
          />
          {traceActions && (
            <GenAITracesTableActions experimentId={experimentId} traceActions={traceActions} traceInfos={traceInfos} />
          )}
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
