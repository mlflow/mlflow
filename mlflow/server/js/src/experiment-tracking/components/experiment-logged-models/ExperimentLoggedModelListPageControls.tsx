import {
  Button,
  ChartLineIcon,
  Checkbox,
  ListIcon,
  SegmentedControlButton,
  SegmentedControlGroup,
  SortAscendingIcon,
  SortDescendingIcon,
  Tooltip,
  Typography,
  useDesignSystemTheme,
  visuallyHidden,
} from '@databricks/design-system';
import { useIntl } from 'react-intl';

import { FormattedMessage } from 'react-intl';
import type { ColDef, ColGroupDef } from '@ag-grid-community/core';
import { ExperimentLoggedModelListPageColumnSelector } from './ExperimentLoggedModelListPageColumnSelector';
import { coerceToEnum } from '@databricks/web-shared/utils';
import { ExperimentLoggedModelListPageMode } from './hooks/useExperimentLoggedModelListPageMode';
import { ExperimentLoggedModelListPageAutoComplete } from './ExperimentLoggedModelListPageAutoComplete';
import type { LoggedModelMetricDataset, LoggedModelProto } from '../../types';
import { ExperimentLoggedModelListPageDatasetDropdown } from './ExperimentLoggedModelListPageDatasetDropdown';
import { ExperimentLoggedModelListPageOrderBySelector } from './ExperimentLoggedModelListPageOrderBySelector';
import type { LoggedModelsTableGroupByMode } from './ExperimentLoggedModelListPageTable.utils';
import { ExperimentLoggedModelListPageGroupBySelector } from './ExperimentLoggedModelListPageGroupBySelector';

export const ExperimentLoggedModelListPageControls = ({
  orderByColumn,
  orderByAsc,
  sortingAndFilteringEnabled,
  onChangeOrderBy,
  onUpdateColumns,
  columnDefs,
  columnVisibility = {},
  viewMode,
  setViewMode,
  searchQuery = '',
  onChangeSearchQuery,
  loggedModelsData,
  selectedFilterDatasets,
  onToggleDataset,
  onClearSelectedDatasets,
  groupBy,
  onChangeGroupBy,
}: {
  orderByColumn?: string;
  orderByAsc?: boolean;
  groupBy?: LoggedModelsTableGroupByMode;
  onChangeGroupBy?: (groupBy: LoggedModelsTableGroupByMode | undefined) => void;
  sortingAndFilteringEnabled?: boolean;
  onChangeOrderBy: (orderByColumn: string, orderByAsc: boolean) => void;
  onUpdateColumns: (columnVisibility: Record<string, boolean>) => void;
  columnDefs?: (ColDef | ColGroupDef)[];
  columnVisibility?: Record<string, boolean>;
  viewMode: ExperimentLoggedModelListPageMode;
  setViewMode: (mode: ExperimentLoggedModelListPageMode) => void;
  searchQuery?: string;
  onChangeSearchQuery: (searchFilter: string) => void;
  loggedModelsData: LoggedModelProto[];
  selectedFilterDatasets?: LoggedModelMetricDataset[];
  onToggleDataset?: (dataset: LoggedModelMetricDataset) => void;
  onClearSelectedDatasets?: () => void;
}) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();

  return (
    <div css={{ display: 'flex', flexWrap: 'wrap', gap: theme.spacing.sm }}>
      <SegmentedControlGroup
        componentId="mlflow.logged_model.list.view-mode"
        name="view-mode"
        value={viewMode}
        onChange={(e) => {
          setViewMode(
            coerceToEnum(ExperimentLoggedModelListPageMode, e.target.value, ExperimentLoggedModelListPageMode.TABLE),
          );
        }}
      >
        <SegmentedControlButton value="TABLE">
          <Tooltip
            componentId="mlflow.logged_model.list.view-mode-table-tooltip"
            content={intl.formatMessage({
              defaultMessage: 'Table view',
              description: 'Label for the table view toggle button in the logged model list page',
            })}
          >
            <ListIcon />
          </Tooltip>
          <span css={visuallyHidden}>
            {intl.formatMessage({
              defaultMessage: 'Table view',
              description: 'Label for the table view toggle button in the logged model list page',
            })}
          </span>
        </SegmentedControlButton>
        <SegmentedControlButton value="CHART">
          <Tooltip
            componentId="mlflow.logged_model.list.view-mode-chart-tooltip"
            content={intl.formatMessage({
              defaultMessage: 'Chart view',
              description: 'Label for the table view toggle button in the logged model list page',
            })}
          >
            <ChartLineIcon />
          </Tooltip>
          <span css={visuallyHidden}>
            {intl.formatMessage({
              defaultMessage: 'Chart view',
              description: 'Label for the table view toggle button in the logged model list page',
            })}
          </span>
        </SegmentedControlButton>
      </SegmentedControlGroup>
      {sortingAndFilteringEnabled ? (
        <>
          <ExperimentLoggedModelListPageAutoComplete
            searchQuery={searchQuery}
            onChangeSearchQuery={onChangeSearchQuery}
            loggedModelsData={loggedModelsData}
          />
          <ExperimentLoggedModelListPageDatasetDropdown
            loggedModelsData={loggedModelsData}
            onToggleDataset={onToggleDataset}
            onClearSelectedDatasets={onClearSelectedDatasets}
            selectedFilterDatasets={selectedFilterDatasets}
          />
          <ExperimentLoggedModelListPageOrderBySelector
            orderByColumn={orderByColumn ?? ''}
            orderByAsc={orderByAsc}
            onChangeOrderBy={onChangeOrderBy}
            columnDefs={columnDefs}
          />
        </>
      ) : (
        <Button
          componentId="mlflow.logged_model.list.sort"
          icon={orderByAsc ? <SortAscendingIcon /> : <SortDescendingIcon />}
          onClick={() => {
            orderByColumn && onChangeOrderBy(orderByColumn, !orderByAsc);
          }}
        >
          <FormattedMessage
            defaultMessage="Sort: Created"
            description="Label for the sort button in the logged model list page"
          />
        </Button>
      )}
      <ExperimentLoggedModelListPageColumnSelector
        columnDefs={columnDefs}
        columnVisibility={columnVisibility}
        onUpdateColumns={onUpdateColumns}
        disabled={viewMode === ExperimentLoggedModelListPageMode.CHART}
      />
      <ExperimentLoggedModelListPageGroupBySelector groupBy={groupBy} onChangeGroupBy={onChangeGroupBy} />
    </div>
  );
};
