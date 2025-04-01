import {
  Button,
  ChartLineIcon,
  ListIcon,
  SegmentedControlButton,
  SegmentedControlGroup,
  SortAscendingIcon,
  SortDescendingIcon,
  Tooltip,
  useDesignSystemTheme,
  visuallyHidden,
} from '@databricks/design-system';
import { useIntl } from 'react-intl';

import { FormattedMessage } from 'react-intl';
import type { ColDef, ColGroupDef } from '@ag-grid-community/core';
import { ExperimentLoggedModelListPageColumnSelector } from './ExperimentLoggedModelListPageColumnSelector';
import { coerceToEnum } from '@databricks/web-shared/utils';
import { ExperimentLoggedModelListPageMode } from './hooks/useExperimentLoggedModelListPageMode';

export const ExperimentLoggedModelListPageControls = ({
  orderByField,
  orderByAsc,
  onChangeOrderBy,
  onUpdateColumns,
  columnDefs,
  columnVisibility = {},
  viewMode,
  setViewMode,
}: {
  orderByField?: string;
  orderByAsc?: boolean;
  onChangeOrderBy: (orderByField: string, orderByAsc: boolean) => void;
  onUpdateColumns: (columnVisibility: Record<string, boolean>) => void;
  columnDefs?: (ColDef | ColGroupDef)[];
  columnVisibility?: Record<string, boolean>;
  viewMode: ExperimentLoggedModelListPageMode;
  setViewMode: (mode: ExperimentLoggedModelListPageMode) => void;
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
      {/* TODO: enable when filtering is available */}
      {/* <Input
        prefix={<SearchIcon />}
        componentId="mlflow.logged_model.list.search"
        css={{ width: 430 }}
        placeholder={intl.formatMessage({
          defaultMessage: 'Search models',
          description: 'Placeholder for the search input in the logged model list page',
        })}
        allowClear
        value={searchQuery}
        onChange={(e) => onChangeSearchQuery(e.target.value)}
      /> */}

      {/* TODO: enable when filtering is available */}
      {/* <DropdownMenu.Root>
        <DropdownMenu.Trigger asChild>
          <Button componentId="mlflow.logged_model.list.filter" icon={<FilterIcon />}>
            <FormattedMessage
              defaultMessage="Filter"
              description="Label for the filter button in the logged model list page"
            />
          </Button>
        </DropdownMenu.Trigger>
        <DropdownMenu.Content>
          <DropdownMenu.Item componentId="">[TODO: implement filters]</DropdownMenu.Item>
        </DropdownMenu.Content>
      </DropdownMenu.Root> */}
      <Button
        componentId="mlflow.logged_model.list.sort"
        icon={orderByAsc ? <SortAscendingIcon /> : <SortDescendingIcon />}
        onClick={() => {
          orderByField && onChangeOrderBy(orderByField, !orderByAsc);
        }}
      >
        <FormattedMessage
          defaultMessage="Sort: Created"
          description="Label for the sort button in the logged model list page"
        />
      </Button>
      <ExperimentLoggedModelListPageColumnSelector
        columnDefs={columnDefs}
        columnVisibility={columnVisibility}
        onUpdateColumns={onUpdateColumns}
        disabled={viewMode === ExperimentLoggedModelListPageMode.CHART}
      />
    </div>
  );
};
