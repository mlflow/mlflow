import {
  ArrowDownIcon,
  ArrowUpIcon,
  Button,
  DropdownMenu,
  Input,
  SearchIcon,
  SortAscendingIcon,
  SortDescendingIcon,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { useMemo, useState } from 'react';

import type { IntlShape, MessageDescriptor } from 'react-intl';
import { defineMessage, FormattedMessage, useIntl } from 'react-intl';
import { ToggleIconButton } from '../../../common/components/ToggleIconButton';
import {
  ExperimentLoggedModelListPageKnownColumns,
  LOGGED_MODEL_LIST_METRIC_COLUMN_PREFIX,
  parseLoggedModelMetricOrderByColumnId,
} from './hooks/useExperimentLoggedModelListPageTableColumns';

interface BasicColumnDef {
  colId?: string;
  groupId?: string;
  headerName?: string;
  children?: BasicColumnDef[];
}

const getSortableColumnLabel = (colId: string | ExperimentLoggedModelListPageKnownColumns, intl: IntlShape) => {
  const labels: Partial<Record<ExperimentLoggedModelListPageKnownColumns | string, MessageDescriptor>> = {
    [ExperimentLoggedModelListPageKnownColumns.CreationTime]: defineMessage({
      defaultMessage: 'Creation time',
      description: 'Label for the creation time column in the logged model list page',
    }),
  };

  const descriptor = labels[colId];

  if (descriptor) {
    return intl.formatMessage(descriptor);
  }

  const parsedColumn = parseLoggedModelMetricOrderByColumnId(colId);

  if (parsedColumn) {
    return parsedColumn.metricKey;
  }

  return colId;
};

export const ExperimentLoggedModelListPageOrderBySelector = ({
  orderByColumn,
  orderByAsc,
  onChangeOrderBy,
  columnDefs = [],
}: {
  orderByColumn: string;
  orderByAsc?: boolean;
  onChangeOrderBy: (orderByColumn: string, orderByAsc: boolean) => void;
  columnDefs: BasicColumnDef[] | undefined;
}) => {
  const intl = useIntl();
  const [filter, setFilter] = useState('');
  const { theme } = useDesignSystemTheme();

  const groupedOrderByOptions = useMemo<BasicColumnDef[]>(() => {
    const lowerCaseFilter = filter.toLowerCase();
    const attributeColumnGroup = {
      groupId: 'attributes',
      headerName: intl.formatMessage({
        defaultMessage: 'Attributes',
        description: 'Label for the attributes column group in the logged model column selector',
      }),
      children: [
        {
          colId: ExperimentLoggedModelListPageKnownColumns.CreationTime,
          headerName: getSortableColumnLabel(ExperimentLoggedModelListPageKnownColumns.CreationTime, intl),
        },
      ].filter(({ headerName }) => headerName?.toLowerCase().includes(lowerCaseFilter)),
    };

    // Next, get all the dataset-grouped metric column groups
    const metricColumnGroups = columnDefs
      .filter((col) => col.groupId?.startsWith(LOGGED_MODEL_LIST_METRIC_COLUMN_PREFIX))
      .map((col) => ({
        ...col,
        children: col.children?.filter(({ colId }) => colId?.includes(lowerCaseFilter)),
        headerName: col.headerName
          ? `Metrics (${col.headerName})`
          : intl.formatMessage({
              defaultMessage: 'Metrics',
              description: 'Label for the ungrouped metrics column group in the logged model column selector',
            }),
      }));

    const sortableColumnGroups = [attributeColumnGroup, ...metricColumnGroups].filter(
      (col) => col.children && col.children.length > 0,
    );

    // If the currently used sorting field is not found, this probably means that
    // user has filtered out results containing this column. Let's add it to the list
    // of sortable columns so that user won't be confused.
    if (
      !sortableColumnGroups.some((group) => group.children && group.children.some((col) => col.colId === orderByColumn))
    ) {
      const { metricKey } = parseLoggedModelMetricOrderByColumnId(orderByColumn);

      if (metricKey) {
        sortableColumnGroups.push({
          groupId: 'current',
          headerName: intl.formatMessage({
            defaultMessage: 'Currently sorted by',
            description: 'Label for the custom column group in the logged model column selector',
          }),
          children: [{ colId: orderByColumn, headerName: metricKey }],
        });
      }
    }
    return sortableColumnGroups;
  }, [columnDefs, intl, filter, orderByColumn]);

  return (
    <DropdownMenu.Root modal={false}>
      <DropdownMenu.Trigger asChild>
        <Button
          componentId="mlflow.logged_model.list.order_by"
          icon={orderByAsc ? <SortAscendingIcon /> : <SortDescendingIcon />}
        >
          <FormattedMessage
            defaultMessage="Sort: {sortBy}"
            description="Label for the filter button in the logged model list page. sortBy is the name of the column by which the table is currently sorted."
            values={{ sortBy: getSortableColumnLabel(orderByColumn, intl) }}
          />
        </Button>
      </DropdownMenu.Trigger>
      <DropdownMenu.Content css={{ maxHeight: 300, overflow: 'auto' }}>
        <div
          css={{
            padding: `${theme.spacing.sm}px ${theme.spacing.lg / 2}px ${theme.spacing.sm}px`,
            width: '100%',
            display: 'flex',
            gap: theme.spacing.xs,
          }}
        >
          <Input
            componentId="mlflow.logged_model.list.order_by.filter"
            prefix={<SearchIcon />}
            value={filter}
            type="search"
            onChange={(e) => setFilter(e.target.value)}
            placeholder={intl.formatMessage({
              defaultMessage: 'Search',
              description: 'Placeholder for the search input in the logged model list page sort column selector',
            })}
            autoFocus
            allowClear
          />
          <div
            css={{
              display: 'flex',
              gap: theme.spacing.xs,
            }}
          >
            <ToggleIconButton
              pressed={!orderByAsc}
              icon={<ArrowDownIcon />}
              componentId="mlflow.logged_model.list.order_by.button_desc"
              onClick={() => onChangeOrderBy(orderByColumn, false)}
              aria-label={intl.formatMessage({
                defaultMessage: 'Sort descending',
                description: 'Label for the sort descending button in the logged model list page',
              })}
            />
            <ToggleIconButton
              pressed={orderByAsc}
              icon={<ArrowUpIcon />}
              componentId="mlflow.logged_model.list.order_by.button_asc"
              onClick={() => onChangeOrderBy(orderByColumn, true)}
              aria-label={intl.formatMessage({
                defaultMessage: 'Sort ascending',
                description: 'Label for the sort ascending button in the logged model list page',
              })}
            />
          </div>
        </div>

        {groupedOrderByOptions.map(({ headerName, children, groupId }) => (
          <DropdownMenu.Group key={groupId} aria-label={headerName}>
            <DropdownMenu.Label>{headerName}</DropdownMenu.Label>
            {children?.map(({ headerName: columnHeaderName, colId }) => (
              <DropdownMenu.CheckboxItem
                key={colId}
                componentId="mlflow.logged_model.list.order_by.column_toggle"
                checked={orderByColumn === colId}
                onClick={() => {
                  if (!colId) {
                    return;
                  }
                  onChangeOrderBy(colId, Boolean(orderByAsc));
                }}
              >
                <DropdownMenu.ItemIndicator />
                {columnHeaderName}
              </DropdownMenu.CheckboxItem>
            ))}
          </DropdownMenu.Group>
        ))}
      </DropdownMenu.Content>
    </DropdownMenu.Root>
  );
};
