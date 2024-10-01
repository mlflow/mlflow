import {
  Empty,
  Input,
  SearchIcon,
  Table,
  TableCell,
  TableHeader,
  TableRow,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { MetricEntitiesByName, MetricEntity, RunInfoEntity } from '../../../types';
import { sum, values } from 'lodash';
import { useMemo, useState } from 'react';
import { Link } from '../../../../common/utils/RoutingUtils';
import Routes from '../../../routes';
import { FormattedMessage, defineMessages, useIntl } from 'react-intl';
import { isSystemMetricKey } from '../../../utils/MetricsUtils';
import { Table as TableDef, flexRender, getCoreRowModel, useReactTable } from '@tanstack/react-table';
import type { UseGetRunQueryResponseRunInfo } from '../hooks/useGetRunQuery';

const { systemMetricsLabel, modelMetricsLabel } = defineMessages({
  systemMetricsLabel: {
    defaultMessage: 'System metrics',
    description: 'Run page > Overview > Metrics table > System charts section > title',
  },
  modelMetricsLabel: {
    defaultMessage: 'Model metrics',
    description: 'Run page > Overview > Metrics table > Model charts section > title',
  },
});

const metricKeyMatchesFilter =
  (filter: string) =>
  ({ key }: MetricEntity) =>
    key.toLowerCase().includes(filter.toLowerCase());

const RunViewMetricsTableSection = ({
  metricsList,
  runInfo,
  header,
  table,
}: {
  runInfo: RunInfoEntity | UseGetRunQueryResponseRunInfo;
  metricsList: MetricEntity[];
  header?: React.ReactNode;
  table: TableDef<MetricEntity>;
}) => {
  const { theme } = useDesignSystemTheme();
  const [{ column: keyColumn }] = table.getLeafHeaders();
  return metricsList.length ? (
    <>
      {header && (
        <TableRow>
          <TableCell css={{ flex: 1, backgroundColor: theme.colors.backgroundSecondary }}>
            <Typography.Text bold>
              {header} ({metricsList.length})
            </Typography.Text>
          </TableCell>
        </TableRow>
      )}
      {metricsList.map(
        ({
          // Get metric key and value to display in table
          key,
          value,
        }) => (
          <TableRow key={key}>
            <TableCell
              style={{
                flexGrow: 0,
                flexBasis: keyColumn.getSize(),
              }}
            >
              <Link to={Routes.getMetricPageRoute([runInfo.runUuid ?? ''], key, [runInfo.experimentId ?? ''])}>
                {key}
              </Link>
            </TableCell>
            <TableCell
              css={{
                flexGrow: 1,
              }}
            >
              {value.toString()}
            </TableCell>
          </TableRow>
        ),
      )}
    </>
  ) : null;
};

/**
 * Displays table with metrics key/values in run detail overview.
 */
export const RunViewMetricsTable = ({
  latestMetrics,
  runInfo,
}: {
  latestMetrics: MetricEntitiesByName;
  runInfo: RunInfoEntity | UseGetRunQueryResponseRunInfo;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [filter, setFilter] = useState('');

  const metricValues = useMemo(() => values(latestMetrics), [latestMetrics]);

  const columns = useMemo(
    () => [
      {
        id: 'key',
        accessorKey: 'key',
        header: () => (
          <FormattedMessage
            defaultMessage="Metric"
            description="Run page > Overview > Metrics table > Key column header"
          />
        ),
        enableResizing: true,
        size: 240,
      },
      {
        id: 'value',
        header: () => (
          <FormattedMessage
            defaultMessage="Value"
            description="Run page > Overview > Metrics table > Value column header"
          />
        ),
        accessorKey: 'value',
        enableResizing: false,
      },
    ],
    [],
  );

  // Break down metric lists into system and model segments. If no system (or model) metrics
  // are detected, return a single segment.
  const metricSegments = useMemo(() => {
    const systemMetrics = metricValues.filter(({ key }) => isSystemMetricKey(key));
    const modelMetrics = metricValues.filter(({ key }) => !isSystemMetricKey(key));
    const isSegmented = systemMetrics.length > 0 && modelMetrics.length > 0;
    if (!isSegmented) {
      return [{ header: undefined, metrics: metricValues.filter(metricKeyMatchesFilter(filter)) }];
    }
    return [
      {
        header: intl.formatMessage(systemMetricsLabel),
        metrics: systemMetrics.filter(metricKeyMatchesFilter(filter)),
      },
      {
        header: intl.formatMessage(modelMetricsLabel),
        metrics: modelMetrics.filter(metricKeyMatchesFilter(filter)),
      },
    ];
  }, [filter, metricValues, intl]);

  const table = useReactTable<MetricEntity>({
    data: metricValues,
    getCoreRowModel: getCoreRowModel(),
    getRowId: (row) => row.key,
    enableColumnResizing: true,
    columnResizeMode: 'onChange',
    columns,
  });

  const renderTableContent = () => {
    if (!metricValues.length) {
      return (
        <div css={{ flex: '1', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <Empty
            description={
              <FormattedMessage
                defaultMessage="No metrics recorded"
                description="Run page > Overview > Metrics table > No metrics recorded"
              />
            }
          />
        </div>
      );
    }

    const areAllResultsFiltered = sum(metricSegments.map(({ metrics }) => metrics.length)) < 1;

    return (
      <>
        <div css={{ marginBottom: theme.spacing.sm }}>
          <Input
            componentId="codegen_mlflow_app_src_experiment-tracking_components_run-page_overview_runviewmetricstable.tsx_186"
            prefix={<SearchIcon />}
            placeholder={intl.formatMessage({
              defaultMessage: 'Search metrics',
              description: 'Run page > Overview > Metrics table > Filter input placeholder',
            })}
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            allowClear
          />
        </div>

        <Table
          scrollable
          empty={
            areAllResultsFiltered ? (
              <div css={{ marginTop: theme.spacing.md * 4 }}>
                <Empty
                  description={
                    <FormattedMessage
                      defaultMessage="No metrics match the search filter"
                      description="Run page > Overview > Metrics table > No results after filtering"
                    />
                  }
                />
              </div>
            ) : null
          }
        >
          <TableRow isHeader>
            {table.getLeafHeaders().map((header) => (
              <TableHeader
                key={header.id}
                resizable={header.column.getCanResize()}
                resizeHandler={header.getResizeHandler()}
                isResizing={header.column.getIsResizing()}
                style={{
                  flexGrow: header.column.getCanResize() ? 0 : 1,
                  flexBasis: header.column.getCanResize() ? header.column.getSize() : undefined,
                }}
              >
                {flexRender(header.column.columnDef.header, header.getContext())}
              </TableHeader>
            ))}
          </TableRow>
          {metricSegments.map((segment, index) => (
            <RunViewMetricsTableSection
              key={segment.header || index}
              metricsList={segment.metrics}
              runInfo={runInfo}
              header={segment.header}
              table={table}
            />
          ))}
        </Table>
      </>
    );
  };
  return (
    <div css={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      <Typography.Title level={4} css={{ flexShrink: 0 }}>
        <FormattedMessage
          defaultMessage="Metrics ({length})"
          description="Run page > Overview > Metrics table > Section title"
          values={{ length: metricValues.filter(metricKeyMatchesFilter(filter)).length }}
        />
      </Typography.Title>
      <div
        css={{
          padding: theme.spacing.sm,
          border: `1px solid ${theme.colors.borderDecorative}`,
          borderRadius: theme.general.borderRadiusBase,
          display: 'flex',
          flexDirection: 'column',
          flex: 1,
          overflow: 'hidden',
        }}
      >
        {renderTableContent()}
      </div>
    </div>
  );
};
