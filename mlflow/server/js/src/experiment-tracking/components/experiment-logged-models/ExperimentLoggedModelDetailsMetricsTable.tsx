import {
  Empty,
  Input,
  SearchIcon,
  Table,
  TableCell,
  TableHeader,
  TableIcon,
  TableRow,
  TableSkeleton,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { useMemo, useState } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import {
  type CellContext,
  type ColumnDef,
  type ColumnDefTemplate,
  flexRender,
  getCoreRowModel,
  getExpandedRowModel,
  useReactTable,
} from '@tanstack/react-table';
import type { LoggedModelProto, LoggedModelMetricProto, RunEntity } from '../../types';
import { ExperimentLoggedModelDetailsTableRunCellRenderer } from './ExperimentLoggedModelDetailsTableRunCellRenderer';
import { ExperimentLoggedModelDatasetButton } from './ExperimentLoggedModelDatasetButton';
import { useExperimentTrackingDetailsPageLayoutStyles } from '../../hooks/useExperimentTrackingDetailsPageLayoutStyles';

interface LoggedModelMetricWithRunData extends LoggedModelMetricProto {
  experimentId?: string | null;
  runName?: string | null;
}

type MetricTableCellRenderer = ColumnDefTemplate<CellContext<LoggedModelMetricWithRunData, unknown>>;
type ColumnMeta = {
  styles?: React.CSSProperties;
};

const SingleDatasetCellRenderer = ({
  getValue,
}: CellContext<
  LoggedModelMetricProto,
  {
    datasetName: string;
    datasetDigest: string;
    runId: string | null;
  }
>) => {
  const { datasetDigest, datasetName, runId } = getValue();

  if (!datasetName) {
    return '-';
  }

  return <ExperimentLoggedModelDatasetButton datasetName={datasetName} datasetDigest={datasetDigest} runId={runId} />;
};

export const ExperimentLoggedModelDetailsMetricsTable = ({
  loggedModel,
  relatedRunsData,
  relatedRunsLoading,
}: {
  loggedModel?: LoggedModelProto;
  relatedRunsData?: RunEntity[];
  relatedRunsLoading?: boolean;
}) => {
  const { theme } = useDesignSystemTheme();
  const { detailsPageTableStyles, detailsPageNoEntriesStyles } = useExperimentTrackingDetailsPageLayoutStyles();
  const intl = useIntl();
  const [filter, setFilter] = useState('');

  const metricsWithRunData = useMemo(() => {
    if (relatedRunsLoading) {
      return [];
    }
    return (
      loggedModel?.data?.metrics?.map((metric) => {
        const runName = relatedRunsData?.find((run) => run.info?.runUuid === metric.run_id)?.info?.runName;
        return {
          ...metric,
          experimentId: loggedModel.info?.experiment_id,
          runName,
        };
      }) ?? []
    );
  }, [loggedModel, relatedRunsLoading, relatedRunsData]);

  const filteredMetrics = useMemo(
    () =>
      metricsWithRunData.filter(({ key, dataset_name, dataset_digest, runName }) => {
        const filterLower = filter.toLowerCase();
        return (
          key?.toLowerCase().includes(filterLower) ||
          dataset_name?.toLowerCase().includes(filterLower) ||
          dataset_digest?.toLowerCase().includes(filterLower) ||
          runName?.toLowerCase().includes(filterLower)
        );
      }),
    [filter, metricsWithRunData],
  );

  const columns = useMemo<ColumnDef<LoggedModelMetricWithRunData>[]>(
    () => [
      {
        id: 'metric',
        accessorKey: 'key',
        header: intl.formatMessage({
          defaultMessage: 'Metric',
          description: 'Label for the metric column in the logged model details metrics table',
        }),
        enableResizing: true,
        size: 240,
      },
      {
        id: 'dataset',
        header: intl.formatMessage({
          defaultMessage: 'Dataset',
          description: 'Label for the dataset column in the logged model details metrics table',
        }),
        accessorFn: ({ dataset_name: datasetName, dataset_digest: datasetDigest, run_id: runId }) => ({
          datasetName,
          datasetDigest,
          runId,
        }),
        enableResizing: true,
        cell: SingleDatasetCellRenderer as MetricTableCellRenderer,
      },
      {
        id: 'sourceRun',
        header: intl.formatMessage({
          defaultMessage: 'Source run',
          description:
            "Label for the column indicating a run being the source of the logged model's metric (i.e. source run). Displayed in the logged model details metrics table.",
        }),
        accessorFn: ({ run_id: runId, runName, experimentId }) => ({
          runId,
          runName,
          experimentId,
        }),
        enableResizing: true,
        cell: ExperimentLoggedModelDetailsTableRunCellRenderer as MetricTableCellRenderer,
      },
      {
        id: 'value',
        header: intl.formatMessage({
          defaultMessage: 'Value',
          description: 'Label for the value column in the logged model details metrics table',
        }),
        accessorKey: 'value',
        // In full-width layout, let "Value" fill the remaining space
        enableResizing: true,
        meta: {
          styles: {
            minWidth: 120,
          },
        },
      },
    ],
    [intl],
  );

  const table = useReactTable({
    data: filteredMetrics,
    getCoreRowModel: getCoreRowModel(),
    getExpandedRowModel: getExpandedRowModel(),
    getRowId: (row) => [row.key, row.dataset_digest, row.run_id].join('.') ?? '',
    enableColumnResizing: true,
    columnResizeMode: 'onChange',
    columns,
  });

  const renderTableContent = () => {
    if (relatedRunsLoading) {
      return <TableSkeleton lines={3} />;
    }
    if (!metricsWithRunData.length) {
      return (
        <div css={detailsPageNoEntriesStyles}>
          <Empty
            description={
              <FormattedMessage
                defaultMessage="No metrics recorded"
                description="Placeholder text when no metrics are recorded for a logged model"
              />
            }
          />
        </div>
      );
    }

    const areAllResultsFiltered = filteredMetrics.length < 1;

    return (
      <>
        <div css={{ marginBottom: theme.spacing.sm }}>
          <Input
            componentId="mlflow.logged_model.details.metrics.table.search"
            prefix={<SearchIcon />}
            placeholder={intl.formatMessage({
              defaultMessage: 'Search metrics',
              description: 'Placeholder text for the search input in the logged model details metrics table',
            })}
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            allowClear
          />
        </div>
        <Table
          ref={(element) => element?.setAttribute('data-testid', 'logged-model-details-metrics-table')}
          scrollable
          empty={
            areAllResultsFiltered ? (
              <div>
                <Empty
                  description={
                    <FormattedMessage
                      defaultMessage="No metrics match the search filter"
                      description="Message displayed when no metrics match the search filter in the logged model details metrics table"
                    />
                  }
                />
              </div>
            ) : null
          }
          css={detailsPageTableStyles}
        >
          <TableRow isHeader>
            {table.getLeafHeaders().map((header, index) => (
              <TableHeader
                componentId="mlflow.logged_model.details.metrics.table.header"
                key={header.id}
                header={header}
                column={header.column}
                setColumnSizing={table.setColumnSizing}
                isResizing={header.column.getIsResizing()}
                css={{
                  flexGrow: header.column.getCanResize() ? 0 : 1,
                  ...(header.column.columnDef.meta as ColumnMeta)?.styles,
                }}
                style={{
                  flexBasis: header.column.getCanResize() ? header.column.getSize() : undefined,
                }}
              >
                {flexRender(header.column.columnDef.header, header.getContext())}
              </TableHeader>
            ))}
          </TableRow>
          {table.getRowModel().rows.map((row) => (
            <TableRow key={row.id}>
              {row.getAllCells().map((cell) => (
                <TableCell
                  key={cell.id}
                  style={{
                    flexGrow: cell.column.getCanResize() ? 0 : 1,
                    flexBasis: cell.column.getCanResize() ? cell.column.getSize() : undefined,
                  }}
                  css={{
                    ...(cell.column.columnDef.meta as ColumnMeta)?.styles,
                  }}
                >
                  {flexRender(cell.column.columnDef.cell, cell.getContext())}
                </TableCell>
              ))}
            </TableRow>
          ))}
        </Table>
      </>
    );
  };

  return (
    <div css={{ display: 'flex', flexDirection: 'column', overflow: 'hidden', maxHeight: 400 }}>
      <Typography.Title level={4}>
        <FormattedMessage
          defaultMessage="Metrics ({length})"
          description="Header for the metrics table on the logged model details page. (Length) is the number of metrics currently displayed."
          values={{ length: metricsWithRunData.length }}
        />
      </Typography.Title>
      <div
        css={{
          padding: theme.spacing.sm,
          border: `1px solid ${theme.colors.border}`,
          borderRadius: theme.general.borderRadiusBase,
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
        }}
      >
        {renderTableContent()}
      </div>
    </div>
  );
};
