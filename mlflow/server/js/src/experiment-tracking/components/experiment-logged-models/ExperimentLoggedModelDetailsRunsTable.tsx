import {
  Empty,
  Input,
  Overflow,
  SearchIcon,
  Table,
  TableCell,
  TableHeader,
  TableRow,
  TableSkeleton,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { useIntl } from 'react-intl';
import type { CellContext, ColumnDef, ColumnDefTemplate } from '@tanstack/react-table';
import { flexRender, getCoreRowModel, getExpandedRowModel, useReactTable } from '@tanstack/react-table';
import { entries, groupBy, isEmpty, uniqBy } from 'lodash';
import { useMemo, useState } from 'react';
import { FormattedMessage } from 'react-intl';
import type { LoggedModelProto, RunEntity } from '../../types';
import { ExperimentLoggedModelDetailsTableRunCellRenderer } from './ExperimentLoggedModelDetailsTableRunCellRenderer';
import { ExperimentLoggedModelDatasetButton } from './ExperimentLoggedModelDatasetButton';
import { useExperimentTrackingDetailsPageLayoutStyles } from '../../hooks/useExperimentTrackingDetailsPageLayoutStyles';

interface RunsTableRow {
  experimentId?: string;
  runName?: string;
  runId: string;
  datasets: {
    datasetName: string;
    datasetDigest: string;
    runId: string;
  }[];
}

type RunsTableCellRenderer = ColumnDefTemplate<CellContext<RunsTableRow, unknown>>;

const DatasetListCellRenderer = ({ getValue }: CellContext<RunsTableRow, RunsTableRow['datasets']>) => {
  const datasets = getValue() ?? [];

  if (isEmpty(datasets)) {
    return <>-</>;
  }

  return (
    <Overflow>
      {datasets.map(({ datasetDigest, datasetName, runId }) => (
        <ExperimentLoggedModelDatasetButton
          datasetName={datasetName}
          datasetDigest={datasetDigest}
          runId={runId}
          key={[datasetName, datasetDigest].join('.')}
        />
      ))}
    </Overflow>
  );
};

export const ExperimentLoggedModelDetailsPageRunsTable = ({
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

  const runsWithDatasets = useMemo(() => {
    if (relatedRunsLoading) {
      return [];
    }
    const allMetrics = loggedModel?.data?.metrics ?? [];
    const runsByDatasets = groupBy(allMetrics, 'run_id');
    if (loggedModel?.info?.source_run_id && !runsByDatasets[loggedModel.info.source_run_id]) {
      runsByDatasets[loggedModel.info.source_run_id] = [];
    }
    return entries(runsByDatasets).map(([runId, metrics]) => {
      // Locate unique dataset entries
      const distinctDatasets = uniqBy(metrics, 'dataset_name')
        .map(({ dataset_digest, dataset_name }) => ({
          datasetDigest: dataset_digest,
          datasetName: dataset_name,
          runId,
        }))
        .filter((dataset) => Boolean(dataset.datasetName) || Boolean(dataset.datasetDigest));

      const runName = relatedRunsData?.find((run) => run.info?.runUuid === runId)?.info?.runName;
      return {
        runId,
        runName,
        datasets: distinctDatasets,
        experimentId: loggedModel?.info?.experiment_id,
      };
    });
  }, [loggedModel, relatedRunsLoading, relatedRunsData]);

  const filteredRunsWithDatasets = useMemo(
    () =>
      runsWithDatasets.filter(({ runName, datasets }) => {
        const filterLower = filter.toLowerCase();
        return (
          runName?.toLowerCase().includes(filterLower) ||
          datasets.find((d) => d.datasetName?.toLowerCase().includes(filterLower))
        );
      }),
    [filter, runsWithDatasets],
  );

  const columns = useMemo<ColumnDef<any>[]>(
    () => [
      {
        id: 'run',
        header: intl.formatMessage({
          defaultMessage: 'Run',
          description: 'Column header for the run name in the runs table on the logged model details page',
        }),
        enableResizing: true,
        size: 240,
        accessorFn: ({ runId, runName, experimentId }) => ({
          runId,
          runName,
          experimentId,
        }),
        cell: ExperimentLoggedModelDetailsTableRunCellRenderer as RunsTableCellRenderer,
      },
      {
        id: 'input',
        header: intl.formatMessage({
          defaultMessage: 'Input',
          description: 'Column header for the input in the runs table on the logged model details page',
        }),
        accessorKey: 'datasets',
        enableResizing: false,
        cell: DatasetListCellRenderer as RunsTableCellRenderer,
      },
    ],
    [intl],
  );

  const table = useReactTable({
    data: filteredRunsWithDatasets,
    getCoreRowModel: getCoreRowModel(),
    getExpandedRowModel: getExpandedRowModel(),
    getRowId: (row) => row.key,
    enableColumnResizing: true,
    columnResizeMode: 'onChange',
    columns,
  });

  const renderTableContent = () => {
    if (relatedRunsLoading) {
      return <TableSkeleton lines={3} />;
    }
    if (!runsWithDatasets.length) {
      return (
        <div css={detailsPageNoEntriesStyles}>
          <Empty
            description={
              <FormattedMessage
                defaultMessage="No runs"
                description="Placeholder text for the runs table on the logged model details page when there are no runs"
              />
            }
          />
        </div>
      );
    }

    const areAllResultsFiltered = filteredRunsWithDatasets.length < 1;

    return (
      <>
        <div css={{ marginBottom: theme.spacing.sm }}>
          <Input
            componentId="mlflow.logged_model.details.runs.table.search"
            prefix={<SearchIcon />}
            placeholder={intl.formatMessage({
              defaultMessage: 'Search runs',
              description: 'Placeholder text for the search input in the runs table on the logged model details page',
            })}
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            allowClear
          />
        </div>
        <Table
          scrollable
          ref={(element) => element?.setAttribute('data-testid', 'logged-model-details-runs-table')}
          empty={
            areAllResultsFiltered ? (
              <div>
                <Empty
                  description={
                    <FormattedMessage
                      defaultMessage="No runs match the search filter"
                      description="No results message for the runs table on the logged model details page"
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
                componentId="mlflow.logged_model.details.runs.table.header"
                key={header.id}
                header={header}
                column={header.column}
                setColumnSizing={table.setColumnSizing}
                isResizing={header.column.getIsResizing()}
                css={{
                  flexGrow: header.column.getCanResize() ? 0 : 1,
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
                  multiline
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
      <Typography.Title css={{ fontSize: 16 }}>Runs</Typography.Title>
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
