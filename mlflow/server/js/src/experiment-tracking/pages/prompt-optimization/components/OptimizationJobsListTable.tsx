import { useReactTable_unverifiedWithReact18 as useReactTable } from '@databricks/web-shared/react-table';
import {
  Empty,
  Table,
  TableCell,
  TableHeader,
  TableRow,
  TableSkeletonRows,
  useDesignSystemTheme,
} from '@databricks/design-system';
import type { ColumnDef } from '@tanstack/react-table';
import { flexRender, getCoreRowModel, getSortedRowModel } from '@tanstack/react-table';
import { useMemo, useCallback } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import { useNavigate } from '../../../../common/utils/RoutingUtils';
import Routes from '../../../routes';
import type { PromptOptimizationJob } from '../types';
import { getOptimizerTypeName } from '../types';
import { OptimizationJobNameCell } from './OptimizationJobNameCell';
import { OptimizationJobStatusCell } from './OptimizationJobStatusCell';
import { OptimizationJobActionsCell } from './OptimizationJobActionsCell';
import { OptimizationJobDatasetCell } from './OptimizationJobDatasetCell';
import { OptimizationJobPromptCell } from './OptimizationJobPromptCell';
import Utils from '../../../../common/utils/Utils';
import { isEmpty } from 'lodash';

export interface OptimizationJobsTableMetadata {
  experimentId: string;
  onCancelJob: (jobId: string) => void;
  onDeleteJob: (jobId: string) => void;
  getDatasetName: (datasetId: string | undefined) => string | undefined;
}

type OptimizationJobsTableColumnDef = ColumnDef<PromptOptimizationJob>;

const useOptimizationJobsTableColumns = () => {
  const intl = useIntl();
  return useMemo(() => {
    const resultColumns: OptimizationJobsTableColumnDef[] = [
      {
        header: intl.formatMessage({
          defaultMessage: 'Job ID',
          description: 'Header for the job ID column in the optimization jobs table',
        }),
        accessorKey: 'job_id',
        id: 'name',
        cell: OptimizationJobNameCell,
      },
      {
        header: intl.formatMessage({
          defaultMessage: 'Source Prompt',
          description: 'Header for the source prompt column in the optimization jobs table',
        }),
        id: 'sourcePrompt',
        cell: ({ row: { original } }) => <OptimizationJobPromptCell promptUri={original.source_prompt_uri} />,
      },
      {
        header: intl.formatMessage({
          defaultMessage: 'Optimized Prompt',
          description: 'Header for the optimized prompt column in the optimization jobs table',
        }),
        id: 'optimizedPrompt',
        cell: ({ row: { original } }) => <OptimizationJobPromptCell promptUri={original.optimized_prompt_uri} />,
      },
      {
        header: intl.formatMessage({
          defaultMessage: 'Dataset',
          description: 'Header for the dataset column in the optimization jobs table',
        }),
        id: 'dataset',
        cell: OptimizationJobDatasetCell,
      },
      {
        header: intl.formatMessage({
          defaultMessage: 'Optimizer',
          description: 'Header for the optimizer column in the optimization jobs table',
        }),
        id: 'optimizer',
        accessorFn: (job) => getOptimizerTypeName(job.config?.optimizer_type),
      },
      {
        header: intl.formatMessage({
          defaultMessage: 'Status',
          description: 'Header for the status column in the optimization jobs table',
        }),
        id: 'status',
        cell: OptimizationJobStatusCell,
      },
      {
        header: intl.formatMessage({
          defaultMessage: 'Created',
          description: 'Header for the created column in the optimization jobs table',
        }),
        id: 'created',
        accessorFn: (job) => job.creation_timestamp_ms,
        cell: ({ row: { original } }) =>
          original.creation_timestamp_ms ? Utils.formatTimestamp(original.creation_timestamp_ms, intl) : '-',
      },
      {
        header: '',
        id: 'actions',
        cell: OptimizationJobActionsCell,
      },
    ];

    return resultColumns;
  }, [intl]);
};

export const OptimizationJobsListTable = ({
  jobs,
  isLoading,
  experimentId,
  onCancelJob,
  onDeleteJob,
  getDatasetName,
}: {
  jobs?: PromptOptimizationJob[];
  isLoading?: boolean;
  experimentId: string;
  onCancelJob: (jobId: string) => void;
  onDeleteJob: (jobId: string) => void;
  getDatasetName: (datasetId: string | undefined) => string | undefined;
}) => {
  const { theme } = useDesignSystemTheme();
  const navigate = useNavigate();
  const columns = useOptimizationJobsTableColumns();

  // Sort jobs by creation time, latest first
  const sortedJobs = useMemo(() => {
    if (!jobs) return [];
    return [...jobs].sort((a, b) => {
      const timeA = a.creation_timestamp_ms ?? 0;
      const timeB = b.creation_timestamp_ms ?? 0;
      return timeB - timeA;
    });
  }, [jobs]);

  const handleRowClick = useCallback(
    (job: PromptOptimizationJob) => {
      if (job.job_id) {
        navigate(Routes.getPromptOptimizationDetailsPageRoute(experimentId, job.job_id));
      }
    },
    [experimentId, navigate],
  );

  const table = useReactTable(
    'mlflow/server/js/src/experiment-tracking/pages/prompt-optimization/components/OptimizationJobsListTable.tsx',
    {
      data: sortedJobs,
      columns,
      getCoreRowModel: getCoreRowModel(),
      getSortedRowModel: getSortedRowModel(),
      getRowId: (row, index) => row.job_id ?? index.toString(),
      meta: { experimentId, onCancelJob, onDeleteJob, getDatasetName } satisfies OptimizationJobsTableMetadata,
    },
  );

  const getEmptyState = () => {
    const isEmptyList = !isLoading && isEmpty(jobs);
    if (isEmptyList) {
      return (
        <Empty
          title={
            <FormattedMessage
              defaultMessage="No optimization jobs"
              description="A header for the empty state in the optimization jobs table"
            />
          }
          description={
            <FormattedMessage
              defaultMessage='Use "Create new optimization" button to start optimizing your prompts'
              description="Guidelines for the user on how to create an optimization job"
            />
          }
        />
      );
    }

    return null;
  };

  return (
    <Table scrollable empty={getEmptyState()}>
      <TableRow isHeader>
        {table.getLeafHeaders().map((header) => (
          <TableHeader componentId="mlflow.prompt-optimization.list.table.header" key={header.id}>
            {flexRender(header.column.columnDef.header, header.getContext())}
          </TableHeader>
        ))}
      </TableRow>
      {isLoading ? (
        <TableSkeletonRows table={table} />
      ) : (
        table.getRowModel().rows.map((row) => (
          <TableRow
            key={row.id}
            onClick={() => handleRowClick(row.original)}
            css={{
              height: theme.general.buttonHeight,
              cursor: 'pointer',
              '&:hover': {
                backgroundColor: theme.colors.actionTertiaryBackgroundHover,
              },
            }}
          >
            {row.getAllCells().map((cell) => (
              <TableCell key={cell.id} css={{ alignItems: 'center' }}>
                {flexRender(cell.column.columnDef.cell, cell.getContext())}
              </TableCell>
            ))}
          </TableRow>
        ))
      )}
    </Table>
  );
};
