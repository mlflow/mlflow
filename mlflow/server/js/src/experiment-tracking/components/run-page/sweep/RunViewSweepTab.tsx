import {
  Empty,
  Table,
  TableCell,
  TableHeader,
  TableRow,
  Tag,
  Tooltip,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import type { ColumnDef } from '@tanstack/react-table';
import { flexRender, getCoreRowModel } from '@tanstack/react-table';
import { useReactTable_unverifiedWithReact18 as useReactTable } from '@databricks/web-shared/react-table';
import { useMemo } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';

import Utils from '../../../../common/utils/Utils';
import type { MetricEntitiesByName } from '../../../types';
import { useExperimentTrackingDetailsPageLayoutStyles } from '../../../hooks/useExperimentTrackingDetailsPageLayoutStyles';
import { findBestConfigs, parseSweepMetrics, type SweepConfigRow } from './parseSweepMetrics';

interface SweepTableRow {
  config: string;
  scorer: string;
  mean?: number;
  ciLow?: number;
  ciHigh?: number;
  std?: number;
  latencyP50?: number;
  latencyP90?: number;
  latencyP99?: number;
  costPerRequestUsd?: number;
  isBest: boolean;
}

const formatNumber = (value: number | undefined) => (value === undefined ? '-' : Utils.formatMetric(value));

const formatInterval = (ciLow: number | undefined, ciHigh: number | undefined) =>
  ciLow === undefined || ciHigh === undefined ? '-' : `[${Utils.formatMetric(ciLow)}, ${Utils.formatMetric(ciHigh)}]`;

const formatMilliseconds = (value: number | undefined) =>
  value === undefined ? '-' : `${Utils.formatMetric(value)} ms`;

// Per-request cost is often a fraction of a cent, so it needs more precision than formatMetric.
const formatCost = (value: number | undefined) => (value === undefined ? '-' : `$${value.toFixed(4)}`);

const buildRows = (configs: SweepConfigRow[], scorerNames: string[]): SweepTableRow[] => {
  const bestConfigsByScorer = new Map(scorerNames.map((scorer) => [scorer, new Set(findBestConfigs(configs, scorer))]));

  return configs.flatMap((row) =>
    Object.entries(row.scorersByName)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([scorer, stats]) => ({
        config: row.config,
        scorer,
        mean: stats.mean,
        ciLow: stats.ciLow,
        ciHigh: stats.ciHigh,
        std: stats.std,
        latencyP50: row.latency?.p50,
        latencyP90: row.latency?.p90,
        latencyP99: row.latency?.p99,
        costPerRequestUsd: row.costPerRequestUsd,
        isBest: bestConfigsByScorer.get(scorer)?.has(row.config) ?? false,
      })),
  );
};

const SweepConfigCell = ({ config, isBest }: { config: string; isBest: boolean }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  return (
    <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs, overflow: 'hidden' }}>
      {/* Config names can be long (a model id, or a user-chosen label), so the name truncates
          while the tag keeps its width — the tag is the part that must stay legible. */}
      <Typography.Text ellipsis title={config}>
        {config}
      </Typography.Text>
      {isBest && (
        <Tooltip
          componentId="mlflow.run-page.sweep-tab.best-tag-tooltip"
          content={intl.formatMessage({
            defaultMessage:
              'Highest mean for this scorer. Other configurations are also tagged when their confidence intervals overlap, since those results are not distinguishable at 95% confidence.',
            description: 'Run page > Evaluation sweep tab > Best configuration tag tooltip',
          })}
        >
          <Tag componentId="mlflow.run-page.sweep-tab.best-tag" color="turquoise" css={{ margin: 0, flexShrink: 0 }}>
            <FormattedMessage
              defaultMessage="Best"
              description="Run page > Evaluation sweep tab > Tag marking the best configuration for a scorer"
            />
          </Tag>
        </Tooltip>
      )}
    </div>
  );
};

/**
 * Run details page tab comparing the configurations of an `mlflow.genai.evaluate_sweep` run.
 *
 * Renders one row per (config, scorer) with the scorer's mean and 95% confidence interval
 * alongside the config's latency percentiles and per-request cost, so quality can be weighed
 * against cost and speed. The best config per scorer is tagged.
 *
 * The sweep flattens these statistics onto its parent run as summary metrics, so the tab reads
 * from `latestMetrics` and needs no extra fetch.
 */
export const RunViewSweepTab = ({ latestMetrics }: { latestMetrics?: MetricEntitiesByName }) => {
  const { theme } = useDesignSystemTheme();
  const { detailsPageTableStyles, detailsPageNoEntriesStyles } = useExperimentTrackingDetailsPageLayoutStyles();
  const intl = useIntl();

  const { configs, scorerNames } = useMemo(() => parseSweepMetrics(latestMetrics), [latestMetrics]);
  const rows = useMemo(() => buildRows(configs, scorerNames), [configs, scorerNames]);

  const columns = useMemo<ColumnDef<SweepTableRow>[]>(
    () => [
      {
        id: 'config',
        accessorKey: 'config',
        header: intl.formatMessage({
          defaultMessage: 'Configuration',
          description: 'Run page > Evaluation sweep tab > Configuration column header',
        }),
        enableResizing: true,
        size: 260,
        // eslint-disable-next-line @databricks/no-unstable-nested-components -- go/no-nested-components
        cell: ({ row }) => <SweepConfigCell config={row.original.config} isBest={row.original.isBest} />,
      },
      {
        id: 'scorer',
        accessorKey: 'scorer',
        header: intl.formatMessage({
          defaultMessage: 'Scorer',
          description: 'Run page > Evaluation sweep tab > Scorer column header',
        }),
        enableResizing: true,
        size: 160,
        cell: ({ row }) => row.original.scorer,
      },
      {
        id: 'mean',
        accessorKey: 'mean',
        header: intl.formatMessage({
          defaultMessage: 'Mean',
          description: 'Run page > Evaluation sweep tab > Scorer mean column header',
        }),
        enableResizing: true,
        cell: ({ row }) => formatNumber(row.original.mean),
      },
      {
        id: 'confidenceInterval',
        header: intl.formatMessage({
          defaultMessage: '95% CI',
          description: 'Run page > Evaluation sweep tab > Confidence interval column header',
        }),
        enableResizing: true,
        size: 160,
        cell: ({ row }) => formatInterval(row.original.ciLow, row.original.ciHigh),
      },
      {
        id: 'std',
        accessorKey: 'std',
        header: intl.formatMessage({
          defaultMessage: 'Std dev',
          description: 'Run page > Evaluation sweep tab > Standard deviation column header',
        }),
        enableResizing: true,
        cell: ({ row }) => formatNumber(row.original.std),
      },
      {
        id: 'costPerRequest',
        accessorKey: 'costPerRequestUsd',
        header: intl.formatMessage({
          defaultMessage: 'Cost/req',
          description: 'Run page > Evaluation sweep tab > Cost per request column header',
        }),
        enableResizing: true,
        cell: ({ row }) => formatCost(row.original.costPerRequestUsd),
      },
      {
        id: 'latencyP50',
        accessorKey: 'latencyP50',
        header: intl.formatMessage({
          defaultMessage: 'Latency p50',
          description: 'Run page > Evaluation sweep tab > Latency p50 column header',
        }),
        enableResizing: true,
        cell: ({ row }) => formatMilliseconds(row.original.latencyP50),
      },
      {
        id: 'latencyP90',
        accessorKey: 'latencyP90',
        header: intl.formatMessage({
          defaultMessage: 'Latency p90',
          description: 'Run page > Evaluation sweep tab > Latency p90 column header',
        }),
        enableResizing: true,
        cell: ({ row }) => formatMilliseconds(row.original.latencyP90),
      },
      {
        id: 'latencyP99',
        accessorKey: 'latencyP99',
        header: intl.formatMessage({
          defaultMessage: 'Latency p99',
          description: 'Run page > Evaluation sweep tab > Latency p99 column header',
        }),
        enableResizing: true,
        cell: ({ row }) => formatMilliseconds(row.original.latencyP99),
      },
    ],
    [intl],
  );

  const table = useReactTable<SweepTableRow>(
    'mlflow/server/js/src/experiment-tracking/components/run-page/sweep/RunViewSweepTab.tsx',
    {
      data: rows,
      getCoreRowModel: getCoreRowModel(),
      getRowId: (row) => `${row.config}.${row.scorer}`,
      enableColumnResizing: true,
      columnResizeMode: 'onChange',
      columns,
    },
  );

  if (rows.length === 0) {
    return (
      <div css={detailsPageNoEntriesStyles}>
        <Empty
          title={
            <FormattedMessage
              defaultMessage="No sweep results"
              description="Run page > Evaluation sweep tab > Empty state title"
            />
          }
          description={
            <FormattedMessage
              defaultMessage="This run has no evaluation sweep summary metrics. They are logged to the sweep's parent run when the sweep completes."
              description="Run page > Evaluation sweep tab > Empty state description"
            />
          }
        />
      </div>
    );
  }

  return (
    <div css={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', padding: theme.spacing.md }}>
      <Typography.Title level={4} css={{ flexShrink: 0 }}>
        <FormattedMessage
          defaultMessage="Configuration comparison ({count})"
          description="Run page > Evaluation sweep tab > Section title with the number of configurations"
          values={{ count: configs.length }}
        />
      </Typography.Title>
      <Typography.Paragraph css={{ flexShrink: 0 }}>
        <FormattedMessage
          defaultMessage="Each configuration was evaluated repeatedly to estimate a confidence interval per scorer. Latency and cost are per request, aggregated across every repeat."
          description="Run page > Evaluation sweep tab > Section description"
        />
      </Typography.Paragraph>
      <Table scrollable css={detailsPageTableStyles}>
        <TableRow isHeader>
          {table.getLeafHeaders().map((header) => (
            <TableHeader
              componentId="mlflow.run-page.sweep-tab.table-header"
              key={header.id}
              header={header}
              column={header.column}
              setColumnSizing={table.setColumnSizing}
              isResizing={header.column.getIsResizing()}
              style={{ flex: header.column.getCanResize() ? header.column.getSize() / 100 : undefined }}
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
                style={{ flex: cell.column.getCanResize() ? cell.column.getSize() / 100 : undefined }}
              >
                {flexRender(cell.column.columnDef.cell, cell.getContext())}
              </TableCell>
            ))}
          </TableRow>
        ))}
      </Table>
    </div>
  );
};
