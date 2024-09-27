import { ExperimentRunsSelectorResult } from '@mlflow/mlflow/src/experiment-tracking/components/experiment-page/utils/experimentRuns.selector';
import {
  DifferenceCardAttributes,
  DifferenceCardConfigCompareGroup,
  RunsChartsCardConfig,
  RunsChartsDifferenceCardConfig,
} from '../../runs-charts.types';
import { RunsChartsRunData } from '../RunsCharts.common';
import { ReactChild, ReactFragment, ReactPortal, useCallback, useMemo, useState } from 'react';
import { MLFLOW_SYSTEM_METRIC_PREFIX } from '@mlflow/mlflow/src/experiment-tracking/constants';
import { ColumnDef, flexRender, getCoreRowModel, useReactTable } from '@tanstack/react-table';
import {
  Table,
  TableCell,
  TableHeader,
  TableRow,
  useDesignSystemTheme,
  Typography,
  Tag,
  LegacyTooltip,
} from '@databricks/design-system';
import { RunColorPill } from '@mlflow/mlflow/src/experiment-tracking/components/experiment-page/components/RunColorPill';
import { FormattedMessage, useIntl } from 'react-intl';
import { MetricEntitiesByName } from '@mlflow/mlflow/src/experiment-tracking/types';
import {
  differenceView,
  getDifferenceViewDataGroups,
  getDifferenceChartDisplayedValue,
} from '../../utils/differenceView';
import { OverflowIcon, Button, DropdownMenu } from '@databricks/design-system';
import Utils from '@mlflow/mlflow/src/common/utils/Utils';
import { TableSkeletonRows } from '@databricks/design-system';
import { ArrowUpIcon } from '@databricks/design-system';
import { ArrowDownIcon } from '@databricks/design-system';
import type { RunsGroupByConfig } from '../../../experiment-page/utils/experimentPage.group-row-utils';

const HEADING_COLUMN_ID = 'headingColumn';
const COLUMN_WIDTH = 200;

/**
 * A placeholder component displayed when no difference view data is available to be displayed
 */

const NoDifferenceDataPlaceholder = () => {
  const { theme } = useDesignSystemTheme();

  return (
    <div css={{ display: 'flex', flex: 1, justifyContent: 'center', alignItems: 'center' }}>
      <div css={{ display: 'flex', flexDirection: 'column', alignItems: 'center', maxWidth: 360 }}>
        <Typography.Title css={{ marginTop: theme.spacing.md, textAlign: 'center' }} color="secondary" level={3}>
          <FormattedMessage
            defaultMessage="No run differences to display"
            description="Experiment tracking > runs charts > charts > DifferenceViewPlot > no runs to display warning > title"
          />
        </Typography.Title>
        <Typography.Text css={{ marginBottom: theme.spacing.md, textAlign: 'center' }} color="secondary">
          <FormattedMessage
            defaultMessage="Select other runs, toggle 'Show differences only', or re-configure the chart to compare model metrics, system metrics, parameters, tag, or attributes."
            description="Experiment tracking > runs charts > charts > DifferenceViewPlot > no runs to display warning > text"
          />
        </Typography.Text>
      </div>
    </div>
  );
};

export const DifferenceViewPlot = ({
  previewData,
  cardConfig,
  groupBy,
  setCardConfig,
}: {
  previewData: RunsChartsRunData[];
  cardConfig: RunsChartsDifferenceCardConfig;
  groupBy: RunsGroupByConfig | null;
  setCardConfig?: (setter: (current: RunsChartsCardConfig) => RunsChartsDifferenceCardConfig) => void;
}) => {
  const { formatMessage } = useIntl();
  const { theme } = useDesignSystemTheme();

  const { modelMetrics, systemMetrics, parameters, tags, attributes } = useMemo(
    () => getDifferenceViewDataGroups(previewData, cardConfig, HEADING_COLUMN_ID, groupBy),
    [previewData, cardConfig, groupBy],
  );
  // Each metric/param is a row in the table
  const getData = [
    ...(cardConfig.compareGroups.includes(DifferenceCardConfigCompareGroup.MODEL_METRICS) ? modelMetrics : []),
    ...(cardConfig.compareGroups.includes(DifferenceCardConfigCompareGroup.SYSTEM_METRICS) ? systemMetrics : []),
    ...(cardConfig.compareGroups.includes(DifferenceCardConfigCompareGroup.PARAMETERS) ? parameters : []),
    ...(cardConfig.compareGroups.includes(DifferenceCardConfigCompareGroup.ATTRIBUTES) ? attributes : []),
    ...(cardConfig.compareGroups.includes(DifferenceCardConfigCompareGroup.TAGS) ? tags : []),
  ];

  const { baselineColumn, nonBaselineColumns } = useMemo(() => {
    // baseline column (can be undefined if no baseline selected)
    let baselineColumn = previewData.find((runData) => runData.uuid === cardConfig.baselineColumnUuid);
    if (baselineColumn === undefined && previewData.length > 0) {
      // Set the first column as baseline column
      baselineColumn = previewData[0];
    }
    // non-baseline columns
    const nonBaselineColumns = previewData.filter(
      (runData) => baselineColumn === undefined || runData.uuid !== baselineColumn.uuid,
    );

    return { baselineColumn, nonBaselineColumns };
  }, [previewData, cardConfig.baselineColumnUuid]);

  // Split columns into baseline/non-baseline
  const getColumns = useMemo(() => {
    const convertRunToColumnInfo = (runData: RunsChartsRunData, isBaseline: boolean) => {
      const accessorFn = (row: Record<string, string | number>) => {
        return {
          text: row[runData.uuid],
          difference: baselineColumn ? differenceView(row[runData.uuid], row[baselineColumn.uuid])?.label : null,
        };
      };

      const baselineAccessorFn = (row: Record<string, string | number>) => {
        return row[runData.uuid];
      };
      if (isBaseline) {
        return {
          id: runData.uuid,
          header: () => {
            return (
              <span css={{ display: 'flex', flexDirection: 'row', justifyContent: 'space-between' }}>
                <span css={{ display: 'inline-flex', gap: theme.spacing.sm, alignItems: 'center' }}>
                  <RunColorPill color={runData.color} />
                  {runData.displayName}
                  <Tag
                    componentId="codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_charts_differenceviewplot.tsx_132"
                    css={{ margin: 0 }}
                  >
                    <FormattedMessage
                      defaultMessage="baseline"
                      description="Runs charts > components > charts > DifferenceViewPlot > baseline tag"
                    />
                  </Tag>
                </span>
              </span>
            );
          },
          size: COLUMN_WIDTH,
          accessorFn: baselineAccessorFn,
          cell: (row: any) => <span>{getDifferenceChartDisplayedValue(row.getValue())}</span>,
        };
      }
      return {
        id: runData.uuid,
        header: () => {
          return (
            <span css={{ display: 'flex', flexDirection: 'row', justifyContent: 'space-between' }}>
              <span css={{ display: 'inline-flex', gap: theme.spacing.sm, alignItems: 'center' }}>
                <RunColorPill color={runData.color} />
                {runData.displayName}
              </span>
            </span>
          );
        },
        size: COLUMN_WIDTH,
        accessorFn: accessorFn,
        cell: (row: any) => (
          <span css={{ display: 'inline-flex', gap: theme.spacing.md, verticalAlign: 'middle' }}>
            <Typography.Text>{getDifferenceChartDisplayedValue(row.getValue().text)}</Typography.Text>
            {baselineColumn &&
              cardConfig.showChangeFromBaseline &&
              row.getValue().difference &&
              (row.getValue().difference[0] === '-' ? (
                <div css={{ display: 'inline-flex', gap: theme.spacing.xs }}>
                  <Typography.Paragraph color="error">{row.getValue().difference}</Typography.Paragraph>
                  <ArrowDownIcon color="danger" />
                </div>
              ) : (
                <div css={{ display: 'inline-flex', gap: theme.spacing.xs }}>
                  <Typography.Paragraph color="success">{row.getValue().difference}</Typography.Paragraph>
                  <ArrowUpIcon color="success" />
                </div>
              ))}
          </span>
        ),
      };
    };

    return [
      {
        id: HEADING_COLUMN_ID,
        header: formatMessage({
          defaultMessage: 'Compare by',
          description: 'Runs charts > components > charts > DifferenceViewPlot > Compare by column heading',
        }),
        accessorKey: HEADING_COLUMN_ID,
        size: COLUMN_WIDTH,
      },
      ...(baselineColumn ? [convertRunToColumnInfo(baselineColumn, true)] : []),
      ...nonBaselineColumns.map((runData) => convertRunToColumnInfo(runData, false)),
    ];
  }, [theme.spacing, formatMessage, baselineColumn, nonBaselineColumns, cardConfig.showChangeFromBaseline]);

  const table = useReactTable({
    data: getData,
    columns: getColumns,
    enableColumnResizing: true,
    columnResizeMode: 'onChange',
    getCoreRowModel: getCoreRowModel(),
  });

  const updateBaselineColumnUuid = useCallback(
    (baselineColumnUuid: string) => {
      if (setCardConfig) {
        setCardConfig((current) => ({
          ...(current as RunsChartsDifferenceCardConfig),
          baselineColumnUuid,
        }));
      }
    },
    [setCardConfig],
  );

  if (previewData.length === 0) {
    return null;
  }

  if (table.getRowModel().rows.length === 0) {
    return <NoDifferenceDataPlaceholder />;
  }

  return (
    <div
      css={{
        display: 'flex',
        overflow: 'auto hidden',
        cursor: 'pointer',
        height: undefined,
        width: '100%',
      }}
    >
      <div css={{ width: '100%' }}>
        <Table style={{ width: table.getTotalSize() }} scrollable>
          {table.getHeaderGroups().map((headerGroup) => (
            <TableRow key={headerGroup.id} isHeader>
              {headerGroup.headers.map((header, index) => {
                return (
                  <TableHeader
                    key={header.id}
                    style={{
                      maxWidth: header.column.getSize(),
                    }}
                    resizable={header.column.getCanResize()}
                    resizeHandler={header.getResizeHandler()}
                  >
                    <div
                      css={{
                        display: 'flex',
                        flexDirection: 'row',
                        gap: theme.spacing.xs,
                        alignItems: 'center',
                      }}
                    >
                      <div css={{ flexShrink: 1, flexGrow: 1, overflow: 'hidden' }}>
                        {flexRender(header.column.columnDef.header, header.getContext())}
                      </div>
                      {index !== 0 && setCardConfig && (
                        <div>
                          <DropdownMenu.Root>
                            <DropdownMenu.Trigger asChild>
                              <Button componentId="set_as_baseline_button" icon={<OverflowIcon />} />
                            </DropdownMenu.Trigger>
                            <DropdownMenu.Content>
                              <DropdownMenu.Item
                                componentId="codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_charts_differenceviewplot.tsx_271"
                                onClick={() => updateBaselineColumnUuid(header.id)}
                              >
                                <FormattedMessage
                                  defaultMessage="Set as baseline"
                                  description="Runs charts > components > charts > DifferenceViewPlot > Set as baseline dropdown option"
                                />
                              </DropdownMenu.Item>
                            </DropdownMenu.Content>
                          </DropdownMenu.Root>
                        </div>
                      )}
                    </div>
                  </TableHeader>
                );
              })}
            </TableRow>
          ))}
          {table.getRowModel().rows.map((row) => (
            <TableRow key={row.id}>
              {row.getAllCells().map((cell) => (
                <TableCell key={cell.id} style={{ maxWidth: cell.column.getSize() }} multiline>
                  {flexRender(cell.column.columnDef.cell, cell.getContext())}
                </TableCell>
              ))}
            </TableRow>
          ))}
        </Table>
      </div>
    </div>
  );
};
