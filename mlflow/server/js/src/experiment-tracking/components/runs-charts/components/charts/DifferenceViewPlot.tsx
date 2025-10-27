import {
  Button,
  ChevronDownIcon,
  ChevronRightIcon,
  Table,
  TableCell,
  TableHeader,
  TableRow,
  useDesignSystemTheme,
} from '@databricks/design-system';
import type { CellContext, ColumnDef, ColumnDefTemplate, Row } from '@tanstack/react-table';
import { flexRender, getCoreRowModel, getExpandedRowModel, useReactTable } from '@tanstack/react-table';
import { useVirtualizer } from '@tanstack/react-virtual';
import { useCallback, useMemo, useRef, useState } from 'react';
import { useIntl } from 'react-intl';
import type { RunsGroupByConfig } from '../../../experiment-page/utils/experimentPage.group-row-utils';
import type { RunsChartsCardConfig, RunsChartsDifferenceCardConfig } from '../../runs-charts.types';
import { DifferenceCardConfigCompareGroup } from '../../runs-charts.types';
import {
  DIFFERENCE_PLOT_EXPAND_COLUMN_ID,
  DIFFERENCE_PLOT_HEADING_COLUMN_ID,
  getDifferencePlotJSONRows,
  getDifferenceViewDataGroups,
} from '../../utils/differenceView';
import type { RunsChartsRunData } from '../RunsCharts.common';
import { DifferencePlotDataCell } from './difference-view-plot/DifferencePlotDataCell';
import { DifferencePlotRunHeaderCell } from './difference-view-plot/DifferencePlotRunHeaderCell';

export type DifferencePlotDataColumnDef = ColumnDef<DifferencePlotDataRow> & {
  meta?: {
    traceData?: RunsChartsRunData;
    updateBaselineColumnUuid: (uuid: string) => void;
    isBaseline: boolean;
    showChangeFromBaseline: boolean;
    baselineColumnUuid?: string;
  };
};

export type DifferencePlotDataRow =
  | Record<string, any>
  | {
      children: DifferencePlotDataRow[];
      key: string;
      [DIFFERENCE_PLOT_HEADING_COLUMN_ID]: string;
    };

const ExpandCell: ColumnDefTemplate<
  CellContext<DifferencePlotDataRow, unknown> & { toggleExpand?: (row: Row<DifferencePlotDataRow>) => void }
> = ({ row, toggleExpand }) => {
  if (row.getCanExpand() && toggleExpand) {
    return (
      <Button
        componentId="mlflow.charts.difference_plot.expand_button"
        size="small"
        type="link"
        onClick={() => toggleExpand(row)}
        icon={row.getIsExpanded() ? <ChevronDownIcon /> : <ChevronRightIcon />}
      />
    );
  }
  return null;
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
    () => getDifferenceViewDataGroups(previewData, cardConfig, DIFFERENCE_PLOT_HEADING_COLUMN_ID, groupBy),
    [previewData, cardConfig, groupBy],
  );

  const { baselineColumn, nonBaselineColumns } = useMemo(() => {
    const dataTracesReverse = previewData.slice().reverse();
    // baseline column (can be undefined if no baseline selected)
    let baselineColumn = dataTracesReverse.find((runData) => runData.uuid === cardConfig.baselineColumnUuid);
    if (baselineColumn === undefined && dataTracesReverse.length > 0) {
      // Set the first column as baseline column
      baselineColumn = dataTracesReverse[0];
    }
    // non-baseline columns
    const nonBaselineColumns = dataTracesReverse.filter(
      (runData) => baselineColumn === undefined || runData.uuid !== baselineColumn.uuid,
    );
    return { baselineColumn, nonBaselineColumns };
  }, [previewData, cardConfig.baselineColumnUuid]);

  const updateBaselineColumnUuid = useCallback(
    (baselineColumnUuid: string) => {
      setCardConfig?.((current) => ({
        ...(current as RunsChartsDifferenceCardConfig),
        baselineColumnUuid,
      }));
    },
    [setCardConfig],
  );

  const dataRows = useMemo<DifferencePlotDataRow[]>(
    () =>
      cardConfig.compareGroups.reduce((acc: DifferencePlotDataRow[], group: DifferenceCardConfigCompareGroup) => {
        switch (group) {
          case DifferenceCardConfigCompareGroup.MODEL_METRICS:
            acc.push({
              [DIFFERENCE_PLOT_HEADING_COLUMN_ID]: formatMessage({
                defaultMessage: `Model Metrics`,
                description:
                  'Experiment tracking > runs charts > cards > RunsChartsDifferenceChartCard > model metrics heading',
              }),
              children: [...modelMetrics],
              key: 'modelMetrics',
            });
            break;
          case DifferenceCardConfigCompareGroup.SYSTEM_METRICS:
            acc.push({
              [DIFFERENCE_PLOT_HEADING_COLUMN_ID]: formatMessage({
                defaultMessage: `System Metrics`,
                description:
                  'Experiment tracking > runs charts > cards > RunsChartsDifferenceChartCard > system metrics heading',
              }),
              children: [...systemMetrics],
              key: 'systemMetrics',
            });
            break;
          case DifferenceCardConfigCompareGroup.PARAMETERS:
            acc.push({
              [DIFFERENCE_PLOT_HEADING_COLUMN_ID]: formatMessage({
                defaultMessage: `Parameters`,
                description:
                  'Experiment tracking > runs charts > cards > RunsChartsDifferenceChartCard > parameters heading',
              }),
              children: getDifferencePlotJSONRows(parameters),
              key: 'parameters',
            });
            break;
          case DifferenceCardConfigCompareGroup.ATTRIBUTES:
            acc.push({
              [DIFFERENCE_PLOT_HEADING_COLUMN_ID]: formatMessage({
                defaultMessage: `Attributes`,
                description:
                  'Experiment tracking > runs charts > cards > RunsChartsDifferenceChartCard > attributes heading',
              }),
              children: [...attributes],
              key: 'attributes',
            });
            break;
          case DifferenceCardConfigCompareGroup.TAGS:
            acc.push({
              [DIFFERENCE_PLOT_HEADING_COLUMN_ID]: formatMessage({
                defaultMessage: `Tags`,
                description: 'Experiment tracking > runs charts > cards > RunsChartsDifferenceChartCard > tags heading',
              }),
              children: [...tags],
              key: 'tags',
            });
            break;
        }
        return acc;
      }, []),
    [modelMetrics, systemMetrics, parameters, tags, attributes, cardConfig.compareGroups, formatMessage],
  );

  const columns = useMemo(() => {
    const columns: DifferencePlotDataColumnDef[] = [
      {
        id: DIFFERENCE_PLOT_EXPAND_COLUMN_ID,
        cell: ExpandCell,
        size: 32,
        enableResizing: false,
      },
      {
        accessorKey: DIFFERENCE_PLOT_HEADING_COLUMN_ID,
        size: 150,
        header: formatMessage({
          defaultMessage: 'Compare by',
          description: 'Runs charts > components > charts > DifferenceViewPlot > Compare by column heading',
        }),
        id: DIFFERENCE_PLOT_HEADING_COLUMN_ID,
        enableResizing: true,
      },
      ...[baselineColumn, ...nonBaselineColumns].map((traceData, index) => ({
        accessorKey: traceData?.uuid,
        size: 200,
        enableResizing: true,
        meta: {
          traceData,
          updateBaselineColumnUuid,
          showChangeFromBaseline: cardConfig.showChangeFromBaseline,
          isBaseline: traceData === baselineColumn,
          baselineColumnUuid: baselineColumn?.uuid,
        },
        id: traceData?.uuid ?? index.toString(),
        header: DifferencePlotRunHeaderCell as ColumnDefTemplate<DifferencePlotDataRow>,
        cell: DifferencePlotDataCell as ColumnDefTemplate<DifferencePlotDataRow>,
      })),
    ];
    return columns;
  }, [formatMessage, baselineColumn, nonBaselineColumns, updateBaselineColumnUuid, cardConfig.showChangeFromBaseline]);

  // Start with all row groups expanded
  const [expanded, setExpanded] = useState({
    modelMetrics: true,
    systemMetrics: true,
    parameters: true,
    attributes: true,
    tags: true,
  });

  const table = useReactTable({
    columns,
    data: dataRows,
    getCoreRowModel: getCoreRowModel(),
    getExpandedRowModel: getExpandedRowModel(),
    columnResizeMode: 'onChange',
    enableExpanding: true,
    getSubRows: (row) => row.children,
    getRowId: (row) => row.key,
    getRowCanExpand: (row) => Boolean(row.subRows.length),
    state: {
      expanded,
      columnPinning: {
        left: [DIFFERENCE_PLOT_EXPAND_COLUMN_ID, DIFFERENCE_PLOT_HEADING_COLUMN_ID],
      },
    },
  });

  const tableContainerRef = useRef<HTMLDivElement>(null);

  const toggleExpand = useCallback((row: Row<DifferencePlotDataRow>) => {
    const key = row.original.key;
    setExpanded((prev) => ({
      ...prev,
      [key]: !row.getIsExpanded(),
    }));
  }, []);

  const { getVirtualItems, getTotalSize } = useVirtualizer({
    count: table.getExpandedRowModel().rows.length,
    getScrollElement: () => tableContainerRef.current,
    estimateSize: () => 33, // Default row height
    paddingStart: 37, // Default header height,
  });

  const expandedRows = table.getExpandedRowModel().rows;

  return (
    <div css={{ flex: 1, overflowX: 'scroll', height: '100%' }} ref={tableContainerRef}>
      <Table css={{ width: table.getTotalSize(), position: 'relative' }}>
        <TableRow isHeader css={{ position: 'sticky', top: 0, zIndex: 101 }}>
          {table.getLeafHeaders().map((header, index) => {
            const isPinned = header.column.getIsPinned();

            return (
              <TableHeader
                header={header}
                column={header.column}
                setColumnSizing={table.setColumnSizing}
                componentId="mlflow.charts.difference_plot.header"
                key={header.id}
                multiline={false}
                style={{
                  left: isPinned === 'left' ? `${header.column.getStart('left')}px` : undefined,
                  position: isPinned ? 'sticky' : 'relative',
                  width: header.column.getSize(),
                  flexBasis: header.column.getSize(),
                  zIndex: isPinned ? 100 : 0,
                }}
                wrapContent={false}
              >
                {flexRender(header.column.columnDef.header, header.getContext())}
              </TableHeader>
            );
          })}
        </TableRow>
        <div css={{ height: getTotalSize() }}>
          {getVirtualItems().map(({ index, start, size }) => {
            const row = expandedRows[index];

            return (
              <TableRow
                key={row.id + index}
                css={{
                  width: 'auto',
                  position: 'absolute',
                  top: 0,
                }}
                style={{
                  transform: `translateY(${start}px)`,
                  height: size,
                }}
              >
                {row.getVisibleCells().map((cell, index) => {
                  const isPinned = cell.column.getIsPinned();

                  const isNameColumn = cell.column.columnDef.id === DIFFERENCE_PLOT_HEADING_COLUMN_ID;
                  return (
                    <TableCell
                      key={cell.id}
                      style={{
                        left: isPinned === 'left' ? `${cell.column.getStart('left')}px` : undefined,
                        position: isPinned ? 'sticky' : 'relative',
                        width: cell.column.getSize(),
                        zIndex: isPinned ? 100 : 0,
                        flexBasis: cell.column.getSize(),
                      }}
                      css={[
                        {
                          backgroundColor: isPinned ? theme.colors.backgroundPrimary : undefined,
                        },
                        isNameColumn && { borderRight: `1px solid ${theme.colors.border}` },
                        isNameColumn && { paddingLeft: row.depth * theme.spacing.lg },
                      ]}
                      wrapContent={false}
                    >
                      {flexRender(cell.column.columnDef.cell, { ...cell.getContext(), toggleExpand })}
                    </TableCell>
                  );
                })}
              </TableRow>
            );
          })}
        </div>
      </Table>
    </div>
  );
};
