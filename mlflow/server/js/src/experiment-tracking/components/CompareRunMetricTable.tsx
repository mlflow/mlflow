import React, { forwardRef, useCallback, useEffect, useImperativeHandle, useMemo, useRef, useState } from 'react';
import { RunInfoEntity } from '@mlflow/mlflow/src/experiment-tracking/types';
import { AutoSizer, Grid, ScrollParams } from 'react-virtualized';
import { Link } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import Routes from '@mlflow/mlflow/src/experiment-tracking/routes';
import Utils from '@mlflow/mlflow/src/common/utils/Utils';
import { LegacyTooltip, useDesignSystemTheme } from '@databricks/design-system';

export type CompareRunMetricTableRef = {
  setScrollLeft: (scrollLeft: number) => void;
};

export type CompareRunMetricTableProps = {
  colWidth: number;
  experimentIds: string[];
  runInfos: RunInfoEntity[];
  metricRows: {
    key: string;
    highlightDiff: boolean;
    values: number[];
  }[];
  onScroll: (params: ScrollParams) => unknown;
};

export const CompareRunMetricTable = forwardRef<CompareRunMetricTableRef, CompareRunMetricTableProps>((props, ref) => {
  const rowHeight = 45;
  const tableMaxHeight = 300;
  const headerRef = useRef<Grid | null>(null);
  const bodyRef = useRef<Grid | null>(null);
  const [hoveredSet, setHoveredSet] = useState<{ row: number; column: number }[]>([]);

  const { tableHeight, columnCount } = useMemo(
    () => ({
      tableHeight: Math.min(rowHeight * props.metricRows.length, tableMaxHeight),
      columnCount: Math.max(...props.metricRows.map((e) => e.values.length)),
    }),
    [rowHeight, tableMaxHeight, props.metricRows],
  );

  const isHovered = useCallback(
    (row?: number, column?: number) => {
      return hoveredSet.some(
        (e) => (row === undefined || e.row === row) && (column === undefined || e.column === column),
      );
    },
    [hoveredSet],
  );

  const addHoveredEntry = useCallback(
    (row: number, column: number) => {
      setHoveredSet((previous) => {
        if (!isHovered(row, column)) {
          return [...previous, { row, column }];
        } else {
          return previous;
        }
      });
    },
    [setHoveredSet, isHovered],
  );

  const removeHoveredEntry = useCallback(
    (row: number, column: number) => {
      setHoveredSet((previous) => {
        return previous.filter((e) => e.row !== row || e.column !== column);
      });
    },
    [setHoveredSet],
  );

  const synchronize = useCallback(() => {
    if (headerRef.current && bodyRef.current) {
      headerRef.current.scrollToPosition({
        scrollTop: bodyRef.current.state.scrollTop,
        scrollLeft: 0,
      });
    }
  }, []);

  const onScroll = useCallback(
    (params: ScrollParams) => {
      synchronize();
      props.onScroll(params);
    },
    [props.onScroll, synchronize],
  );

  useImperativeHandle(
    ref,
    () => ({
      setScrollLeft: (scrollLeft: number) => {
        if (bodyRef.current && bodyRef.current.state.scrollLeft !== scrollLeft) {
          bodyRef.current?.scrollToPosition({
            scrollLeft,
            scrollTop: bodyRef.current.state.scrollTop,
          });
        }
      },
    }),
    [],
  );

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'row',
      }}
    >
      <div
        css={{
          width: `${props.colWidth}px`,
          display: 'flex',
        }}
      >
        <Grid
          ref={headerRef}
          style={{ overflow: 'hidden' }}
          width={props.colWidth}
          height={tableHeight}
          rowCount={props.metricRows.length}
          rowHeight={rowHeight}
          columnCount={1}
          columnWidth={props.colWidth}
          cellRenderer={({ rowIndex, key, style }) => (
            <CompareRunMetricTableHeaderCell
              key={key}
              style={style}
              onMouseEnter={() => addHoveredEntry(rowIndex, 0)}
              onMouseLeave={() => removeHoveredEntry(rowIndex, 0)}
              highlightDiff={props.metricRows[rowIndex].highlightDiff}
              metricRowKey={props.metricRows[rowIndex].key}
              metricPageRoute={Routes.getMetricPageRoute(
                props.runInfos
                  .map((info) => info.runUuid)
                  .filter((_uuid, idx) => props.metricRows[rowIndex].values[idx] !== undefined),
                props.metricRows[rowIndex].key,
                props.experimentIds,
              )}
            />
          )}
        />
      </div>
      <div
        css={{
          flex: '1',
          display: 'flex',
        }}
      >
        <AutoSizer>
          {({ width }) => (
            <Grid
              ref={bodyRef}
              onScroll={onScroll}
              width={width}
              height={tableHeight}
              rowCount={props.metricRows.length}
              rowHeight={rowHeight}
              columnCount={columnCount}
              columnWidth={props.colWidth}
              cellRenderer={({ columnIndex, rowIndex, key, style }) => (
                <CompareRunMetricTableBodyCell
                  key={key}
                  style={style}
                  onMouseEnter={() => addHoveredEntry(rowIndex, columnIndex + 1)}
                  onMouseLeave={() => removeHoveredEntry(rowIndex, columnIndex + 1)}
                  formattedMetric={Utils.formatMetric(props.metricRows[rowIndex].values[columnIndex])}
                  highlightDiff={props.metricRows[rowIndex].highlightDiff}
                  rowHovered={isHovered(rowIndex, undefined)}
                />
              )}
            />
          )}
        </AutoSizer>
      </div>
    </div>
  );
});

interface CompareRunMetricTableHeaderCellProps {
  key: string;
  style: React.CSSProperties;
  highlightDiff: boolean;
  metricRowKey: string;
  metricPageRoute: string;
  onMouseEnter: () => void;
  onMouseLeave: () => void;
}

function CompareRunMetricTableHeaderCell({
  key,
  style,
  highlightDiff,
  metricRowKey,
  metricPageRoute,
  onMouseEnter,
  onMouseLeave,
}: CompareRunMetricTableHeaderCellProps) {
  const { theme } = useDesignSystemTheme();
  return (
    <div
      key={key}
      style={style}
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
      css={{
        overflow: 'hidden',
        overflowWrap: 'break-word',
        padding: '12px 8px',
        borderBottom: `1px solid ${theme.colors.border}`,
        color: highlightDiff ? theme.colors.textSecondary : theme.colors.textPrimary,
        fontWeight: 500,
        backgroundColor: highlightDiff ? theme.colors.backgroundWarning : theme.colors.backgroundSecondary,
        textAlign: 'left',
      }}
    >
      <Link to={metricPageRoute} title="Plot chart">
        {metricRowKey}
        <i className="fa fa-chart-line" css={{ paddingLeft: '6px' }} />
      </Link>
    </div>
  );
}

interface CompareRunMetricTableBodyCellProps {
  key: string;
  style: React.CSSProperties;
  formattedMetric: string;
  highlightDiff: boolean;
  rowHovered: boolean;
  onMouseEnter: () => void;
  onMouseLeave: () => void;
}

function CompareRunMetricTableBodyCell({
  key,
  style,
  formattedMetric,
  highlightDiff,
  rowHovered,
  onMouseEnter,
  onMouseLeave,
}: CompareRunMetricTableBodyCellProps) {
  const { theme } = useDesignSystemTheme();
  return (
    <div
      key={key}
      style={style}
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
      css={{
        overflow: 'hidden',
        textOverflow: 'ellipsis',
        padding: '12px 8px',
        color: highlightDiff ? theme.colors.textSecondary : undefined,
        borderBottom: `1px solid ${theme.colors.border}`,
        backgroundColor: highlightDiff
          ? theme.colors.backgroundWarning
          : rowHovered
            ? theme.colors.backgroundSecondary
            : undefined,
      }}
    >
      <LegacyTooltip
        title={formattedMetric}
        // @ts-expect-error TS(2322): Type '{ children: any; title: any; color: string; ... Remove this comment to see the full error message
        color="gray"
        placement="topLeft"
        overlayStyle={{ maxWidth: '400px' }}
        // mouseEnterDelay prop is not available in DuBois design system (yet)
        dangerouslySetAntdProps={{ mouseEnterDelay: 1 }}
      >
        {formattedMetric}
      </LegacyTooltip>
    </div>
  );
}
