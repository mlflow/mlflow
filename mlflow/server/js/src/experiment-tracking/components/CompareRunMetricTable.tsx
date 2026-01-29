import React, { forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState } from 'react';
import { RunInfoEntity } from '@mlflow/mlflow/src/experiment-tracking/types';
import { AutoSizer, Grid, ScrollParams } from 'react-virtualized';
import { Link } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import Routes from '@mlflow/mlflow/src/experiment-tracking/routes';
import Utils from '@mlflow/mlflow/src/common/utils/Utils';
import { LegacyTooltip } from '@databricks/design-system';

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
  const tableHeight = Math.min(rowHeight * props.metricRows.length, tableMaxHeight);
  const columnCount = Math.max(...props.metricRows.map((e) => e.values.length));
  const headerRef = useRef<Grid | null>(null);
  const bodyRef = useRef<Grid | null>(null);
  const [hoveredSet, setHoveredSet] = useState<{ row: number; column: number }[]>([]);

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
            <div
              key={key}
              style={style}
              onMouseEnter={() => addHoveredEntry(rowIndex, 0)}
              onMouseLeave={() => removeHoveredEntry(rowIndex, 0)}
              css={{
                overflow: 'hidden',
                overflowWrap: 'break-word',
                padding: '12px 8px',
                borderBottom: '1px solid #e8e8e8',
                color: props.metricRows[rowIndex].highlightDiff ? '#555' : 'rgba(0, 0, 0, 0.85)',
                fontWeight: 500,
                backgroundColor: props.metricRows[rowIndex].highlightDiff
                  ? 'rgba(249, 237, 190, 1)'
                  : 'rgb(250, 250, 250)',
                textAlign: 'left',
              }}
            >
              <Link
                to={Routes.getMetricPageRoute(
                  props.runInfos
                    .map((info) => info.runUuid)
                    .filter((_uuid, idx) => props.metricRows[rowIndex].values[idx] !== undefined),
                  props.metricRows[rowIndex].key,
                  props.experimentIds,
                )}
                title="Plot chart"
              >
                {props.metricRows[rowIndex].key}
                <i className="fa fa-chart-line" css={{ paddingLeft: '6px' }} />
              </Link>
            </div>
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
              cellRenderer={({ columnIndex, rowIndex, key, style }) => {
                const formatted = Utils.formatMetric(props.metricRows[rowIndex].values[columnIndex]);
                return (
                  <div
                    key={key}
                    style={style}
                    onMouseEnter={() => addHoveredEntry(rowIndex, columnIndex + 1)}
                    onMouseLeave={() => removeHoveredEntry(rowIndex, columnIndex + 1)}
                    css={{
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      padding: '12px 8px',
                      color: props.metricRows[rowIndex].highlightDiff ? '#555' : undefined,
                      borderBottom: '1px solid #e8e8e8',
                      backgroundColor: props.metricRows[rowIndex].highlightDiff
                        ? 'rgba(249, 237, 190, 1)'
                        : isHovered(rowIndex, undefined)
                        ? 'rgb(250, 250, 250)'
                        : undefined,
                    }}
                  >
                    <LegacyTooltip
                      title={formatted}
                      // @ts-expect-error TS(2322): Type '{ children: any; title: any; color: string; ... Remove this comment to see the full error message
                      color="gray"
                      placement="topLeft"
                      overlayStyle={{ maxWidth: '400px' }}
                      // mouseEnterDelay prop is not available in DuBois design system (yet)
                      dangerouslySetAntdProps={{ mouseEnterDelay: 1 }}
                    >
                      {formatted}
                    </LegacyTooltip>
                  </div>
                );
              }}
            />
          )}
        </AutoSizer>
      </div>
    </div>
  );
});
