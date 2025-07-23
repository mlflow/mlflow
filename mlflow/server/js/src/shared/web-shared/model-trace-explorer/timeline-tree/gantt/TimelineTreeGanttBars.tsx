import { useMemo, useRef } from 'react';

import { Typography, useDesignSystemTheme } from '@databricks/design-system';
import { useResizeObserver } from '@databricks/web-shared/hooks';

import { TimelineTreeGanttNode } from './TimelineTreeGanttNode';
import type { ModelTraceSpanNode } from '../../ModelTrace.types';
import { spanTimeFormatter, TimelineTreeZIndex } from '../TimelineTree.utils';

// the amount of space required to accomodate the collapse buttons
const TIMELINE_BAR_LEFT_OFFSET = 32;

// this function generates an array of "nice" x-ticks (e.g. nearest 0.1, 0.2, 0.5 to the value)
function getNiceXTicks(left: number, right: number, graphWidth: number, minPixelsBetweenTicks = 60): number[] {
  const range = right - left;
  if (range <= 0 || graphWidth <= 0) return [];

  const maxTickCount = Math.floor(graphWidth / minPixelsBetweenTicks);
  if (maxTickCount < 1) return [];

  // Step 1: raw interval
  const rawInterval = range / maxTickCount;

  // Step 2: round to a "nice" interval
  const magnitude = Math.pow(10, Math.floor(Math.log10(rawInterval)));
  const residual = rawInterval / magnitude;

  let niceFraction;
  if (residual <= 1) niceFraction = 1;
  else if (residual <= 2) niceFraction = 2;
  else if (residual <= 5) niceFraction = 5;
  else niceFraction = 10;

  const niceInterval = niceFraction * magnitude;

  // Step 3: extend right bound so we always overshoot it
  // this guarantees that there will be enough space to
  // render the span labels.
  const extendedRight = right + 2 * niceInterval;

  // Step 4: Generate tick positions
  const firstTick = Math.ceil(left / niceInterval) * niceInterval;
  const ticks: number[] = [];

  for (let tick = firstTick; tick <= extendedRight; tick += niceInterval) {
    ticks.push(Number(tick.toFixed(10))); // Avoid float errors
  }

  return ticks;
}

// converts timestamp numbers to real pixel values
function scaleX(value: number, left: number, right: number, width: number) {
  return ((value - left) / (right - left)) * width;
}

export const TimelineTreeGanttBars = ({
  nodes,
  selectedKey,
  onSelect,
  traceStartTime,
  traceEndTime,
  expandedKeys,
  setExpandedKeys,
}: {
  nodes: ModelTraceSpanNode[];
  selectedKey: string | number;
  onSelect: ((node: ModelTraceSpanNode) => void) | undefined;
  traceStartTime: number;
  traceEndTime: number;
  expandedKeys: Set<string | number>;
  setExpandedKeys: (keys: Set<string | number>) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const treeContainerRef = useRef<HTMLDivElement>(null);
  const treeElementWidth = useResizeObserver({ ref: treeContainerRef })?.width ?? 0;
  const initialXTicks = useMemo(
    () => getNiceXTicks(traceStartTime, traceEndTime, treeElementWidth),
    [traceEndTime, traceStartTime, treeElementWidth],
  );
  const left = Math.min(...initialXTicks);
  // for the right limit of the graph, we take the average of the last
  // two ticks so that the graph does not end directly on a line. if
  // the graph ends on the line, the ticklabel at the top might render
  // slightly off screen, which looks bad
  const right = (initialXTicks[initialXTicks.length - 1] + initialXTicks[initialXTicks.length - 2]) / 2;
  // pop the last tick since we will not render it (it's beyond the right limit)
  const xTicks = initialXTicks.slice(0, -1);

  const scaleDurationToTreeWidth = (value: number) => scaleX(value, left, right, treeElementWidth);

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        position: 'relative',
        boxSizing: 'border-box',
      }}
    >
      {/* gantt bar header with the tick labels */}
      <div
        ref={treeContainerRef}
        css={{
          display: 'flex',
          width: '100%',
          flexDirection: 'row',
          height: theme.typography.lineHeightBase,
          paddingLeft: TIMELINE_BAR_LEFT_OFFSET,
          paddingRight: theme.spacing.lg,
          boxSizing: 'border-box',
          position: 'sticky',
          top: 0,
          backgroundColor: theme.colors.backgroundPrimary,
          zIndex: TimelineTreeZIndex.HIGH,
        }}
      >
        <div
          data-testid="time-marker-area"
          css={{
            position: 'relative',
          }}
        >
          {xTicks.map((n) => (
            <Typography.Text
              css={{
                position: 'absolute',
                transform: `translateX(-50%)`,
                left: scaleDurationToTreeWidth(n),
                whiteSpace: 'nowrap',
              }}
              key={n}
            >
              {spanTimeFormatter(n)}
            </Typography.Text>
          ))}
        </div>
      </div>
      {/* vertical gantt markers */}
      <div
        css={{
          flex: 1,
          pointerEvents: 'none',
          zIndex: TimelineTreeZIndex.LOW,
        }}
      >
        <div
          css={{
            position: 'absolute',
            height: '100%',
            width: '100%',
          }}
        >
          {xTicks.map((n) => (
            <div
              key={n}
              css={{
                position: 'absolute',
                left: scaleDurationToTreeWidth(n) + TIMELINE_BAR_LEFT_OFFSET,
                borderRight: `1px solid ${theme.colors.border}`,
                height: '100%',
              }}
            />
          ))}
        </div>
      </div>
      {/* colored horizontal gantt bars */}
      {nodes.map((node) => {
        const leftOffset = scaleDurationToTreeWidth(node.start);
        const width = scaleDurationToTreeWidth(node.end) - leftOffset;
        return (
          <TimelineTreeGanttNode
            key={node.key}
            selectedKey={selectedKey}
            onSelect={onSelect}
            node={node}
            leftOffset={leftOffset}
            width={width}
            expandedKeys={expandedKeys}
            setExpandedKeys={setExpandedKeys}
          />
        );
      })}
    </div>
  );
};
