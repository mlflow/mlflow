import { Typography, useDesignSystemTheme } from '@databricks/design-system';

import React from 'react';
import type { Dash } from 'plotly.js';

const STROKE_WIDTH = 3;

/**
 * Replicating plotly.js's dasharrays for each dash type, with smaller spaces
 * https://github.com/plotly/plotly.js/blob/master/src/components/drawing/index.js#L162
 */
const getDashArray = (dashType: Dash) => {
  switch (dashType) {
    case 'dot':
      return `${STROKE_WIDTH}`;
    case 'dash':
      return `${2 * STROKE_WIDTH}, ${STROKE_WIDTH}`;
    case 'longdash':
      return `${3 * STROKE_WIDTH}, ${STROKE_WIDTH}`;
    case 'dashdot':
      return `${2 * STROKE_WIDTH}, ${STROKE_WIDTH}, ${STROKE_WIDTH}, ${STROKE_WIDTH}`;
    case 'longdashdot':
      return `${3 * STROKE_WIDTH}, ${STROKE_WIDTH}, ${STROKE_WIDTH}, ${STROKE_WIDTH}`;
    default:
      return '';
  }
};

export type LegendLabelData = {
  label: string;
  color: string;
  dashStyle?: Dash;
  uuid?: string;
  metricKey?: string;
};

const TraceLabel: React.FC<React.PropsWithChildren<LegendLabelData>> = ({ label, color, dashStyle }) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        display: 'flex',
        alignItems: 'center',
        textOverflow: 'ellipsis',
        flexShrink: 0,
        marginRight: theme.spacing.md,
        maxWidth: '100%',
      }}
    >
      <TraceLabelColorIndicator color={color} dashStyle={dashStyle} />
      <Typography.Text
        color="secondary"
        size="sm"
        css={{ whiteSpace: 'nowrap', textOverflow: 'ellipsis', overflow: 'hidden' }}
      >
        {label}
      </Typography.Text>
    </div>
  );
};

export const TraceLabelColorIndicator: React.FC<
  React.PropsWithChildren<Pick<LegendLabelData, 'color' | 'dashStyle'>>
> = ({ color, dashStyle }) => {
  const { theme } = useDesignSystemTheme();
  const strokeDasharray = dashStyle ? getDashArray(dashStyle) : undefined;
  const pathYOffset = theme.typography.fontSizeSm / 2;

  return (
    <svg
      css={{
        height: theme.typography.fontSizeSm,
        width: STROKE_WIDTH * 8,
        marginRight: theme.spacing.xs,
        flexShrink: 0,
      }}
    >
      <path
        d={`M0,${pathYOffset}h${STROKE_WIDTH * 8}`}
        style={{
          strokeWidth: STROKE_WIDTH,
          stroke: color,
          strokeDasharray,
        }}
      />
    </svg>
  );
};
type RunsMetricsLegendProps = {
  labelData: LegendLabelData[];
  height: number;
  fullScreen?: boolean;
};

const RunsMetricsLegend: React.FC<React.PropsWithChildren<RunsMetricsLegendProps>> = ({
  labelData,
  height,
  fullScreen,
}) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        display: 'flex',
        flexWrap: 'wrap',
        height,
        alignContent: fullScreen ? 'flex-start' : 'normal',
        gap: fullScreen ? theme.spacing.sm : 0,
        overflowY: 'auto',
        overflowX: 'hidden',
        marginTop: fullScreen ? theme.spacing.lg : theme.spacing.sm,
      }}
    >
      {labelData.map((labelDatum) => (
        <TraceLabel
          key={
            labelDatum.uuid && labelDatum.metricKey ? `${labelDatum.uuid}-${labelDatum.metricKey}` : labelDatum.label
          }
          {...labelDatum}
        />
      ))}
    </div>
  );
};

export default RunsMetricsLegend;
