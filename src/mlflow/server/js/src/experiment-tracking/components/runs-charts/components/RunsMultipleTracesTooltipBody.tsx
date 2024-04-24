import { isUndefined } from 'lodash';

import { RunsCompareMultipleTracesTooltipData } from './RunsMetricsLinePlot';
import React from 'react';
import { TraceLabelColorIndicator } from './RunsMetricsLegend';
import { normalizeMetricChartTooltipValue } from '../../../utils/MetricsUtils';
import { FormattedDate, FormattedTime, defineMessages, useIntl } from 'react-intl';
import { getChartAxisLabelDescriptor, RunsChartsLineChartXAxisType } from './RunsCharts.common';
import { useDesignSystemTheme } from '@databricks/design-system';

// Sadly when hovering outside data point, we can't get the date-time value from Plotly chart
// so we have to format it ourselves in a way that resembles Plotly's logic
const PlotlyLikeFormattedTime = ({ value }: { value: string | number }) => (
  <>
    <FormattedDate value={value} year="numeric" />-
    <FormattedDate value={value} month="2-digit" />-
    <FormattedDate value={value} day="2-digit" /> <FormattedTime value={value} hour="numeric" hourCycle="h24" />:
    <FormattedTime value={value} minute="2-digit" />:
    {/* @ts-expect-error "fractionalSecondDigits" is supported but missing from TS types */}
    <FormattedTime value={value} second="2-digit" fractionalSecondDigits={3} />
  </>
);

/**
 * Variant of the tooltip body for the line chart that displays multiple traces at once.
 * Used in the compare runs page and run details metrics page.
 */
export const RunsMultipleTracesTooltipBody = ({ hoverData }: { hoverData: RunsCompareMultipleTracesTooltipData }) => {
  const { tooltipLegendItems, hoveredDataPoint: singleTraceHoverData, xValue, xAxisKeyLabel } = hoverData;
  const { traceUuid: runUuid, metricEntity } = singleTraceHoverData || {};

  const intl = useIntl();
  const { theme } = useDesignSystemTheme();

  const hoveredTraceUuid = `${runUuid}.${metricEntity?.key}`;
  const displayedXValueLabel =
    hoverData.xAxisKey === RunsChartsLineChartXAxisType.METRIC
      ? xAxisKeyLabel
      : intl.formatMessage(getChartAxisLabelDescriptor(hoverData.xAxisKey));

  if (tooltipLegendItems) {
    return (
      <div>
        {!isUndefined(xValue) && (
          <div css={{ marginBottom: theme.spacing.xs }}>
            <span css={{ fontWeight: 'bold' }}>{displayedXValueLabel}</span>{' '}
            {hoverData.xAxisKey === 'time' ? <PlotlyLikeFormattedTime value={xValue} /> : xValue}
          </div>
        )}
        <div
          css={{
            display: 'grid',
            gridTemplateColumns: `${theme.general.iconSize}px auto auto`,
            columnGap: theme.spacing.sm,
            rowGap: theme.spacing.sm / 4,
            alignItems: 'center',
          }}
        >
          {tooltipLegendItems.map(({ displayName, color, uuid, value, dashStyle }) => (
            <React.Fragment key={uuid}>
              <TraceLabelColorIndicator color={color || 'transparent'} dashStyle={dashStyle} />

              <div
                css={{
                  marginRight: theme.spacing.md,
                  fontSize: theme.typography.fontSizeSm,
                  color: hoveredTraceUuid === uuid ? 'unset' : theme.colors.textPlaceholder,
                }}
              >
                {displayName}
              </div>
              <div>
                {!isUndefined(value) && (
                  <span
                    css={{
                      fontWeight: hoveredTraceUuid === uuid ? 'bold' : 'normal',
                      color: hoveredTraceUuid === uuid ? 'unset' : theme.colors.textPlaceholder,
                    }}
                  >
                    {normalizeMetricChartTooltipValue(value, 2)}
                  </span>
                )}
              </div>
            </React.Fragment>
          ))}
        </div>
      </div>
    );
  }
  return null;
};
