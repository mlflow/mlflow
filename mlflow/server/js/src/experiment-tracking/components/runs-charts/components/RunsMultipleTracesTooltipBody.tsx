import { isUndefined } from 'lodash';
import Utils from '../../../../common/utils/Utils';
import { computeColumnFormatSpec } from '../../experiment-page/utils/metricColumnFormat';
import { useSmartNumberFormattingEnabled } from '../../experiment-page/utils/useSmartNumberFormatting';

import type { RunsCompareMultipleTracesTooltipData } from './RunsMetricsLinePlot';
import React from 'react';
import { TraceLabelColorIndicator } from './RunsMetricsLegend';
import { FormattedDate, FormattedTime, useIntl } from 'react-intl';
import { getChartAxisLabelDescriptor, RunsChartsLineChartXAxisType } from './RunsCharts.common';
import { useDesignSystemTheme } from '@databricks/design-system';
import { shouldEnableRelativeTimeDateAxis } from '@mlflow/mlflow/src/common/utils/FeatureUtils';

// Sadly when hovering outside data point, we can't get the date-time value from Plotly chart
// so we have to format it ourselves in a way that resembles Plotly's logic
const PlotlyLikeFormattedTime = ({ value }: { value: string | number }) => (
  <>
    <FormattedDate value={value} year="numeric" />-
    <FormattedDate value={value} month="2-digit" />-
    <FormattedDate value={value} day="2-digit" /> <FormattedTime value={value} hour="numeric" hourCycle="h24" />:
    <FormattedTime value={value} minute="2-digit" />:
    <FormattedTime value={value} second="2-digit" fractionalSecondDigits={3} />
  </>
);

// Sadly when hovering outside data point, we can't get the date-time value from Plotly chart
// so we have to format it ourselves in a way that resembles Plotly's logic
const PlotlyLikeFormattedTimestamp = ({ value }: { value: string | number }) => (
  <>
    <FormattedTime value={value} hour="2-digit" hourCycle="h23" />:
    <FormattedTime value={value} minute="2-digit" />:
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
  const smartFormatting = useSmartNumberFormattingEnabled();

  const hoveredTraceUuid = `${runUuid}.${metricEntity?.key}`;
  const displayedXValueLabel =
    hoverData.xAxisKey === RunsChartsLineChartXAxisType.METRIC
      ? xAxisKeyLabel
      : intl.formatMessage(getChartAxisLabelDescriptor(hoverData.xAxisKey));

  if (tooltipLegendItems) {
    // Prefer the pre-computed spec (derived from ALL y-values in the chart, across all steps)
    // so that formatting is consistent even at steps where every visible value happens to be 0.
    // Fall back to computing from just the visible step's values when no global spec is available.
    const allValues = tooltipLegendItems.map((item) => item.value).filter((v): v is number => typeof v === 'number');
    const formatSpec = smartFormatting
      ? (hoverData.formatSpec ?? computeColumnFormatSpec(allValues))
      : null;

    return (
      <div>
        {!isUndefined(xValue) && (
          <div css={{ marginBottom: theme.spacing.xs }}>
            <span css={{ fontWeight: 'bold' }}>{displayedXValueLabel}</span>{' '}
            {hoverData.xAxisKey === RunsChartsLineChartXAxisType.TIME ? (
              <PlotlyLikeFormattedTime value={xValue} />
            ) : shouldEnableRelativeTimeDateAxis() &&
              hoverData.xAxisKey === RunsChartsLineChartXAxisType.TIME_RELATIVE ? (
              <PlotlyLikeFormattedTimestamp value={xValue} />
            ) : (
              xValue
            )}
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
          {formatSpec?.headerAnnotation && (
            <>
              <span />
              <span />
              <span
                css={{
                  fontSize: theme.typography.fontSizeSm,
                  color: theme.colors.textSecondary,
                  textAlign: 'right',
                }}
              >
                {formatSpec.headerAnnotation}
              </span>
            </>
          )}
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
                    {formatSpec ? formatSpec.format(value) : Utils.formatMetric(value)}
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
