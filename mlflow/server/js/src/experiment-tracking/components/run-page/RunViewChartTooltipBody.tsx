import { Button, CloseIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import type { MetricHistoryByName, RunInfoEntity } from '../../types';
import type { RunsChartsTooltipBodyProps } from '../runs-charts/hooks/useRunsChartsTooltip';
import { getStableColorForRun } from '../../utils/RunNameUtils';
import {
  isSystemMetricKey,
  normalizeChartMetricKey,
  normalizeMetricChartTooltipValue,
} from '../../utils/MetricsUtils';
import Utils from '../../../common/utils/Utils';
import { FormattedMessage } from 'react-intl';
import { first, isUndefined } from 'lodash';
import type { RunsMetricsLinePlotHoverData } from '../runs-charts/components/RunsMetricsLinePlot';
import type { RunsMetricsBarPlotHoverData } from '../runs-charts/components/RunsMetricsBarPlot';

type RunViewChartTooltipHoverData = RunsMetricsLinePlotHoverData | RunsMetricsBarPlotHoverData;

/**
 * Internal util function, returns values displayed in a tooltip
 * based on the metric history
 */
const getDisplayedMetricData = (
  metricsForRun: MetricHistoryByName,
  metricKey: string,
  hoverData: RunViewChartTooltipHoverData,
) => {
  const metricEntities = metricsForRun?.[metricKey];
  const hoveredIndex = hoverData?.index;

  // Display value and step or timestamp if there's a history
  if (metricEntities?.length > 1 && !isUndefined(hoveredIndex)) {
    const { step, timestamp, value } = metricEntities?.[hoveredIndex];
    return { step, timestamp, value };
  }

  // If there's no metric history (only one entry), display only its value
  const firstMetric = first(metricEntities);
  const { value } = firstMetric || {};
  return { value };
};

/**
 * Tooltip body displayed when hovering over run view metric charts
 */
export const RunViewChartTooltipBody = ({
  contextData: { runInfo, metricsForRun },
  hoverData,
  chartData: { metricKey },
  closeContextMenu,
  isHovering,
}: RunsChartsTooltipBodyProps<
  { runInfo: RunInfoEntity; metricsForRun: MetricHistoryByName },
  { metricKey: string },
  RunViewChartTooltipHoverData
>) => {
  const { theme } = useDesignSystemTheme();

  const { timestamp, step, value } = getDisplayedMetricData(metricsForRun, metricKey, hoverData);

  const color = getStableColorForRun(runInfo.run_uuid);

  const displayedMetricKey = normalizeChartMetricKey(metricKey);
  const isSystemMetric = isSystemMetricKey(metricKey);
  const displayTimestamp = isSystemMetric && !isUndefined(timestamp);
  const displayStep = !isSystemMetric && !isUndefined(step);

  return (
    <div>
      <div
        css={{
          display: 'flex',
          gap: theme.spacing.sm,
          alignItems: 'center',
          marginBottom: 12,
          justifyContent: 'space-between',
          height: theme.typography.lineHeightLg,
        }}
      >
        <div
          css={{
            display: 'flex',
            gap: theme.spacing.sm,
            alignItems: 'center',
          }}
        >
          <div
            css={{ width: 12, height: 12, borderRadius: '100%' }}
            style={{ backgroundColor: color }}
          />
          <Typography.Hint>{runInfo.run_name || runInfo.run_uuid}</Typography.Hint>
        </div>
        {!isHovering && <Button size='small' onClick={closeContextMenu} icon={<CloseIcon />} />}
      </div>

      {displayStep && (
        <div css={styles.valueField}>
          <strong>
            <FormattedMessage
              defaultMessage='Step'
              description='Run page > Charts tab > Chart tooltip > Step label'
            />
            :
          </strong>{' '}
          {step}
        </div>
      )}
      {displayTimestamp && (
        <div css={styles.valueField}>
          <strong>
            <FormattedMessage
              defaultMessage='Timestamp'
              description='Run page > Charts tab > Chart tooltip > Timestamp label'
            />
            :
          </strong>{' '}
          {Utils.formatTimestamp(timestamp)}
        </div>
      )}
      {value && (
        <div css={styles.valueField}>
          <strong>{displayedMetricKey}:</strong> {normalizeMetricChartTooltipValue(value)}
        </div>
      )}
    </div>
  );
};

const styles = {
  valueField: {
    maxWidth: 300,
    whiteSpace: 'nowrap' as const,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
  },
};
