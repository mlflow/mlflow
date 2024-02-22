import { Button, CloseIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import type { MetricHistoryByName, RunInfoEntity } from '../../types';
import type { RunsChartsTooltipBodyProps } from '../runs-charts/hooks/useRunsChartsTooltip';
import { getStableColorForRun } from '../../utils/RunNameUtils';
import { isSystemMetricKey, normalizeChartMetricKey, normalizeMetricChartTooltipValue } from '../../utils/MetricsUtils';
import Utils from '../../../common/utils/Utils';
import { FormattedMessage } from 'react-intl';
import { first, isUndefined } from 'lodash';
import type { RunsMetricsLinePlotHoverData } from '../runs-charts/components/RunsMetricsLinePlot';
import type { RunsMetricsBarPlotHoverData } from '../runs-charts/components/RunsMetricsBarPlot';

type RunViewChartTooltipHoverData = RunsMetricsLinePlotHoverData | RunsMetricsBarPlotHoverData;

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

  if (!hoverData.metricEntity) {
    return null;
  }

  const { timestamp, step, value } = hoverData.metricEntity;

  const metricContainsHistory = metricsForRun?.[metricKey].length > 1;
  const isSystemMetric = isSystemMetricKey(metricKey);
  const displayTimestamp = metricContainsHistory && isSystemMetric && !isUndefined(timestamp);
  const displayStep = metricContainsHistory && !isSystemMetric && !isUndefined(step);

  return (
    <div>
      {displayStep && (
        <div css={styles.valueField}>
          <strong>
            <FormattedMessage defaultMessage="Step" description="Run page > Charts tab > Chart tooltip > Step label" />:
          </strong>{' '}
          {step}
        </div>
      )}
      {displayTimestamp && (
        <div css={styles.valueField}>
          <strong>
            <FormattedMessage
              defaultMessage="Timestamp"
              description="Run page > Charts tab > Chart tooltip > Timestamp label"
            />
            :
          </strong>{' '}
          {Utils.formatTimestamp(timestamp)}
        </div>
      )}
      {value && (
        <div css={styles.valueField}>
          <strong>{metricKey}:</strong> {normalizeMetricChartTooltipValue(value)}
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
