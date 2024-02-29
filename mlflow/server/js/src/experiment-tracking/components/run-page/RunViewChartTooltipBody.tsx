import { Button, CloseIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import type { MetricHistoryByName, RunInfoEntity } from '../../types';
import {
  containsMultipleRunsTooltipData,
  RunsChartsTooltipMode,
  type RunsChartsTooltipBodyProps,
} from '../runs-charts/hooks/useRunsChartsTooltip';
import { getStableColorForRun } from '../../utils/RunNameUtils';
import { isSystemMetricKey, normalizeChartMetricKey, normalizeMetricChartTooltipValue } from '../../utils/MetricsUtils';
import Utils from '../../../common/utils/Utils';
import { FormattedMessage } from 'react-intl';
import { first, isUndefined } from 'lodash';
import type {
  RunsCompareMultipleTracesTooltipData,
  RunsMetricsSingleTraceTooltipData,
} from '../runs-charts/components/RunsMetricsLinePlot';
import type { RunsMetricsBarPlotHoverData } from '../runs-charts/components/RunsMetricsBarPlot';
import { RunsMultipleTracesTooltipBody } from '../runs-charts/components/RunsMultipleTracesTooltipBody';

/**
 * Tooltip body displayed when hovering over run view metric charts
 */
export const RunViewChartTooltipBody = ({
  contextData: { runInfo, metricsForRun },
  hoverData,
  chartData: { metricKey },
  isHovering,
  mode,
}: RunsChartsTooltipBodyProps<
  { runInfo: RunInfoEntity; metricsForRun: MetricHistoryByName },
  { metricKey: string },
  RunsMetricsBarPlotHoverData | RunsMetricsSingleTraceTooltipData | RunsCompareMultipleTracesTooltipData
>) => {
  const singleTraceHoverData = containsMultipleRunsTooltipData(hoverData) ? hoverData.hoveredDataPoint : hoverData;

  if (
    mode === RunsChartsTooltipMode.MultipleTracesWithScanline &&
    containsMultipleRunsTooltipData(hoverData) &&
    isHovering
  ) {
    return <RunsMultipleTracesTooltipBody hoverData={hoverData} />;
  }

  if (!singleTraceHoverData?.metricEntity) {
    return null;
  }

  const { timestamp, step, value } = singleTraceHoverData.metricEntity;

  const metricContainsHistory = metricsForRun?.[metricKey]?.length > 1;
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
