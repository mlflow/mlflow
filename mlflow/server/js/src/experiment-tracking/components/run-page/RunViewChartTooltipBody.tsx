import type { MetricHistoryByName, RunInfoEntity } from '../../types';
import {
  containsMultipleRunsTooltipData,
  RunsChartsTooltipMode,
  type RunsChartsTooltipBodyProps,
} from '../runs-charts/hooks/useRunsChartsTooltip';
import { isSystemMetricKey } from '../../utils/MetricsUtils';
import Utils from '../../../common/utils/Utils';
import { FormattedMessage, useIntl } from 'react-intl';
import { isUndefined } from 'lodash';
import type {
  RunsCompareMultipleTracesTooltipData,
  RunsMetricsSingleTraceTooltipData,
} from '../runs-charts/components/RunsMetricsLinePlot';
import type { RunsMetricsBarPlotHoverData } from '../runs-charts/components/RunsMetricsBarPlot';
import { RunsMultipleTracesTooltipBody } from '../runs-charts/components/RunsMultipleTracesTooltipBody';
import { Spacer, Typography } from '@databricks/design-system';

/**
 * Tooltip body displayed when hovering over run view metric charts
 */
export const RunViewChartTooltipBody = ({
  contextData: { metricsForRun },
  hoverData,
  chartData: { metricKey },
  isHovering,
  mode,
}: RunsChartsTooltipBodyProps<
  { metricsForRun: MetricHistoryByName },
  { metricKey: string },
  RunsMetricsBarPlotHoverData | RunsMetricsSingleTraceTooltipData | RunsCompareMultipleTracesTooltipData
>) => {
  const singleTraceHoverData = containsMultipleRunsTooltipData(hoverData) ? hoverData.hoveredDataPoint : hoverData;
  const intl = useIntl();

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
          {Utils.formatTimestamp(timestamp, intl)}
        </div>
      )}
      {value && (
        <div>
          <Typography.Text bold>{metricKey}</Typography.Text>
          <Spacer size="xs" />
          <Typography.Text>{value}</Typography.Text>
        </div>
      )}
    </div>
  );
};

const styles = {
  valueField: {
    whiteSpace: 'nowrap' as const,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
  },
};
