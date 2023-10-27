import {
  Button,
  CloseIcon,
  PinIcon,
  PinFillIcon,
  Tooltip,
  VisibleIcon,
} from '@databricks/design-system';
import { Theme } from '@emotion/react';
import { FormattedMessage } from 'react-intl';
import { Link } from '../../../common/utils/RoutingUtils';
import Routes from '../../routes';
import { useExperimentIds } from '../experiment-page/hooks/useExperimentIds';
import { RunsChartsRunData } from '../runs-charts/components/RunsCharts.common';
import { RunsChartsTooltipBodyProps } from '../runs-charts/hooks/useRunsChartsTooltip';
import {
  RunsCompareBarCardConfig,
  RunsCompareCardConfig,
  RunsCompareChartType,
  RunsCompareScatterCardConfig,
  RunsCompareContourCardConfig,
  RunsCompareLineCardConfig,
  RunsCompareParallelCardConfig,
} from './runs-compare.types';
import { normalizeMetricChartTooltipValue } from '../../utils/MetricsUtils';
import type { RunsMetricsLinePlotHoverData } from '../runs-charts/components/RunsMetricsLinePlot';

interface CompareChartContextMenuContentDataType {
  runs: RunsChartsRunData[];
  onTogglePin?: (runUuid: string) => void;
  onHideRun?: (runUuid: string) => void;
}

type CompareChartContextMenuHoverDataType = RunsCompareCardConfig;

const createBarChartValuesBox = (
  cardConfig: RunsCompareBarCardConfig,
  activeRun: RunsChartsRunData,
) => {
  const { metricKey } = cardConfig;
  const metric = activeRun?.metrics[metricKey];

  if (!metric) {
    return null;
  }

  return (
    <div css={styles.value}>
      <strong>{metric.key}:</strong> {normalizeMetricChartTooltipValue(metric.value)}
    </div>
  );
};

const createScatterChartValuesBox = (
  cardConfig: RunsCompareScatterCardConfig,
  activeRun: RunsChartsRunData,
) => {
  const { xaxis, yaxis } = cardConfig;
  const xKey = xaxis.key;
  const yKey = yaxis.key;

  const xValue =
    xaxis.type === 'METRIC' ? activeRun.metrics[xKey]?.value : activeRun.params[xKey]?.value;

  const yValue =
    yaxis.type === 'METRIC' ? activeRun.metrics[yKey]?.value : activeRun.params[yKey]?.value;

  return (
    <>
      {xValue && (
        <div css={styles.value}>
          <strong>X ({xKey}):</strong> {normalizeMetricChartTooltipValue(xValue)}
        </div>
      )}
      {yValue && (
        <div css={styles.value}>
          <strong>Y ({yKey}):</strong> {normalizeMetricChartTooltipValue(yValue)}
        </div>
      )}
    </>
  );
};

const createContourChartValuesBox = (
  cardConfig: RunsCompareContourCardConfig,
  activeRun: RunsChartsRunData,
) => {
  const { xaxis, yaxis, zaxis } = cardConfig;
  const xKey = xaxis.key;
  const yKey = yaxis.key;
  const zKey = zaxis.key;

  const xValue =
    xaxis.type === 'METRIC' ? activeRun.metrics[xKey]?.value : activeRun.params[xKey]?.value;

  const yValue =
    yaxis.type === 'METRIC' ? activeRun.metrics[yKey]?.value : activeRun.params[yKey]?.value;

  const zValue =
    zaxis.type === 'METRIC' ? activeRun.metrics[zKey]?.value : activeRun.params[zKey]?.value;

  return (
    <>
      {xValue && (
        <div css={styles.value}>
          <strong>X ({xKey}):</strong> {normalizeMetricChartTooltipValue(xValue)}
        </div>
      )}
      {yValue && (
        <div css={styles.value}>
          <strong>Y ({yKey}):</strong> {normalizeMetricChartTooltipValue(yValue)}
        </div>
      )}
      {zValue && (
        <div css={styles.value}>
          <strong>Z ({zKey}):</strong> {normalizeMetricChartTooltipValue(zValue)}
        </div>
      )}
    </>
  );
};

const createLineChartValuesBox = (
  cardConfig: RunsCompareLineCardConfig,
  activeRun: RunsChartsRunData,
  hoverData?: RunsMetricsLinePlotHoverData,
) => {
  const { metricKey } = cardConfig;
  // If there's available value from x axis (step or time), extract entry from
  // metric history instead of latest metric.
  const metricValue = hoverData?.yValue || activeRun?.metrics[metricKey].value;

  if (!metricValue) {
    return null;
  }

  return (
    <>
      {hoverData && (
        <div css={styles.value}>
          <strong>{hoverData.label}:</strong> {hoverData.xValue}
        </div>
      )}
      <div css={styles.value}>
        <strong>{metricKey}:</strong> {normalizeMetricChartTooltipValue(metricValue)}
      </div>
    </>
  );
};

const createParallelChartValuesBox = (
  cardConfig: RunsCompareParallelCardConfig,
  activeRun: RunsChartsRunData,
  isHovering?: boolean,
) => {
  const { selectedParams, selectedMetrics } = cardConfig as RunsCompareParallelCardConfig;
  const paramsList = selectedParams.map((paramKey) => {
    const param = activeRun?.params[paramKey];
    if (param) {
      return (
        <div key={paramKey}>
          <strong>{param.key}:</strong> {normalizeMetricChartTooltipValue(param.value)}
        </div>
      );
    }
    return true;
  });
  const metricsList = selectedMetrics.map((metricKey) => {
    const metric = activeRun?.metrics[metricKey];
    if (metric) {
      return (
        <div key={metricKey}>
          <strong>{metric.key}:</strong> {normalizeMetricChartTooltipValue(metric.value)}
        </div>
      );
    }
    return true;
  });

  // show only first 3 params and primary metric if hovering, else show all
  if (isHovering) {
    return (
      <>
        {paramsList.slice(0, 3)}
        {(paramsList.length > 3 || metricsList.length > 1) && <div>...</div>}
        {metricsList[metricsList.length - 1]}
      </>
    );
  } else {
    return (
      <>
        {paramsList}
        {metricsList}
      </>
    );
  }
};

/**
 * Internal component that displays metrics/params - its final design
 * is a subject to change
 */
const ValuesBox = ({
  activeRun,
  cardConfig,
  isHovering,
  hoverData,
}: {
  activeRun: RunsChartsRunData;
  cardConfig: RunsCompareCardConfig;
  isHovering?: boolean;
  hoverData?: RunsMetricsLinePlotHoverData;
}) => {
  if (cardConfig.type === RunsCompareChartType.BAR) {
    return createBarChartValuesBox(cardConfig as RunsCompareBarCardConfig, activeRun);
  }

  if (cardConfig.type === RunsCompareChartType.SCATTER) {
    return createScatterChartValuesBox(cardConfig as RunsCompareScatterCardConfig, activeRun);
  }

  if (cardConfig.type === RunsCompareChartType.CONTOUR) {
    return createContourChartValuesBox(cardConfig as RunsCompareContourCardConfig, activeRun);
  }

  if (cardConfig.type === RunsCompareChartType.LINE) {
    return createLineChartValuesBox(cardConfig as RunsCompareLineCardConfig, activeRun, hoverData);
  }

  if (cardConfig.type === RunsCompareChartType.PARALLEL) {
    return createParallelChartValuesBox(
      cardConfig as RunsCompareParallelCardConfig,
      activeRun,
      isHovering,
    );
  }

  return null;
};

export const RunsCompareTooltipBody = ({
  closeContextMenu,
  contextData,
  hoverData,
  chartData,
  runUuid,
  isHovering,
}: RunsChartsTooltipBodyProps<
  CompareChartContextMenuContentDataType,
  CompareChartContextMenuHoverDataType,
  RunsMetricsLinePlotHoverData
>) => {
  const { runs, onTogglePin, onHideRun } = contextData;
  const [experimentId] = useExperimentIds();
  const activeRun = runs?.find((run) => run.runInfo.run_uuid === runUuid);

  if (!activeRun) {
    return null;
  }

  return (
    <div>
      <div css={styles.contentWrapper}>
        <div css={styles.header}>
          <div css={styles.colorPill} style={{ backgroundColor: activeRun.color }} />
          <Link
            to={Routes.getRunPageRoute(experimentId, runUuid)}
            target='_blank'
            css={styles.runLink}
            onClick={closeContextMenu}
          >
            {activeRun.runInfo.run_name || activeRun.runInfo.run_uuid}
          </Link>
        </div>
        {!isHovering && <Button size='small' onClick={closeContextMenu} icon={<CloseIcon />} />}
      </div>

      <ValuesBox
        isHovering={isHovering}
        activeRun={activeRun}
        cardConfig={chartData}
        hoverData={hoverData}
      />

      <div css={styles.actionsWrapper}>
        {activeRun.pinnable && onTogglePin && (
          <Tooltip
            title={
              activeRun.pinned ? (
                <FormattedMessage
                  defaultMessage='Unpin run'
                  description='A tooltip for the pin icon button in the runs table next to the pinned run'
                />
              ) : (
                <FormattedMessage
                  defaultMessage='Click to pin the run'
                  description='A tooltip for the pin icon button in the runs chart tooltip next to the not pinned run'
                />
              )
            }
            placement='bottom'
          >
            <Button
              size='small'
              onClick={() => {
                onTogglePin(runUuid);
                closeContextMenu();
              }}
              icon={activeRun.pinned ? <PinFillIcon /> : <PinIcon />}
            />
          </Tooltip>
        )}
        {onHideRun && (
          <Tooltip
            title={
              <FormattedMessage
                defaultMessage='Click to hide the run'
                description='A tooltip for the "hide" icon button in the runs chart tooltip'
              />
            }
            placement='bottom'
          >
            <Button
              data-testid='experiment-view-compare-runs-tooltip-visibility-button'
              size='small'
              onClick={() => {
                onHideRun(runUuid);
                closeContextMenu();
              }}
              icon={<VisibleIcon />}
            />
          </Tooltip>
        )}
      </div>
    </div>
  );
};

const styles = {
  runLink: (theme: Theme) => ({
    color: theme.colors.primary,
    '&:hover': {},
  }),
  actionsWrapper: {
    marginTop: 8,
    display: 'flex',
    gap: 8,
    alignItems: 'center',
  },
  header: {
    display: 'flex',
    gap: 8,
    alignItems: 'center',
  },
  value: {
    maxWidth: 300,
    whiteSpace: 'nowrap' as const,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
  },
  contentWrapper: {
    display: 'flex',
    gap: 8,
    alignItems: 'center',
    marginBottom: 12,
    justifyContent: 'space-between',
    height: 24,
  },
  colorPill: { width: 12, height: 12, borderRadius: '100%' },
};
