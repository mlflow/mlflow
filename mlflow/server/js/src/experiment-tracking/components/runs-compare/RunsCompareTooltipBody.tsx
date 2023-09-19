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
import { CompareChartRunData, truncateString } from './charts/CompareRunsCharts.common';
import { CompareRunsTooltipBodyProps } from './hooks/useCompareRunsTooltip';
import {
  RunsCompareBarCardConfig,
  RunsCompareCardConfig,
  RunsCompareChartType,
  RunsCompareScatterCardConfig,
  RunsCompareContourCardConfig,
  RunsCompareLineCardConfig,
  RunsCompareParallelCardConfig,
} from './runs-compare.types';

interface LineXAxisValues {
  value: number;
  index: number;
  label: string;
}

interface CompareChartContextMenuContentDataType {
  runs: CompareChartRunData[];
  onTogglePin?: (runUuid: string) => void;
  onHideRun?: (runUuid: string) => void;
}

type CompareChartContextMenuHoverDataType = RunsCompareCardConfig;

const normalizeValue = (value: string | number, decimalPlaces = 6) => {
  if (typeof value === 'number') {
    return value.toFixed(decimalPlaces);
  }
  // cast to numbers that have for values that have been previously stringified
  const castToNumber = Number(value);
  if (!isNaN(castToNumber)) {
    return castToNumber.toFixed(decimalPlaces);
  }
  // truncate strings that are too long
  return truncateString(value, 8);
};

const createBarChartValuesBox = (
  cardConfig: RunsCompareBarCardConfig,
  activeRun: CompareChartRunData,
) => {
  const { metricKey } = cardConfig;
  const metric = activeRun?.metrics[metricKey];

  if (!metric) {
    return null;
  }

  return (
    <div css={styles.value}>
      <strong>{metric.key}:</strong> {normalizeValue(metric.value)}
    </div>
  );
};

const createScatterChartValuesBox = (
  cardConfig: RunsCompareScatterCardConfig,
  activeRun: CompareChartRunData,
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
          <strong>X ({xKey}):</strong> {normalizeValue(xValue)}
        </div>
      )}
      {yValue && (
        <div css={styles.value}>
          <strong>Y ({yKey}):</strong> {normalizeValue(yValue)}
        </div>
      )}
    </>
  );
};

const createContourChartValuesBox = (
  cardConfig: RunsCompareContourCardConfig,
  activeRun: CompareChartRunData,
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
          <strong>X ({xKey}):</strong> {normalizeValue(xValue)}
        </div>
      )}
      {yValue && (
        <div css={styles.value}>
          <strong>Y ({yKey}):</strong> {normalizeValue(yValue)}
        </div>
      )}
      {zValue && (
        <div css={styles.value}>
          <strong>Z ({zKey}):</strong> {normalizeValue(zValue)}
        </div>
      )}
    </>
  );
};

const createLineChartValuesBox = (
  cardConfig: RunsCompareLineCardConfig,
  activeRun: CompareChartRunData,
  xAxisValues?: LineXAxisValues,
) => {
  const { metricKey } = cardConfig;
  const metric =
    // If there's available value from x axis (step or time), extract entry from
    // metric history instead of latest metric.
    (xAxisValues && activeRun?.metricsHistory?.[metricKey]?.[xAxisValues?.index]) ??
    activeRun?.metrics[metricKey];

  if (!metric) {
    return null;
  }

  return (
    <>
      {xAxisValues && (
        <div css={styles.value}>
          <strong>{xAxisValues.label}:</strong> {xAxisValues.value}
        </div>
      )}
      <div css={styles.value}>
        <strong>{metric.key}:</strong> {normalizeValue(metric.value)}
      </div>
    </>
  );
};

const createParallelChartValuesBox = (
  cardConfig: RunsCompareParallelCardConfig,
  activeRun: CompareChartRunData,
  isHovering?: boolean,
) => {
  const { selectedParams, selectedMetrics } = cardConfig as RunsCompareParallelCardConfig;
  const paramsList = selectedParams.map((paramKey) => {
    const param = activeRun?.params[paramKey];
    if (param) {
      return (
        <div>
          <strong>{param.key}:</strong> {normalizeValue(param.value)}
        </div>
      );
    }
    return true;
  });
  const metricsList = selectedMetrics.map((metricKey) => {
    const metric = activeRun?.metrics[metricKey];
    if (metric) {
      return (
        <div>
          <strong>{metric.key}:</strong> {normalizeValue(metric.value)}
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
        {metricsList[metricsList.length - 1]}
        {(paramsList.length > 3 || metricsList.length > 1) && <div>...</div>}
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
  xAxisValues,
}: {
  activeRun: CompareChartRunData;
  cardConfig: RunsCompareCardConfig;
  isHovering?: boolean;
  xAxisValues?: LineXAxisValues;
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
    return createLineChartValuesBox(
      cardConfig as RunsCompareLineCardConfig,
      activeRun,
      xAxisValues,
    );
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
  additionalAxisData,
  hoverData: cardConfig,
  runUuid,
  isHovering,
}: CompareRunsTooltipBodyProps<
  CompareChartContextMenuContentDataType,
  CompareChartContextMenuHoverDataType
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
        cardConfig={cardConfig}
        xAxisValues={additionalAxisData}
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
