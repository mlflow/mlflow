import {
  Empty,
  Input,
  SearchIcon,
  Spacer,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { useMemo, useState } from 'react';
import { MetricHistoryByName, RunInfoEntity } from '../../types';
import { RunViewMetricChart } from './RunViewMetricChart';
import { useSelector } from 'react-redux';
import { ReduxState } from '../../../redux-types';
import { useFetchCompareRunsMetricHistory } from '../runs-compare/hooks/useFetchCompareRunsMetricHistory';
import { CollapsibleSection } from '../../../common/components/CollapsibleSection';
import { FormattedMessage, defineMessages, useIntl } from 'react-intl';
import { RunsChartsTooltipWrapper } from '../runs-charts/hooks/useRunsChartsTooltip';
import { RunViewChartTooltipBody } from './RunViewChartTooltipBody';
import { isSystemMetricKey, normalizeChartMetricKey } from '../../utils/MetricsUtils';

const { systemMetricChartsLabel, modelMetricChartsLabel } = defineMessages({
  systemMetricChartsLabel: {
    defaultMessage: 'System metrics',
    description: 'Run page > Charts tab > System charts section > title',
  },
  modelMetricChartsLabel: {
    defaultMessage: 'Model metrics',
    description: 'Run page > Charts tab > Model charts section > title',
  },
});

const EmptyMetricsFiltered = () => (
  <Empty
    title='No matching metric keys'
    description='All metrics in this section are filtered. Clear the search filter to see hidden metrics.'
  />
);

const EmptyMetricsNotRecorded = () => (
  <Empty title='No metrics recorded' description='No metrics recorded' />
);

const metricKeyMatchesFilter = (filter: string, metricKey: string) =>
  metricKey.toLowerCase().includes(filter.toLowerCase());

/**
 * Internal component that displays a single collapsible section with charts
 */
const RunViewMetricChartsSection = ({
  metricKeys,
  title,
  search,
  metrics,
  runInfo,
  isLoading,
}: {
  metricKeys: string[];
  title: string;
  search: string;
  metrics: MetricHistoryByName;
  runInfo: RunInfoEntity;
  isLoading: boolean;
}) => {
  const { theme } = useDesignSystemTheme();

  const filteredMetricKeys = metricKeys.filter((metricKey) =>
    metricKeyMatchesFilter(search, metricKey),
  );

  const charts = filteredMetricKeys.length ? (
    <div css={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24, overflow: 'hidden' }}>
      {metricKeys.map((metricKey) => (
        <div
          key={metricKey}
          css={{
            border: `1px solid ${theme.colors.borderDecorative}`,
            padding: 16,
            overflow: 'hidden',
            display: metricKeyMatchesFilter(search, metricKey) ? 'block' : 'none',
          }}
          aria-hidden={!metricKeyMatchesFilter(search, metricKey)}
        >
          <Typography.Title level={4}>{normalizeChartMetricKey(metricKey)}</Typography.Title>
          {
            <RunViewMetricChart
              isLoading={isLoading}
              metricKey={metricKey}
              metricsHistory={metrics}
              runInfo={runInfo}
            />
          }
        </div>
      ))}
    </div>
  ) : (
    <EmptyMetricsFiltered />
  );

  if (!title) return charts;

  return (
    <CollapsibleSection
      title={
        <Typography.Title level={3} css={{ marginBottom: 0 }}>
          {title} ({filteredMetricKeys.length})
        </Typography.Title>
      }
    >
      {charts}
    </CollapsibleSection>
  );
};

/**
 * Component displaying metric charts for a single run
 */
export const RunViewMetricCharts = ({
  runInfo,
  metricKeys,
}: {
  metricKeys: string[];
  runInfo: RunInfoEntity;
}) => {
  const { metricsForRun } = useSelector(({ entities }: ReduxState) => ({
    metricsForRun: entities.metricsByRunUuid[runInfo.run_uuid],
  }));

  const [search, setSearch] = useState('');
  const { formatMessage } = useIntl();

  const { isLoading } = useFetchCompareRunsMetricHistory(metricKeys, [{ runInfo }]);

  // Let's split charts into segments - system and model.
  // If there is no distinction detected, return only one generic section.
  const chartSegments = useMemo(() => {
    const systemMetricKeys = metricKeys.filter(isSystemMetricKey);
    const modelMetricKeys = metricKeys.filter((key) => !isSystemMetricKey(key));

    const isSegmented = systemMetricKeys.length > 0 && modelMetricKeys.length > 0;
    const segments: [string, string[]][] = isSegmented
      ? [
          [formatMessage(systemMetricChartsLabel), systemMetricKeys],
          [formatMessage(modelMetricChartsLabel), modelMetricKeys],
        ]
      : [['', metricKeys]];

    return segments;
  }, [metricKeys, formatMessage]);

  const noMetricsRecorded = !metricKeys.length;
  const allMetricsFilteredOut =
    !noMetricsRecorded &&
    !metricKeys.some((metricKey) => metricKeyMatchesFilter(search, metricKey));
  const showFilterInput = !noMetricsRecorded;
  const showCharts = !noMetricsRecorded && !allMetricsFilteredOut;

  const tooltipContext = useMemo(() => ({ runInfo, metricsForRun }), [runInfo, metricsForRun]);

  return (
    <RunsChartsTooltipWrapper contextData={tooltipContext} component={RunViewChartTooltipBody}>
      <Typography.Title level={3}>
        <FormattedMessage
          defaultMessage='Metric charts'
          description='Run page > Charts tab > title'
        />
      </Typography.Title>
      {showFilterInput && (
        <Input
          role='searchbox'
          prefix={<SearchIcon />}
          value={search}
          allowClear
          onChange={(e) => setSearch(e.target.value)}
          placeholder={formatMessage({
            defaultMessage: 'Search metric charts',
            description: 'Run page > Charts tab > Filter metric charts input > placeholder',
          })}
        />
      )}
      <Spacer />
      {noMetricsRecorded && <EmptyMetricsNotRecorded />}
      {allMetricsFilteredOut && <EmptyMetricsFiltered />}
      {showCharts &&
        chartSegments.map(([title, segmentMetricKeys]) => (
          <RunViewMetricChartsSection
            key={title}
            isLoading={isLoading}
            metricKeys={segmentMetricKeys}
            title={title}
            metrics={metricsForRun}
            runInfo={runInfo}
            search={search}
          />
        ))}
    </RunsChartsTooltipWrapper>
  );
};
