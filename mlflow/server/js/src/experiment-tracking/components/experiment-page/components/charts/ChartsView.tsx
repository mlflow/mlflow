import { useMemo } from 'react';
import { useDesignSystemTheme } from '@databricks/design-system';
import { ChartsToolbar } from './ChartsToolbar';
import { TracesCountChart } from './TracesCountChart';
import { TokensCountChart } from './TokensCountChart';
import { TraceLatencyChart } from './TraceLatencyChart';
import { AssessmentAnalysisChart } from './AssessmentAnalysisChart';
import { TracesStatistics } from './TracesStatistics';
import {
  getAbsoluteStartEndTime,
  useMonitoringFilters,
} from '@mlflow/mlflow/src/experiment-tracking/hooks/useMonitoringFilters';
import { useMonitoringConfig } from '@mlflow/mlflow/src/experiment-tracking/hooks/useMonitoringConfig';

export const ChartsView = ({ experimentIds }: { experimentIds: string[] }) => {
  const { theme } = useDesignSystemTheme();
  const [monitoringFilters] = useMonitoringFilters();
  const monitoringConfig = useMonitoringConfig();

  const timeRange = useMemo(() => {
    const { startTime, endTime } = getAbsoluteStartEndTime(monitoringConfig.dateNow, monitoringFilters);
    return {
      startTime: startTime ? new Date(startTime).getTime().toString() : undefined,
      endTime: endTime ? new Date(endTime).getTime().toString() : undefined,
    };
  }, [monitoringConfig.dateNow, monitoringFilters]);

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.sm,
        height: '100%',
        overflowY: 'auto',
      }}
    >
      <ChartsToolbar />
      <div
        css={{
          padding: theme.spacing.md,
          display: 'flex',
          flexDirection: 'column',
          gap: theme.spacing.lg,
        }}
      >
        <TracesStatistics experimentIds={experimentIds} timeRange={timeRange} />
        <TracesCountChart experimentIds={experimentIds} timeRange={timeRange} />
        <TokensCountChart experimentIds={experimentIds} timeRange={timeRange} />
        <TraceLatencyChart experimentIds={experimentIds} timeRange={timeRange} />
        <AssessmentAnalysisChart experimentIds={experimentIds} timeRange={timeRange} />
      </div>
    </div>
  );
};
