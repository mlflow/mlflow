import { useDesignSystemTheme } from '@databricks/design-system';
import { ChartsView } from './charts/ChartsView';
import {
  MonitoringConfigProvider,
} from '@mlflow/mlflow/src/experiment-tracking/hooks/useMonitoringConfig';

export const ExperimentViewCharts = ({ experimentIds }: { experimentIds: string[] }) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div
      css={{
        minHeight: 225,
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.sm,
        flex: 1,
        overflow: 'hidden',
      }}
    >
      <MonitoringConfigProvider>
        <ChartsView experimentIds={experimentIds} />
      </MonitoringConfigProvider>
    </div>
  );
};
