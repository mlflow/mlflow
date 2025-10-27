import { useDesignSystemTheme } from '@databricks/design-system';
import { TracesView } from '../../traces/TracesView';
import {
  shouldEnableTracesV3View,
  isExperimentEvalResultsMonitoringUIEnabled,
} from '../../../../common/utils/FeatureUtils';
import { TracesV3View } from './traces-v3/TracesV3View';

export const ExperimentViewTraces = ({ experimentIds }: { experimentIds: string[] }) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div
      css={{
        minHeight: 225, // This is the exact height for displaying a minimum five rows and table header
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.sm,
        flex: 1,
        overflow: 'hidden',
      }}
    >
      <TracesComponent experimentIds={experimentIds} />
    </div>
  );
};

const TracesComponent = ({ experimentIds }: { experimentIds: string[] }) => {
  if (shouldEnableTracesV3View() || isExperimentEvalResultsMonitoringUIEnabled()) {
    return <TracesV3View experimentIds={experimentIds} />;
  }
  return <TracesView experimentIds={experimentIds} />;
};
