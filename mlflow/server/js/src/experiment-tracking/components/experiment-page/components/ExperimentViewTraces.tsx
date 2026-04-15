import { useDesignSystemTheme } from '@databricks/design-system';
import { TracesV3View } from './traces-v3/TracesV3View';
import { useGetExperimentQuery } from '../../../hooks/useExperimentQuery';

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
  const { loading: isLoadingExperiment } = useGetExperimentQuery({
    experimentId: experimentIds[0],
  });

  return <TracesV3View experimentIds={experimentIds} isLoadingExperiment={isLoadingExperiment} />;
};
