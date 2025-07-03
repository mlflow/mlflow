import { useDesignSystemTheme } from '@databricks/design-system';
import { TracesView } from '../../traces/TracesView';
import { ExperimentViewRunsModeSwitch } from './runs/ExperimentViewRunsModeSwitch';

export const ExperimentViewTraces = ({ experimentIds }: { experimentIds: string[] }) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div
      css={{
        minHeight: 225, // This is the exact height for displaying a minimum five rows and table header
        marginTop: theme.spacing.sm,
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.sm,
        flex: 1,
        overflow: 'hidden',
      }}
    >
      <ExperimentViewRunsModeSwitch hideBorder={false} />
      <TracesComponent experimentIds={experimentIds} />
    </div>
  );
};

const TracesComponent = ({ experimentIds }: { experimentIds: string[] }) => {
  return <TracesView experimentIds={experimentIds} />;
};
