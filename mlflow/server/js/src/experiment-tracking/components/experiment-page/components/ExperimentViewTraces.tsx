import { useDesignSystemTheme } from '@databricks/design-system';
import { TracesView } from '../../traces/TracesView';
import { ExperimentViewRunsModeSwitch } from './runs/ExperimentViewRunsModeSwitch';

export const ExperimentViewTraces = ({ experimentIds }: { experimentIds: string[] }) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div
      css={{
        marginTop: theme.spacing.md,
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.sm,
        flex: 1,
        overflow: 'hidden',
      }}
    >
      <ExperimentViewRunsModeSwitch hideBorder={false} />
      <TracesView experimentIds={experimentIds} />
    </div>
  );
};
