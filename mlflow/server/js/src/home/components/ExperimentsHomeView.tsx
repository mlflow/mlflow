import { useMemo, useState } from 'react';
import {
  Alert,
  BeakerIcon,
  Button,
  Empty,
  PlusIcon,
  Spacer,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import type { RowSelectionState, SortingState } from '@tanstack/react-table';
import { FormattedMessage } from 'react-intl';
import { ExperimentListTable } from '../../experiment-tracking/components/ExperimentListTable';
import Routes from '../../experiment-tracking/routes';
import { Link } from '../../common/utils/RoutingUtils';
import type { ExperimentEntity } from '../../experiment-tracking/types';
import { isDemoExperiment } from '../../experiment-tracking/utils/isDemoExperiment';
import { useUpdateExperimentTags } from '../../experiment-tracking/components/experiment-page/hooks/useUpdateExperimentTags';

type ExperimentsHomeViewProps = {
  experiments?: ExperimentEntity[];
  isLoading: boolean;
  error?: Error | null;
  onCreateExperiment: () => void;
  onRetry: () => void;
  onTagsUpdated?: () => void;
};

const ExperimentsEmptyState = ({ onCreateExperiment }: { onCreateExperiment: () => void }) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: theme.spacing.lg,
      }}
    >
      <Empty
        image={<BeakerIcon />}
        title={
          <FormattedMessage
            defaultMessage="Create your first experiment"
            description="Home page experiments empty state title"
          />
        }
        description={
          <FormattedMessage
            defaultMessage="Create your first experiment to start tracking ML workflows."
            description="Home page experiments empty state description"
          />
        }
        button={
          <Button
            componentId="mlflow.home.experiments.create"
            onClick={onCreateExperiment}
            type="primary"
            icon={<PlusIcon />}
          >
            <FormattedMessage defaultMessage="Create experiment" description="Home page experiments empty state CTA" />
          </Button>
        }
      />
    </div>
  );
};

export const ExperimentsHomeView = ({
  experiments,
  isLoading,
  error,
  onCreateExperiment,
  onRetry,
  onTagsUpdated,
}: ExperimentsHomeViewProps) => {
  const { theme } = useDesignSystemTheme();
  const [rowSelection, setRowSelection] = useState<RowSelectionState>({});
  const [sorting, setSorting] = useState<SortingState>([]);
  const { EditTagsModal, showEditExperimentTagsModal } = useUpdateExperimentTags({
    onSuccess: onTagsUpdated,
  });

  const topExperiments = useMemo(() => {
    const sliced = experiments?.slice(0, 5) ?? [];
    const demo = sliced.filter(isDemoExperiment);
    const nonDemo = sliced.filter((e) => !isDemoExperiment(e));
    return [...demo, ...nonDemo];
  }, [experiments]);
  const shouldShowEmptyState = !isLoading && !error && topExperiments.length === 0;

  return (
    <section css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: theme.spacing.md,
          paddingRight: theme.spacing.sm,
        }}
      >
        <Typography.Title level={3} css={{ margin: 0 }}>
          <FormattedMessage defaultMessage="Recent Experiments" description="Home page experiments preview title" />
        </Typography.Title>
        <Link to={Routes.experimentsObservatoryRoute}>
          <FormattedMessage defaultMessage="View all" description="Home page experiments view all link" />
        </Link>
      </div>
      <div
        css={{
          border: `1px solid ${theme.colors.borderDecorative}`,
          borderRadius: theme.general.borderRadiusBase,
          overflow: 'hidden',
          backgroundColor: theme.colors.backgroundPrimary,
        }}
      >
        {error ? (
          <div css={{ padding: theme.spacing.lg, display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
            <Alert
              type="error"
              closable={false}
              componentId="mlflow.home.experiments.error"
              message={
                <FormattedMessage
                  defaultMessage="We couldn't load your experiments."
                  description="Home page experiments error message"
                />
              }
              description={error.message}
            />
            <div>
              <Button componentId="mlflow.home.experiments.retry" onClick={onRetry}>
                <FormattedMessage defaultMessage="Retry" description="Home page experiments retry CTA" />
              </Button>
            </div>
          </div>
        ) : shouldShowEmptyState ? (
          <ExperimentsEmptyState onCreateExperiment={onCreateExperiment} />
        ) : (
          <ExperimentListTable
            experiments={topExperiments}
            isLoading={isLoading}
            rowSelection={rowSelection}
            setRowSelection={setRowSelection}
            sortingProps={{ sorting, setSorting }}
            onEditTags={showEditExperimentTagsModal}
          />
        )}
      </div>
      <Spacer shrinks={false} />
      {EditTagsModal}
    </section>
  );
};

export default ExperimentsHomeView;
