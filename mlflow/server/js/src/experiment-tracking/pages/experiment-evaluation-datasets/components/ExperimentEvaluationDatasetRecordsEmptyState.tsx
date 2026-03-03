import { useState } from 'react';
import { Button, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { Link, useParams } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import Routes from '@mlflow/mlflow/src/experiment-tracking/routes';
import { AddManuallyModal } from './AddManuallyModal';
import { AddViaCodeModal } from './AddViaCodeModal';

const ActionCard = ({
  title,
  description,
  onClick,
}: {
  title: React.ReactNode;
  description: React.ReactNode;
  onClick?: () => void;
}) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      role="button"
      tabIndex={0}
      onClick={onClick}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          onClick?.();
        }
      }}
      css={{
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.xs,
        padding: theme.spacing.md,
        border: `1px solid ${theme.colors.border}`,
        borderRadius: theme.borders.borderRadiusMd,
        cursor: 'pointer',
        flex: 1,
        minWidth: 180,
        '&:hover': {
          borderColor: theme.colors.actionDefaultBorderHover,
          backgroundColor: theme.colors.actionDefaultBackgroundHover,
        },
      }}
    >
      <Typography.Text bold>{title}</Typography.Text>
      <Typography.Text color="secondary" size="sm">
        {description}
      </Typography.Text>
    </div>
  );
};

export const ExperimentEvaluationDatasetRecordsEmptyState = ({
  datasetId,
  datasetName,
}: {
  datasetId: string;
  datasetName: string;
}) => {
  const { theme } = useDesignSystemTheme();
  const { experimentId } = useParams();
  const [addManuallyVisible, setAddManuallyVisible] = useState(false);
  const [addViaCodeVisible, setAddViaCodeVisible] = useState(false);

  const tracesRoute = experimentId ? Routes.getExperimentPageTracesTabRoute(experimentId) : undefined;

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        padding: theme.spacing.lg,
        gap: theme.spacing.sm,
        textAlign: 'center',
      }}
    >
      <Typography.Title level={4} color="secondary">
        <FormattedMessage
          defaultMessage="Add items to your dataset"
          description="Title for the empty state when an evaluation dataset has no records"
        />
      </Typography.Title>
      <Typography.Paragraph color="secondary" css={{ maxWidth: 460, marginBottom: theme.spacing.sm }}>
        <FormattedMessage
          defaultMessage="Datasets are collections of specific edge cases and underrepresented patterns used to evaluate your application."
          description="Description for the empty state when an evaluation dataset has no records"
        />
      </Typography.Paragraph>
      <div
        css={{
          display: 'flex',
          gap: theme.spacing.md,
          flexWrap: 'wrap',
          justifyContent: 'center',
          maxWidth: 700,
        }}
      >
        <ActionCard
          title={
            <FormattedMessage
              defaultMessage="Add Manually"
              description="Action card title for manually adding a record to a dataset"
            />
          }
          description={
            <FormattedMessage
              defaultMessage="Manually input a single item"
              description="Action card description for manually adding a record to a dataset"
            />
          }
          onClick={() => setAddManuallyVisible(true)}
        />
        <ActionCard
          title={
            <FormattedMessage
              defaultMessage="Add via Code"
              description="Action card title for adding records via Python SDK"
            />
          }
          description={
            <FormattedMessage
              defaultMessage="Use the MLflow Python SDK"
              description="Action card description for adding records via Python SDK"
            />
          }
          onClick={() => setAddViaCodeVisible(true)}
        />
        {tracesRoute && (
          <Link to={tracesRoute} css={{ flex: 1, minWidth: 180, textDecoration: 'none' }}>
            <ActionCard
              title={
                <FormattedMessage
                  defaultMessage="Select Traces"
                  description="Action card title for selecting traces to add to a dataset"
                />
              }
              description={
                <FormattedMessage
                  defaultMessage='Select traces in the Traces tab, then use the "Actions" button to add them to a dataset'
                  description="Action card description for selecting traces to add to a dataset via actions button"
                />
              }
            />
          </Link>
        )}
      </div>
      <AddManuallyModal
        visible={addManuallyVisible}
        onCancel={() => setAddManuallyVisible(false)}
        datasetId={datasetId}
      />
      <AddViaCodeModal
        visible={addViaCodeVisible}
        onCancel={() => setAddViaCodeVisible(false)}
        datasetName={datasetName}
      />
    </div>
  );
};
