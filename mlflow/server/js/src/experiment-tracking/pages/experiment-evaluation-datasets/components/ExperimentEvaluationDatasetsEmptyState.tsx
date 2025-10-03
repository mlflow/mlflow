import { Empty, Typography, useDesignSystemTheme } from '@databricks/design-system';
import datasetsEmptyImg from '@mlflow/mlflow/src/common/static/eval-datasets-empty.svg';
import { FormattedMessage } from 'react-intl';
import { CreateEvaluationDatasetButton } from './CreateEvaluationDatasetButton';

export const ExperimentEvaluationDatasetsEmptyState = ({ experimentId }: { experimentId: string }) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        display: 'flex',
        flex: 1,
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
      }}
    >
      <Typography.Title level={3} color="secondary">
        <FormattedMessage
          defaultMessage="Create an evaluation dataset"
          description="Evaluation datasets empty state title"
        />
      </Typography.Title>
      <Typography.Paragraph color="secondary" css={{ maxWidth: 600 }}>
        <FormattedMessage
          defaultMessage="Create evaluation datasets in order to iteratively evaluate and improve your app. For example, build a dataset from production traces with negative feedback. {learnMoreLink}"
          description="Description for a quickstart guide on MLflow evaluation datasets"
          values={{
            learnMoreLink: (
              <Typography.Link
                componentId="mlflow.eval-datasets.learn-more-link"
                href="https://mlflow.org/docs/latest/genai/datasets/"
                openInNewTab
              >
                {/* eslint-disable-next-line formatjs/enforce-description */}
                <FormattedMessage defaultMessage="Learn more" />
              </Typography.Link>
            ),
          }}
        />
      </Typography.Paragraph>
      <img css={{ marginBottom: theme.spacing.md }} src={datasetsEmptyImg} alt="No datasets found" />
      <CreateEvaluationDatasetButton experimentId={experimentId} />
    </div>
  );
};
