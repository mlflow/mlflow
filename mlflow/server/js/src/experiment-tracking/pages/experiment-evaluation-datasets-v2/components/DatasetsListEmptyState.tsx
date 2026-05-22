import { Empty, SearchIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import datasetsEmptyImg from '@mlflow/mlflow/src/common/static/eval-datasets-empty.svg';
import type { Dataset } from '../hooks/useDatasetsQueries';
import { CreateDatasetButton } from '../../experiment-evaluation-datasets/components/CreateDatasetModal';
import { getEvalMonitorDocsLink } from '../utils/docsLinks';

interface NoResultsEmptyStateProps {
  searchQuery: string;
  onClearSearch: () => void;
}

/**
 * Shown when a search returns no datasets. Distinct from the "no datasets at all" state
 * so users don't think their first dataset got lost.
 */
export const DatasetsListNoResultsEmptyState = ({ searchQuery, onClearSearch }: NoResultsEmptyStateProps) => {
  return (
    <div
      css={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100%',
        minHeight: 320,
        width: '100%',
        '& > div': {
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          alignItems: 'center',
        },
      }}
    >
      <Empty
        image={<SearchIcon />}
        title={
          <FormattedMessage
            defaultMessage='No datasets match "{query}"'
            description="Title for the empty state when a search returns no datasets in the V2 evaluation datasets list"
            values={{ query: searchQuery }}
          />
        }
        description={
          <Typography.Link componentId="mlflow.eval-datasets-v2.list.clear-search-link" onClick={onClearSearch}>
            <FormattedMessage
              defaultMessage="Clear the search"
              description="Link text to clear the search filter on the V2 evaluation datasets list page"
            />
          </Typography.Link>
        }
      />
    </div>
  );
};

interface NoDatasetsEmptyStateProps {
  experimentId: string;
  onDatasetCreated: (dataset: Dataset) => void;
  refetch: () => void;
}

/**
 * "Create your first evaluation dataset" prompt — preserves the legacy copy and image.
 */
export const DatasetsListEmptyState = ({ experimentId, onDatasetCreated, refetch }: NoDatasetsEmptyStateProps) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100%',
        minHeight: 400,
        width: '100%',
        padding: theme.spacing.md,
      }}
    >
      <Typography.Title level={3}>
        <FormattedMessage
          defaultMessage="Create an evaluation dataset"
          description="Title for the V2 evaluation datasets empty state"
        />
      </Typography.Title>
      <Typography.Paragraph color="secondary" css={{ maxWidth: 600, textAlign: 'center' }}>
        <FormattedMessage
          defaultMessage="Create evaluation datasets in order to iteratively evaluate and improve your app. For example, build a dataset from production traces with negative feedback. {learnMoreLink}"
          description="Description for the V2 evaluation datasets empty state"
          values={{
            learnMoreLink: (
              <Typography.Link
                componentId="mlflow.eval-datasets-v2.list.learn-more-link"
                href={getEvalMonitorDocsLink('build-eval-dataset')}
                openInNewTab
              >
                <FormattedMessage
                  defaultMessage="Learn more"
                  description="Link text to learn more about evaluation datasets"
                />
              </Typography.Link>
            ),
          }}
        />
      </Typography.Paragraph>
      <img css={{ maxWidth: 'min(100%, 600px)' }} src={datasetsEmptyImg} alt="" />
      <CreateDatasetButton
        experimentId={experimentId}
        onSuccess={onDatasetCreated}
        refetch={async () => refetch()}
        buttonProps={{ css: { marginTop: theme.spacing.md } }}
      />
    </div>
  );
};
