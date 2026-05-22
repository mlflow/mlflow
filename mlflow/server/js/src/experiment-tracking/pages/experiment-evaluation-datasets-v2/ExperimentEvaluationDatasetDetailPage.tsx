import invariant from 'invariant';
import {
  Empty,
  ParagraphSkeleton,
  TitleSkeleton,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { Link, useParams } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import Routes from '@mlflow/mlflow/src/experiment-tracking/routes';
import { ExperimentPageTabName } from '@mlflow/mlflow/src/experiment-tracking/constants';
import { ExperimentEvaluationDatasetsPageWrapper } from '../experiment-evaluation-datasets/ExperimentEvaluationDatasetsPageWrapper';
import { useGetDatasetQuery } from './hooks/useDatasetsQueries';
import { DatasetDetailPageContent } from './components/DatasetDetailPageContent';
import { DatasetsBreadcrumbs } from './components/DatasetsBreadcrumbs';

const DatasetDetailLoadingSkeleton = () => {
  const { theme } = useDesignSystemTheme();
  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        flex: 1,
        minHeight: 0,
        // Outer wrappers (PageWrapper + ExperimentPageTabs) already contribute spacing
        // on the right (24px) and bottom (8px); only top and left need padding here.
        paddingTop: theme.spacing.md,
        paddingLeft: theme.spacing.md,
        gap: theme.spacing.md,
      }}
    >
      <ParagraphSkeleton seed="dataset-breadcrumbs" style={{ width: 220 }} />
      <TitleSkeleton seed="dataset-title" style={{ width: 320 }} loadingDescription="DatasetDetailPage" />
      <ParagraphSkeleton seed="dataset-toolbar" style={{ width: '100%' }} />
      {Array.from({ length: 6 }, (_, i) => (
        <ParagraphSkeleton key={`dataset-skeleton-row-${i}`} seed={`dataset-row-${i}`} style={{ width: '100%' }} />
      ))}
    </div>
  );
};

const ExperimentEvaluationDatasetDetailPageImpl = () => {
  const { experimentId, datasetId } = useParams();
  invariant(experimentId, 'Experiment ID must be defined');
  invariant(datasetId, 'Dataset ID must be defined');

  const { theme } = useDesignSystemTheme();

  // Fetch the dataset here (not inside the page/controller) so loading / error / 404 can be
  // handled at the page boundary. Downstream components receive a fully-loaded `Dataset` and
  // never have to render half-states for "dataset missing". `retry: false` short-circuits the
  // 404/deleted-dataset path — we surface a friendly error UI immediately rather than spending
  // ~7s on exponential-backoff retries.
  const datasetQuery = useGetDatasetQuery(datasetId, { retry: false });

  if (datasetQuery.isLoading) {
    return <DatasetDetailLoadingSkeleton />;
  }

  if (datasetQuery.error || !datasetQuery.data) {
    const listRoute = Routes.getExperimentPageTabRoute(experimentId, ExperimentPageTabName.Datasets);
    return (
      <div
        css={{
          display: 'flex',
          flexDirection: 'column',
          flex: 1,
          minHeight: 0,
          // Outer wrappers (PageWrapper + ExperimentPageTabs) already contribute spacing
          // on the right (24px) and bottom (8px); only top and left need padding here.
          paddingTop: theme.spacing.md,
          paddingLeft: theme.spacing.md,
          gap: theme.spacing.md,
        }}
      >
        <DatasetsBreadcrumbs experimentId={experimentId} />
        <div
          css={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            flex: 1,
            minHeight: 400,
            '& > div': {
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
            },
          }}
        >
          <Empty
            title={
              <FormattedMessage
                defaultMessage="Couldn't load this dataset"
                description="Heading shown when the V2 evaluation dataset detail page fails to fetch its dataset"
              />
            }
            description={
              <>
                <Typography.Paragraph>
                  <FormattedMessage
                    defaultMessage="The dataset may have been deleted, or you may not have access to it."
                    description="Body shown when the V2 evaluation dataset detail page fails to fetch its dataset"
                  />
                </Typography.Paragraph>
                <Link to={listRoute}>
                  <FormattedMessage
                    defaultMessage="Back to datasets"
                    description="Link returning the user to the V2 evaluation datasets list page after a fetch error"
                  />
                </Link>
              </>
            }
          />
        </div>
      </div>
    );
  }

  return <DatasetDetailPageContent experimentId={experimentId} datasetId={datasetId} dataset={datasetQuery.data} />;
};

export const ExperimentEvaluationDatasetDetailPage = () => {
  return (
    <ExperimentEvaluationDatasetsPageWrapper>
      <ExperimentEvaluationDatasetDetailPageImpl />
    </ExperimentEvaluationDatasetsPageWrapper>
  );
};

export default ExperimentEvaluationDatasetDetailPage;
