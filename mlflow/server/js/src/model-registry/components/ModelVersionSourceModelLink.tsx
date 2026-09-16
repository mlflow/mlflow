import { ParagraphSkeleton, Typography } from '@databricks/design-system';
import { useIntl } from 'react-intl';
import { Link } from '../../common/utils/RoutingUtils';
import { useGetLoggedModelQuery } from '../../experiment-tracking/hooks/logged-models/useGetLoggedModelQuery';
import Routes from '../../experiment-tracking/routes';

/**
 * Link to the logged model that a model version was registered from.
 * Falls back to the plain model ID when the logged model cannot be fetched or has no experiment.
 */
export const ModelVersionSourceModelLink = ({ loggedModelId }: { loggedModelId: string }) => {
  const intl = useIntl();
  const { data, isLoading, error } = useGetLoggedModelQuery({ loggedModelId });

  if (isLoading) {
    return (
      <ParagraphSkeleton
        css={{ width: 80 }}
        data-testid="source-model-loading"
        label={intl.formatMessage({
          defaultMessage: 'Loading source model',
          description: 'Screen reader label shown while the source logged model of a model version is loading',
        })}
      />
    );
  }

  const experimentId = data?.info?.experiment_id;
  if (error || !experimentId) {
    return <Typography.Text data-testid="source-model-id">{loggedModelId}</Typography.Text>;
  }

  return (
    <Link
      componentId="mlflow.model_registry.version_view.source_model_link"
      data-testid="source-model-link"
      to={Routes.getExperimentLoggedModelDetailsPageRoute(experimentId, loggedModelId)}
    >
      {data?.info?.name || loggedModelId}
    </Link>
  );
};
