import { Link } from '../../common/utils/RoutingUtils';
import { useGetLoggedModelQuery } from '../../experiment-tracking/hooks/logged-models/useGetLoggedModelQuery';
import Routes from '../../experiment-tracking/routes';

/**
 * Link to the logged model that a model version was registered from; plain model ID until it resolves.
 */
export const ModelVersionSourceModelLink = ({ loggedModelId }: { loggedModelId: string }) => {
  const { data } = useGetLoggedModelQuery({ loggedModelId });
  const experimentId = data?.info?.experiment_id;
  if (!experimentId) {
    return <span data-testid="source-model-id">{loggedModelId}</span>;
  }
  return (
    <Link
      componentId="mlflow.model_registry.version_view.source_model_link"
      data-testid="source-model-link"
      to={Routes.getExperimentLoggedModelDetailsPageRoute(experimentId, loggedModelId)}
    >
      {data?.info?.name ?? loggedModelId}
    </Link>
  );
};
