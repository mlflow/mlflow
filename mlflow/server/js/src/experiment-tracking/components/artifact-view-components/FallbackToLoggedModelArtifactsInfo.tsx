import { Alert, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { Link } from '../../../common/utils/RoutingUtils';
import { useGetLoggedModelQuery } from '../../hooks/logged-models/useGetLoggedModelQuery';
import Routes from '../../routes';

export const FallbackToLoggedModelArtifactsInfo = ({ loggedModelId }: { loggedModelId: string }) => {
  const { data } = useGetLoggedModelQuery({ loggedModelId });
  const experimentId = data?.info?.experiment_id;
  const { theme } = useDesignSystemTheme();
  return (
    <Alert
      type="info"
      componentId="mlflow.artifacts.logged_model_fallback_info"
      message={
        <FormattedMessage
          defaultMessage="You're viewing artifacts assigned to a <link>logged model</link> associated with this run."
          description="Alert message to inform the user that they are viewing artifacts assigned to a logged model associated with this run."
          values={{
            link: (chunks) =>
              experimentId ? (
                <Link to={Routes.getExperimentLoggedModelDetailsPage(experimentId, loggedModelId)}>{chunks}</Link>
              ) : (
                <>{chunks}</>
              ),
          }}
        />
      }
      closable={false}
      css={{ margin: theme.spacing.xs }}
    />
  );
};
