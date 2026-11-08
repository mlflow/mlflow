import { Button, PlayIcon, Tooltip } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { Link } from '../../../common/utils/RoutingUtils';
import Routes from '../../../experiment-tracking/routes';

interface TryInPlaygroundButtonProps {
  endpointName: string;
  /** The endpoint's usage-tracking experiment, if it has one. */
  experimentId?: string;
}

export const TryInPlaygroundButton = ({ endpointName, experimentId }: TryInPlaygroundButtonProps) => {
  const intl = useIntl();

  const label = (
    <FormattedMessage
      defaultMessage="Try in Playground"
      description="Gateway > Endpoint details > Button that opens the playground with this endpoint selected"
    />
  );

  // The playground is scoped to an experiment, and the endpoint's usage-tracking experiment is the
  // only one tied to it. Without it there is no experiment to open the playground in.
  if (!experimentId) {
    return (
      <Tooltip
        componentId="mlflow.gateway.edit-endpoint.try-in-playground-tooltip"
        content={intl.formatMessage({
          defaultMessage: 'Enable Usage Tracking in the Overview tab to try this endpoint in the Playground',
          description:
            'Tooltip on the disabled "Try in Playground" button explaining that usage tracking must be enabled first',
        })}
      >
        <Button componentId="mlflow.gateway.edit-endpoint.try-in-playground" icon={<PlayIcon />} disabled>
          {label}
        </Button>
      </Tooltip>
    );
  }

  return (
    <Link
      componentId="mlflow.gateway.edit-endpoint.try-in-playground-link"
      to={Routes.getExperimentPagePlaygroundRoute(experimentId, { endpointName })}
    >
      <Button componentId="mlflow.gateway.edit-endpoint.try-in-playground" icon={<PlayIcon />}>
        {label}
      </Button>
    </Link>
  );
};
