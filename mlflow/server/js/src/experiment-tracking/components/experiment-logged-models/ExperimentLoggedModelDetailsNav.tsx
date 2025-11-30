import { NavigationMenu } from '@databricks/design-system';
import { Link } from '../../../common/utils/RoutingUtils';
import Routes from '../../routes';

import { FormattedMessage } from 'react-intl';

export const ExperimentLoggedModelDetailsNav = ({
  experimentId,
  modelId,
  activeTabName,
}: {
  experimentId: string;
  modelId: string;
  activeTabName?: string;
}) => {
  return (
    <NavigationMenu.Root>
      <NavigationMenu.List>
        <NavigationMenu.Item key="overview" active={!activeTabName}>
          <Link to={Routes.getExperimentLoggedModelDetailsPageRoute(experimentId, modelId)}>
            <FormattedMessage
              defaultMessage="Overview"
              description="Label for the overview tab on the logged model details page"
            />
          </Link>
        </NavigationMenu.Item>
        {/* TODO: Implement when available */}
        {/* <NavigationMenu.Item key="evaluations" active={activeTabName === 'evaluations'}>
          <Link to={Routes.getExperimentLoggedModelDetailsPageRoute(experimentId, modelId, 'evaluations')}>
            <FormattedMessage
              defaultMessage="Evaluations"
              description="Label for the evaluations tab on the logged model details page"
            />
          </Link>
        </NavigationMenu.Item> */}
        <NavigationMenu.Item key="traces" active={activeTabName === 'traces'}>
          <Link to={Routes.getExperimentLoggedModelDetailsPageRoute(experimentId, modelId, 'traces')}>
            <FormattedMessage
              defaultMessage="Traces"
              description="Label for the traces tab on the logged model details page"
            />
          </Link>
        </NavigationMenu.Item>
        <NavigationMenu.Item key="artifacts" active={activeTabName === 'artifacts'}>
          <Link to={Routes.getExperimentLoggedModelDetailsPageRoute(experimentId, modelId, 'artifacts')}>
            <FormattedMessage
              defaultMessage="Artifacts"
              description="Label for the artifacts tab on the logged model details page"
            />
          </Link>
        </NavigationMenu.Item>
      </NavigationMenu.List>
    </NavigationMenu.Root>
  );
};
