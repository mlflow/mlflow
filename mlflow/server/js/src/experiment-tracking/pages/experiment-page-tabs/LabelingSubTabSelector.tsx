import { Spacer, Tabs, useDesignSystemTheme } from '@databricks/design-system';
import { ExperimentPageTabName } from '../../constants';
import Routes from '@mlflow/mlflow/src/experiment-tracking/routes';
import { Link } from '../../../common/utils/RoutingUtils';
import { FormattedMessage } from 'react-intl';

export const LabelingSubTabSelector = ({
  experimentId,
  activeTab,
}: {
  experimentId: string;
  activeTab: ExperimentPageTabName;
}) => {
  const { theme } = useDesignSystemTheme();
  // BEGIN-EDGE
  return (
    <Tabs.Root
      componentId="mlflow.labeling-sub-tab-selector"
      value={activeTab}
      css={{ '& > div': { marginBottom: 0 } }}
    >
      <Tabs.List>
        <Tabs.Trigger value={ExperimentPageTabName.LabelingSessions}>
          <Link
            css={{ color: theme.colors.textPrimary }}
            to={Routes.getExperimentPageTabRoute(experimentId, ExperimentPageTabName.LabelingSessions)}
          >
            <FormattedMessage
              defaultMessage="Sessions"
              description="Label for the labeling sessions sub-tab in the MLflow experiment navbar"
            />
          </Link>
        </Tabs.Trigger>
        <Tabs.Trigger value={ExperimentPageTabName.LabelingSchemas}>
          <Link
            css={{ color: theme.colors.textPrimary }}
            to={Routes.getExperimentPageTabRoute(experimentId, ExperimentPageTabName.LabelingSchemas)}
          >
            <FormattedMessage
              defaultMessage="Schemas"
              description="Label for the labeling schemas sub-tab in the MLflow experiment navbar"
            />
          </Link>
        </Tabs.Trigger>
      </Tabs.List>
    </Tabs.Root>
  );
};
