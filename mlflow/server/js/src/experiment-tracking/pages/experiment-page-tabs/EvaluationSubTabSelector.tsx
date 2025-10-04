import { Tabs, useDesignSystemTheme } from '@databricks/design-system';
import { ExperimentPageTabName } from '../../constants';
import Routes from '@mlflow/mlflow/src/experiment-tracking/routes';
import { Link } from '../../../common/utils/RoutingUtils';
import { FormattedMessage } from 'react-intl';

export const EvaluationSubTabSelector = ({
  experimentId,
  activeTab,
}: {
  experimentId: string;
  activeTab: ExperimentPageTabName;
}) => {
  const { theme } = useDesignSystemTheme();
  return (
    <Tabs.Root componentId={experimentId} value={activeTab} css={{ '& > div': { marginBottom: 0 } }}>
      <Tabs.List>
        <Tabs.Trigger value={ExperimentPageTabName.EvaluationRuns}>
          <Link
            css={{ color: theme.colors.textPrimary }}
            to={Routes.getExperimentPageTabRoute(experimentId, ExperimentPageTabName.EvaluationRuns)}
          >
            <FormattedMessage
              defaultMessage="Runs"
              description="Label for the evaluation runs sub-tab in the MLflow experiment navbar"
            />
          </Link>
        </Tabs.Trigger>
        <Tabs.Trigger value={ExperimentPageTabName.Datasets}>
          <Link
            css={{ color: theme.colors.textPrimary }}
            to={Routes.getExperimentPageTabRoute(experimentId, ExperimentPageTabName.Datasets)}
          >
            <FormattedMessage
              defaultMessage="Datasets"
              description="Label for the evaluation datasets sub-tab in the MLflow experiment navbar"
            />
          </Link>
        </Tabs.Trigger>
      </Tabs.List>
    </Tabs.Root>
  );
};
