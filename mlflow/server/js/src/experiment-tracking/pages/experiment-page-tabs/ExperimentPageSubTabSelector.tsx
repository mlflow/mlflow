import { Spacer, Tabs, useDesignSystemTheme } from '@databricks/design-system';
import { ExperimentPageTabName } from '../../constants';
import { Link } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import { FormattedMessage } from '@databricks/i18n';
import Routes from '../../routes';

export const ExperimentPageSubTabSelector = ({
  experimentId,
  activeTab,
}: {
  experimentId: string;
  activeTab: ExperimentPageTabName;
}) => {
  const { theme } = useDesignSystemTheme();

  if (activeTab === ExperimentPageTabName.EvaluationRuns || activeTab === ExperimentPageTabName.Datasets) {
    return (
      <Tabs.Root
        componentId="mlflow.experiment.details.sub-tab-selector"
        value={activeTab}
        css={{ '& > div': { marginBottom: 0 } }}
      >
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
  }

  return (
    <>
      <Spacer size="sm" shrinks={false} />
      <div css={{ width: '100%', borderTop: `1px solid ${theme.colors.border}` }} />
    </>
  );
};
