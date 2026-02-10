import { ArrowLeftIcon, BeakerIcon, Spinner, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { useGetExperimentQuery } from '../../experiment-tracking/hooks/useExperimentQuery';
import type { Location } from '../utils/RoutingUtils';
import { matchPath } from '../utils/RoutingUtils';
import ExperimentTrackingRoutes from '../../experiment-tracking/routes';
import { MlflowSidebarLink } from './MlflowSidebarLink';
import { getExperimentKindForWorkflowType } from '../../experiment-tracking/utils/ExperimentKindUtils';
import {
  ExperimentPageSideNavSectionKey,
  getExperimentPageSideNavSectionLabel,
  useExperimentPageSideNavConfig,
} from '../../experiment-tracking/pages/experiment-page-tabs/side-nav/constants';
import { useExperimentEvaluationRunsData } from '../../experiment-tracking/components/experiment-page/hooks/useExperimentEvaluationRunsData';
import { WorkflowType } from '../contexts/WorkflowTypeContext';
import { useGetExperimentPageActiveTabByRoute } from '../../experiment-tracking/components/experiment-page/hooks/useGetExperimentPageActiveTabByRoute';
import { ExperimentPageTabName } from '../../experiment-tracking/constants';

const isExperimentsActive = (location: Location) =>
  Boolean(
    matchPath({ path: '/experiments', end: true }, location.pathname) ||
    matchPath('/compare-experiments/*', location.pathname),
  );

export const MlflowSidebarExperimentItems = ({
  collapsed,
  experimentId,
  workflowType,
  onBackClick,
}: {
  collapsed: boolean;
  experimentId: string | undefined;
  workflowType: WorkflowType;
  onBackClick?: () => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const { data: experiment, loading } = useGetExperimentQuery({ experimentId });
  const { trainingRuns } = useExperimentEvaluationRunsData({
    experimentId: experimentId || '',
    enabled: Boolean(experimentId) && workflowType === WorkflowType.GENAI,
    filter: '', // not important in this case, we show the runs tab if there are any training runs
  });
  const config = useExperimentPageSideNavConfig({
    experimentKind: getExperimentKindForWorkflowType(workflowType),
    hasTrainingRuns: (trainingRuns?.length ?? 0) > 0,
  });
  const { tabName: activeTabByRoute } = useGetExperimentPageActiveTabByRoute();

  return (
    <>
      <div
        css={{
          borderTop: `1px solid ${theme.colors.actionDefaultBorderDefault}`,
          width: '100%',
          paddingBottom: theme.spacing.sm,
          marginTop: theme.spacing.xs,
        }}
      />
      <MlflowSidebarLink
        css={{ border: `1px solid ${theme.colors.actionDefaultBorderDefault}`, marginBottom: theme.spacing.sm }}
        to={ExperimentTrackingRoutes.experimentsObservatoryRoute}
        componentId="mlflow.experiment-sidebar.back-button"
        isActive={isExperimentsActive}
        onClick={onBackClick}
        icon={<ArrowLeftIcon />}
        collapsed={collapsed}
      >
        <BeakerIcon />
        {loading ? <Spinner /> : <Typography.Text ellipsis>{experiment?.name}</Typography.Text>}
      </MlflowSidebarLink>
      {Object.entries(config).map(([sectionKey, items]) => (
        <>
          {sectionKey !== 'top-level' && collapsed ? (
            <div
              css={{
                paddingLeft: theme.spacing.lg,
                marginTop: theme.spacing.sm,
                marginBottom: theme.spacing.sm,
                borderBottom: `1px solid ${theme.colors.actionDefaultBorderDefault}`,
              }}
            />
          ) : (
            <li css={{ paddingLeft: theme.spacing.lg, marginTop: theme.spacing.sm, marginBottom: theme.spacing.sm }}>
              <Typography.Text size="sm" color="secondary">
                {getExperimentPageSideNavSectionLabel(sectionKey as ExperimentPageSideNavSectionKey, items)}
              </Typography.Text>
            </li>
          )}
          {items.map((item) => {
            const isActive = () => {
              if (item.tabName === ExperimentPageTabName.ChatSessions) {
                return (
                  activeTabByRoute === ExperimentPageTabName.ChatSessions ||
                  activeTabByRoute === ExperimentPageTabName.SingleChatSession
                );
              }
              return activeTabByRoute === item.tabName;
            };
            return (
              <MlflowSidebarLink
                css={{ paddingLeft: collapsed ? undefined : theme.spacing.lg }}
                key={item.componentId}
                to={ExperimentTrackingRoutes.getExperimentPageTabRoute(experimentId ?? '', item.tabName)}
                componentId={item.componentId}
                isActive={isActive}
                collapsed={collapsed}
                icon={item.icon}
              >
                {item.label}
              </MlflowSidebarLink>
            );
          })}
        </>
      ))}
      <div
        css={{
          borderBottom: `1px solid ${theme.colors.actionDefaultBorderDefault}`,
          width: '100%',
          paddingTop: theme.spacing.sm,
          marginBottom: theme.spacing.xs,
        }}
      />
    </>
  );
};
