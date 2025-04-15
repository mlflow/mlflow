import { NavigationMenu, useDesignSystemTheme } from '@databricks/design-system';
import { PreviewBadge } from '@mlflow/mlflow/src/shared/building_blocks/PreviewBadge';
import { FeatureBadge } from '@mlflow/mlflow/src/shared/building_blocks/FeatureBadge';
import { FormattedMessage } from 'react-intl';
import { Link } from '../../../../../common/utils/RoutingUtils';
import Routes from '../../../../routes';
import type { ExperimentViewRunsCompareMode } from '../../../../types';
import { EXPERIMENT_PAGE_VIEW_MODE_QUERY_PARAM_KEY } from '../../hooks/useExperimentPageViewMode';
import {
  isExperimentEvalResultsMonitoringUIEnabled,
  isExperimentLoggedModelsUIEnabled,
  shouldEnableTracingUI,
} from '../../../../../common/utils/FeatureUtils';
import { ExperimentPageTabName } from '../../../../constants';

export interface ExperimentViewRunsModeSwitchProps {
  explicitViewMode?: ExperimentViewRunsCompareMode;
  experimentId?: string;
  activeTab: ExperimentPageTabName;
}

/**
 * Allows switching between modes of the experiment page view.
 * Based on new <NavigationMenu> component instead of the legacy <Tabs>.
 * Used only in logged models page (tab) for now, will be expanded to other tabs in the future.
 */
export const ExperimentViewRunsModeSwitchV2 = ({ experimentId = '', activeTab }: ExperimentViewRunsModeSwitchProps) => {
  const { theme } = useDesignSystemTheme();

  const getLinkToMode = (mode: ExperimentViewRunsCompareMode) =>
    [Routes.getExperimentPageRoute(experimentId), [EXPERIMENT_PAGE_VIEW_MODE_QUERY_PARAM_KEY, mode].join('=')].join(
      '?',
    );

  return (
    <NavigationMenu.Root>
      <NavigationMenu.List
        css={{
          // N/B: Styles from this component are customized in order to match the styles of
          // the legacy <Tabs> component, so the transition to the new component is seamless.
          marginBottom: 0,
          li: {
            lineHeight: theme.typography.lineHeightBase,
            marginRight: theme.spacing.lg,
            paddingTop: theme.spacing.xs * 1.5,
            paddingBottom: theme.spacing.xs * 1.5,
            '&>a': { padding: 0 },
            alignItems: 'center',
          },
          'li+li': {
            marginLeft: theme.spacing.xs * 0.5,
          },
        }}
      >
        {isExperimentLoggedModelsUIEnabled() && (
          <NavigationMenu.Item key="MODELS" active={activeTab === ExperimentPageTabName.Models}>
            <Link to={Routes.getExperimentPageTabRoute(experimentId, ExperimentPageTabName.Models)}>
              <FormattedMessage
                defaultMessage="Models"
                description="A button navigating to logged models table on the experiment page"
              />
              <PreviewBadge />
            </Link>
          </NavigationMenu.Item>
        )}
        <NavigationMenu.Item key="RUNS">
          <Link to={getLinkToMode('TABLE')}>
            <FormattedMessage
              defaultMessage="Runs"
              description="A button enabling combined runs table and charts mode on the experiment page"
            />
          </Link>
        </NavigationMenu.Item>
        <NavigationMenu.Item key="ARTIFACT">
          <Link to={getLinkToMode('ARTIFACT')}>
            <FormattedMessage
              defaultMessage="Evaluation"
              description="A button enabling compare runs (evaluation) mode on the experiment page"
            />
            <PreviewBadge />
          </Link>
        </NavigationMenu.Item>
        {shouldEnableTracingUI() && (
          <NavigationMenu.Item key="TRACES">
            <Link to={getLinkToMode('TRACES')}>
              <FormattedMessage
                defaultMessage="Traces"
                description="A button enabling traces mode on the experiment page"
              />
            </Link>
          </NavigationMenu.Item>
        )}
      </NavigationMenu.List>
    </NavigationMenu.Root>
  );
};
