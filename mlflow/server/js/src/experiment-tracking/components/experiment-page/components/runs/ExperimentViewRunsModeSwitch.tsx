import { InfoPopover, LegacyTabs, LegacyTooltip, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { ExperimentPageViewState } from '../../models/ExperimentPageViewState';
import type { ExperimentViewRunsCompareMode } from '../../../../types';
import { PreviewBadge } from '@mlflow/mlflow/src/shared/building_blocks/PreviewBadge';
import { FeatureBadge } from '@mlflow/mlflow/src/shared/building_blocks/FeatureBadge';
import { getExperimentPageDefaultViewMode, useExperimentPageViewMode } from '../../hooks/useExperimentPageViewMode';
import { shouldUseRenamedUnifiedTracesTab } from '../../../../../common/utils/FeatureUtils';
import { MONITORING_BETA_EXPIRATION_DATE } from '../../../../constants';
import { useExperimentPageSearchFacets } from '../../hooks/useExperimentPageSearchFacets';

export interface ExperimentViewRunsModeSwitchProps {
  viewState?: ExperimentPageViewState;
  runsAreGrouped?: boolean;
  hideBorder?: boolean;
  explicitViewMode?: ExperimentViewRunsCompareMode;
  experimentId?: string;
}

/**
 * Allows switching between various modes of the experiment page view.
 * Handles legacy part of the mode switching, based on "compareRunsMode" query parameter.
 * Modern part of the mode switching is handled by <ExperimentViewRunsModeSwitchV2> which works using route params.
 */
export const ExperimentViewRunsModeSwitch = ({
  viewState,
  runsAreGrouped,
  hideBorder = true,
}: ExperimentViewRunsModeSwitchProps) => {
  const [, experimentIds] = useExperimentPageSearchFacets();
  const { theme } = useDesignSystemTheme();
  const [viewMode, setViewModeInURL] = useExperimentPageViewMode();
  const { classNamePrefix } = useDesignSystemTheme();
  const currentViewMode = viewMode || getExperimentPageDefaultViewMode();
  const validRunsTabModes = shouldUseRenamedUnifiedTracesTab() ? ['TABLE', 'CHART', 'ARTIFACT'] : ['TABLE', 'CHART'];
  const activeTab = validRunsTabModes.includes(currentViewMode) ? 'RUNS' : currentViewMode;

  // Extract experiment ID from the URL but only if it's a single experiment.
  // In case of multiple experiments (compare mode), the experiment ID is undefined.
  const singleExperimentId = experimentIds.length === 1 ? experimentIds[0] : undefined;

  return (
    <LegacyTabs
      dangerouslyAppendEmotionCSS={{
        [`.${classNamePrefix}-tabs-nav`]: {
          marginBottom: 0,
          '::before': {
            display: hideBorder ? 'none' : 'block',
          },
        },
      }}
      activeKey={activeTab}
      onChange={(tabKey) => {
        const newValue = tabKey as ExperimentViewRunsCompareMode | 'RUNS';

        if (activeTab === newValue) {
          return;
        }

        if (newValue === 'RUNS') {
          return setViewModeInURL('TABLE');
        }

        setViewModeInURL(newValue, singleExperimentId);
      }}
    >
      <LegacyTabs.TabPane
        tab={
          <span data-testid="experiment-runs-mode-switch-combined">
            <FormattedMessage
              defaultMessage="Runs"
              description="A button enabling combined runs table and charts mode on the experiment page"
            />
          </span>
        }
        key="RUNS"
      />
      {/* Display the "Models" tab if we have only one experiment and the feature is enabled. */}
      {singleExperimentId && (
        <LegacyTabs.TabPane
          key="MODELS"
          tab={
            <span data-testid="experiment-runs-mode-switch-models">
              <FormattedMessage
                defaultMessage="Models"
                description="A button navigating to logged models table on the experiment page"
              />
              <PreviewBadge />
            </span>
          }
        />
      )}
      <LegacyTabs.TabPane
        disabled={shouldUseRenamedUnifiedTracesTab() || runsAreGrouped}
        tab={
          <LegacyTooltip
            title={
              !shouldUseRenamedUnifiedTracesTab() && runsAreGrouped ? (
                <FormattedMessage
                  defaultMessage="Unavailable when runs are grouped"
                  description="Experiment page > view mode switch > evaluation mode disabled tooltip"
                />
              ) : undefined
            }
          >
            <span
              data-testid="experiment-runs-mode-switch-evaluation"
              css={
                shouldUseRenamedUnifiedTracesTab() && {
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: theme.spacing.xs,
                }
              }
            >
              <FormattedMessage
                defaultMessage="Evaluation"
                description="A button enabling compare runs (evaluation) mode on the experiment page"
              />
              {shouldUseRenamedUnifiedTracesTab() ? (
                <InfoPopover popoverProps={{ maxWidth: 350 }} iconProps={{ style: { marginRight: 0 } }}>
                  <FormattedMessage
                    defaultMessage='Accessing artifact evaluation by "Evaluation" tab is being discontinued. In order to use this feature, use <link>"Artifacts evaluation" mode in Runs tab</link> instead.'
                    description="A button enabling compare runs (evaluation) mode on the experiment page"
                    values={{
                      link: (children) =>
                        viewMode === 'ARTIFACT' ? (
                          children
                        ) : (
                          <Typography.Link
                            componentId="mlflow.experiment_page.evaluation_tab_migration_info_link"
                            onClick={() => setViewModeInURL('ARTIFACT', singleExperimentId)}
                          >
                            {children}
                          </Typography.Link>
                        ),
                    }}
                  />
                </InfoPopover>
              ) : (
                <PreviewBadge />
              )}
            </span>
          </LegacyTooltip>
        }
        key="ARTIFACT"
      />
      <LegacyTabs.TabPane
        tab={
          <span data-testid="experiment-runs-mode-switch-traces">
            <FormattedMessage
              defaultMessage="Traces"
              description="A button enabling traces mode on the experiment page"
            />
          </span>
        }
        key="TRACES"
      />
    </LegacyTabs>
  );
};
