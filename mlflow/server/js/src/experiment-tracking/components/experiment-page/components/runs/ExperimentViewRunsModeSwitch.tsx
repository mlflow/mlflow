import { Button, Popover, Tabs, Tag, LegacyTooltip, Typography, useDesignSystemTheme } from '@databricks/design-system';
import React, { useState, useEffect, useCallback } from 'react';
import { FormattedMessage } from 'react-intl';
import { ExperimentPageViewState } from '../../models/ExperimentPageViewState';
import { useExperimentViewLocalStore } from '../../hooks/useExperimentViewLocalStore';
import type { ExperimentViewRunsCompareMode } from '../../../../types';
import { PreviewBadge } from '@mlflow/mlflow/src/shared/building_blocks/PreviewBadge';
import { getExperimentPageDefaultViewMode, useExperimentPageViewMode } from '../../hooks/useExperimentPageViewMode';
import { shouldEnableTracingUI } from '../../../../../common/utils/FeatureUtils';
import { useShouldShowCombinedRunsTab } from '../../hooks/useShouldShowCombinedRunsTab';

const COMPARE_RUNS_TOOLTIP_STORAGE_KEY = 'compareRunsTooltip';
const COMPARE_RUNS_TOOLTIP_STORAGE_ITEM = 'seenBefore';

export interface ExperimentViewRunsModeSwitchProps {
  viewState?: ExperimentPageViewState;
  runsAreGrouped?: boolean;
  hideBorder?: boolean;
}

const ChartViewButtonTooltip: React.FC<{
  isTableMode: boolean;
  multipleRunsSelected: boolean;
}> = ({ multipleRunsSelected, isTableMode }) => {
  const seenTooltipStore = useExperimentViewLocalStore(COMPARE_RUNS_TOOLTIP_STORAGE_KEY);
  const [isToolTipOpen, setToolTipOpen] = useState(
    multipleRunsSelected && !seenTooltipStore.getItem(COMPARE_RUNS_TOOLTIP_STORAGE_ITEM),
  );

  useEffect(() => {
    const hasSeenTooltipBefore = seenTooltipStore.getItem(COMPARE_RUNS_TOOLTIP_STORAGE_ITEM);
    if (multipleRunsSelected && isTableMode && !hasSeenTooltipBefore) {
      setToolTipOpen(true);
    } else {
      setToolTipOpen(false);
    }
  }, [multipleRunsSelected, isTableMode, seenTooltipStore]);

  const updateIsTooltipOpen = useCallback(
    (isOpen) => {
      setToolTipOpen(isOpen);
      seenTooltipStore.setItem(COMPARE_RUNS_TOOLTIP_STORAGE_ITEM, true);
    },
    [setToolTipOpen, seenTooltipStore],
  );

  return (
    <>
      <Popover.Root open={isToolTipOpen}>
        <Popover.Trigger asChild>
          <div css={{ position: 'absolute', inset: 0 }} />
        </Popover.Trigger>
        <Popover.Content align="start">
          <div css={{ maxWidth: '200px' }}>
            <Typography.Paragraph>
              <FormattedMessage
                defaultMessage="You can now switch to the chart view to compare runs"
                description="Tooltip to push users to use the chart view instead of compare view"
              />
            </Typography.Paragraph>
            <div css={{ textAlign: 'right' }}>
              <Button
                componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunsmodeswitch.tsx_65"
                onClick={() => updateIsTooltipOpen(false)}
                type="primary"
              >
                <FormattedMessage defaultMessage="Got it" description="Button action text for chart switcher tooltip" />
              </Button>
            </div>
          </div>
          <Popover.Arrow />
        </Popover.Content>
      </Popover.Root>
    </>
  );
};

/**
 * Allows switching between "table", "chart", "evaluation" and "traces" modes of experiment view
 */
export const ExperimentViewRunsModeSwitch = ({
  viewState,
  runsAreGrouped,
  hideBorder = true,
}: ExperimentViewRunsModeSwitchProps) => {
  const [viewMode, setViewModeInURL] = useExperimentPageViewMode();
  const { classNamePrefix } = useDesignSystemTheme();
  const currentViewMode = viewMode || getExperimentPageDefaultViewMode();
  const showCombinedRuns = useShouldShowCombinedRunsTab();
  const activeTab = showCombinedRuns && ['TABLE', 'CHART'].includes(currentViewMode) ? 'RUNS' : currentViewMode;

  return (
    <Tabs
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

        setViewModeInURL(newValue);
      }}
    >
      {showCombinedRuns ? (
        <Tabs.TabPane
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
      ) : (
        <>
          <Tabs.TabPane
            tab={
              <span data-testid="experiment-runs-mode-switch-list">
                <FormattedMessage
                  defaultMessage="Table"
                  description="A button enabling table mode on the experiment page"
                />
              </span>
            }
            key="TABLE"
          />
          <Tabs.TabPane
            tab={
              <>
                <span data-testid="experiment-runs-mode-switch-compare">
                  <FormattedMessage
                    defaultMessage="Chart"
                    description="A button enabling compare runs (chart) mode on the experiment page"
                  />
                </span>
                <ChartViewButtonTooltip
                  isTableMode={viewMode === 'TABLE'}
                  multipleRunsSelected={viewState ? Object.keys(viewState.runsSelected).length > 1 : false}
                />
              </>
            }
            key="CHART"
          />
        </>
      )}

      <Tabs.TabPane
        disabled={runsAreGrouped}
        tab={
          <LegacyTooltip
            title={
              runsAreGrouped ? (
                <FormattedMessage
                  defaultMessage="Unavailable when runs are grouped"
                  description="Experiment page > view mode switch > evaluation mode disabled tooltip"
                />
              ) : undefined
            }
          >
            <span data-testid="experiment-runs-mode-switch-evaluation">
              <FormattedMessage
                defaultMessage="Evaluation"
                description="A button enabling compare runs (evaluation) mode on the experiment page"
              />
              <PreviewBadge />
            </span>
          </LegacyTooltip>
        }
        key="ARTIFACT"
      />
      {shouldEnableTracingUI() && (
        <Tabs.TabPane
          tab={
            <span data-testid="experiment-runs-mode-switch-traces">
              <FormattedMessage
                defaultMessage="Traces"
                description="A button enabling traces mode on the experiment page"
              />
              <PreviewBadge />
            </span>
          }
          key="TRACES"
        />
      )}
    </Tabs>
  );
};
