import { Button, Popover, Tabs, Tag, Tooltip, Typography, useDesignSystemTheme } from '@databricks/design-system';
import React, { useState, useEffect, useCallback } from 'react';
import { FormattedMessage } from 'react-intl';
import { SearchExperimentRunsViewState } from '../../models/SearchExperimentRunsViewState';
import { useExperimentViewLocalStore } from '../../hooks/useExperimentViewLocalStore';
import { shouldEnableShareExperimentViewByTags } from '../../../../../common/utils/FeatureUtils';
import type { ExperimentViewRunsCompareMode } from '../../../../types';
import { PreviewBadge } from 'shared/building_blocks/PreviewBadge';
import { useExperimentPageViewMode } from '../../hooks/useExperimentPageViewMode';

const COMPARE_RUNS_TOOLTIP_STORAGE_KEY = 'compareRunsTooltip';
const COMPARE_RUNS_TOOLTIP_STORAGE_ITEM = 'seenBefore';

export interface ExperimentViewRunsModeSwitchProps {
  compareRunsMode: ExperimentViewRunsCompareMode;
  setCompareRunsMode: (newCompareRunsMode: ExperimentViewRunsCompareMode) => void;
  viewState: SearchExperimentRunsViewState;
  runsAreGrouped?: boolean;
}

const ChartViewButtonTooltip: React.FC<{
  isComparingRuns: boolean;
  multipleRunsSelected: boolean;
}> = ({ multipleRunsSelected, isComparingRuns }) => {
  const seenTooltipStore = useExperimentViewLocalStore(COMPARE_RUNS_TOOLTIP_STORAGE_KEY);
  const [isToolTipOpen, setToolTipOpen] = useState(
    multipleRunsSelected && !seenTooltipStore.getItem(COMPARE_RUNS_TOOLTIP_STORAGE_ITEM),
  );

  useEffect(() => {
    const hasSeenTooltipBefore = seenTooltipStore.getItem(COMPARE_RUNS_TOOLTIP_STORAGE_ITEM);
    if (multipleRunsSelected && !isComparingRuns && !hasSeenTooltipBefore) {
      setToolTipOpen(true);
    } else {
      setToolTipOpen(false);
    }
  }, [multipleRunsSelected, isComparingRuns, seenTooltipStore]);

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
 * Allows switching between "table", "chart" and "evaluation" modes of experiment view
 */
export const ExperimentViewRunsModeSwitch = ({
  compareRunsMode: compareRunsModeFromProps,
  setCompareRunsMode,
  viewState,
  runsAreGrouped,
}: ExperimentViewRunsModeSwitchProps) => {
  const usingNewViewStateModel = shouldEnableShareExperimentViewByTags();
  const [viewModeFromURL, setViewModeInURL] = useExperimentPageViewMode();

  // In the new view state model, use the view mode serialized in the URL
  const compareRunsMode = usingNewViewStateModel ? compareRunsModeFromProps : viewModeFromURL;

  const isComparingRuns = compareRunsMode !== undefined;
  const { classNamePrefix } = useDesignSystemTheme();

  const activeTab = (usingNewViewStateModel ? viewModeFromURL : compareRunsMode) || 'TABLE';

  return (
    <Tabs
      dangerouslyAppendEmotionCSS={{
        [`.${classNamePrefix}-tabs-nav`]: {
          marginBottom: 0,
          '::before': { border: 'none' },
        },
      }}
      activeKey={activeTab}
      onChange={(tabKey) => {
        const newValue = (tabKey === 'TABLE' ? undefined : tabKey) as ExperimentViewRunsCompareMode;
        if (usingNewViewStateModel) {
          setViewModeInURL(newValue);
        } else {
          setCompareRunsMode(newValue);
        }
      }}
    >
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
              isComparingRuns={isComparingRuns}
              multipleRunsSelected={Object.keys(viewState.runsSelected).length > 1}
            />
          </>
        }
        key="CHART"
      />

      <Tabs.TabPane
        disabled={runsAreGrouped}
        tab={
          <Tooltip
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
          </Tooltip>
        }
        key="ARTIFACT"
      />
    </Tabs>
  );
};
