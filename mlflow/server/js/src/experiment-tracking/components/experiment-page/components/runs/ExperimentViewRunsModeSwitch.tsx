import {
  BarChartIcon,
  Button,
  ListBorderIcon,
  Popover,
  SegmentedControlButton,
  SegmentedControlGroup,
  Tabs,
  Tag,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import React, { useState, useEffect, useCallback } from 'react';
import { FormattedMessage } from 'react-intl';
import { SearchExperimentRunsViewState } from '../../models/SearchExperimentRunsViewState';
import { useExperimentViewLocalStore } from '../../hooks/useExperimentViewLocalStore';
import { shouldEnableArtifactBasedEvaluation } from '../../../../../common/utils/FeatureUtils';
import type { ExperimentViewRunsCompareMode } from '../../../../types';

const COMPARE_RUNS_TOOLTIP_STORAGE_KEY = 'compareRunsTooltip';
const COMPARE_RUNS_TOOLTIP_STORAGE_ITEM = 'seenBefore';

export interface ExperimentViewRunsModeSwitchProps {
  compareRunsMode: ExperimentViewRunsCompareMode;
  setCompareRunsMode: (newCompareRunsMode: ExperimentViewRunsCompareMode) => void;
  viewState: SearchExperimentRunsViewState;
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
        <Popover.Content align='start'>
          <div css={{ maxWidth: '200px' }}>
            <Typography.Paragraph>
              <FormattedMessage
                defaultMessage='You can now switch to the chart view to compare runs'
                description='Tooltip to push users to use the chart view instead of compare view'
              />
            </Typography.Paragraph>
            <div css={{ textAlign: 'right' }}>
              <Button onClick={() => updateIsTooltipOpen(false)} type='primary'>
                <FormattedMessage
                  defaultMessage='Got it'
                  description='Button action text for chart switcher tooltip'
                />
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
  compareRunsMode,
  setCompareRunsMode,
  viewState,
}: ExperimentViewRunsModeSwitchProps) => {
  const isComparingRuns = compareRunsMode !== undefined;
  const { classNamePrefix } = useDesignSystemTheme();

  const activeTab = compareRunsMode || 'TABLE';

  return (
    <Tabs
      dangerouslyAppendEmotionCSS={{
        [`.${classNamePrefix}-tabs-nav`]: { marginBottom: 0, '::before': { border: 'none' } },
      }}
      activeKey={activeTab}
      onChange={(tabKey) =>
        setCompareRunsMode(
          (tabKey === 'TABLE' ? undefined : tabKey) as ExperimentViewRunsCompareMode,
        )
      }
    >
      <Tabs.TabPane
        tab={
          <span data-testid='experiment-runs-mode-switch-list'>
            <FormattedMessage
              defaultMessage='Table'
              description='A button enabling table mode on the experiment page'
            />
          </span>
        }
        key='TABLE'
      />
      <Tabs.TabPane
        tab={
          <>
            <span data-testid='experiment-runs-mode-switch-compare'>
              <FormattedMessage
                defaultMessage='Chart'
                description='A button enabling compare runs (chart) mode on the experiment page'
              />
            </span>
            <ChartViewButtonTooltip
              isComparingRuns={isComparingRuns}
              multipleRunsSelected={Object.keys(viewState.runsSelected).length > 1}
            />
          </>
        }
        key='CHART'
      />
      {shouldEnableArtifactBasedEvaluation() && (
        <Tabs.TabPane
          tab={
            <span data-testid='experiment-runs-mode-switch-evaluation'>
              <FormattedMessage
                defaultMessage='Evaluation'
                description='A button enabling compare runs (evaluation) mode on the experiment page'
              />{' '}
              <Tag style={{ marginLeft: '4px' }} color='turquoise'>
                <FormattedMessage
                  defaultMessage='Experimental'
                  description='Experimental badge shown for features which are experimental'
                />
              </Tag>
            </span>
          }
          key='ARTIFACT'
        />
      )}
    </Tabs>
  );
};
