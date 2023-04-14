import {
  BarChartIcon,
  Button,
  ListBorderIcon,
  PopoverV2,
  SegmentedControlButton,
  SegmentedControlGroup,
  Typography,
} from '@databricks/design-system';
import React, { useState, useEffect, useCallback } from 'react';
import { FormattedMessage } from 'react-intl';
import { SearchExperimentRunsViewState } from '../../models/SearchExperimentRunsViewState';
import { useExperimentViewLocalStore } from '../../hooks/useExperimentViewLocalStore';

const COMPARE_RUNS_TOOLTIP_STORAGE_KEY = 'compareRunsTooltip';
const COMPARE_RUNS_TOOLTIP_STORAGE_ITEM = 'seenBefore';

export interface ExperimentViewRunsModeSwitchProps {
  isComparingRuns: boolean;
  setIsComparingRuns: (isComparingRuns: boolean) => void;
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
      <PopoverV2.Root open={isToolTipOpen}>
        <PopoverV2.Trigger asChild>
          <div css={{ position: 'absolute', inset: 0 }} />
        </PopoverV2.Trigger>
        <PopoverV2.Content align='start'>
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
          <PopoverV2.Arrow />
        </PopoverV2.Content>
      </PopoverV2.Root>
    </>
  );
};

/**
 * Allows switching between "list" and "compare runs" modes of experiment view
 */
export const ExperimentViewRunsModeSwitch = ({
  isComparingRuns,
  setIsComparingRuns,
  viewState,
}: ExperimentViewRunsModeSwitchProps) => {
  return (
    <SegmentedControlGroup
      value={isComparingRuns ? 'COMPARE' : 'LIST'}
      onChange={({ target: { value } }) => {
        setIsComparingRuns(value === 'COMPARE');
      }}
    >
      <SegmentedControlButton value='LIST' data-testid='experiment-runs-mode-switch-list'>
        <ListBorderIcon />{' '}
        <FormattedMessage
          defaultMessage='Table view'
          description='A button enabling table mode on the experiment page'
        />
      </SegmentedControlButton>
      <SegmentedControlButton value='COMPARE' data-testid='experiment-runs-mode-switch-compare'>
        <BarChartIcon />{' '}
        <FormattedMessage
          defaultMessage='Chart view'
          description='A button enabling compare runs (chart) mode on the experiment page'
        />
        <ChartViewButtonTooltip
          isComparingRuns={isComparingRuns}
          multipleRunsSelected={Object.keys(viewState.runsSelected).length > 1}
        />
      </SegmentedControlButton>
    </SegmentedControlGroup>
  );
};
