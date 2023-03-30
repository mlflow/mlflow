import {
  BarChartIcon,
  ListBorderIcon,
  SegmentedControlButton,
  SegmentedControlGroup,
} from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

export interface ExperimentViewRunsModeSwitchProps {
  isComparingRuns: boolean;
  setIsComparingRuns: (isComparingRuns: boolean) => void;
}

/**
 * Allows switching between "list" and "compare runs" modes of experiment view
 */
export const ExperimentViewRunsModeSwitch = ({
  isComparingRuns,
  setIsComparingRuns,
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
      </SegmentedControlButton>
    </SegmentedControlGroup>
  );
};
