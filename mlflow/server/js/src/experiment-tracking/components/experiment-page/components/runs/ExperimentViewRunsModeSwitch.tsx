import {
  ChartLineIcon,
  ListBorderIcon,
  SegmentedControlButton,
  SegmentedControlGroup,
} from '@databricks/design-system';

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
      <SegmentedControlButton value='LIST'>
        <ListBorderIcon />
      </SegmentedControlButton>
      <SegmentedControlButton value='COMPARE'>
        <ChartLineIcon />
      </SegmentedControlButton>
    </SegmentedControlGroup>
  );
};
