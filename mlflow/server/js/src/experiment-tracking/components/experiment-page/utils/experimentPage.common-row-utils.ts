import { RUNS_VISIBILITY_MODE } from '../models/ExperimentPageUIState';

// Utility function that determines if a particular table run should be hidden,
// based on the selected mode, position on the list and current state of manually hidden runs array.
export const determineIfRowIsHidden = (
  runsHiddenMode: RUNS_VISIBILITY_MODE,
  runsHidden: string[],
  runUuid: string,
  rowIndex: number,
) => {
  if (runsHiddenMode === RUNS_VISIBILITY_MODE.CUSTOM) {
    return runsHidden.includes(runUuid);
  }
  if (runsHiddenMode === RUNS_VISIBILITY_MODE.HIDEALL) {
    return true;
  }
  if (runsHiddenMode === RUNS_VISIBILITY_MODE.FIRST_10_RUNS) {
    return rowIndex >= 10;
  }
  if (runsHiddenMode === RUNS_VISIBILITY_MODE.FIRST_20_RUNS) {
    return rowIndex >= 20;
  }

  return false;
};
