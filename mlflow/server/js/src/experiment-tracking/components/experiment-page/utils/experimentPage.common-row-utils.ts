import { isUndefined } from 'lodash';
import { shouldUseRunRowsVisibilityMap } from '../../../../common/utils/FeatureUtils';
import { RUNS_VISIBILITY_MODE } from '../models/ExperimentPageUIState';

// Utility function that determines if a particular table run should be hidden,
// based on the selected mode, position on the list and current state of manually hidden runs array.
export const determineIfRowIsHidden = (
  runsHiddenMode: RUNS_VISIBILITY_MODE,
  /**
   * @deprecated Use "runsVisibilityMap" field instead which has better control over visibility
   */
  runsHidden: string[],
  runUuid: string,
  rowIndex: number,
  runsVisibilityMap: Record<string, boolean>,
  runStatus?: string,
) => {
  // If using rows visibility map, we should always use it to determine visibility
  if (shouldUseRunRowsVisibilityMap()) {
    if (!isUndefined(runsVisibilityMap[runUuid])) {
      return !runsVisibilityMap[runUuid];
    }
  } else if (runsHiddenMode === RUNS_VISIBILITY_MODE.CUSTOM) {
    /**
     * TODO: clean up runsHidden after ramping up runsVisibilityMap
     */
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
  if (runsHiddenMode === RUNS_VISIBILITY_MODE.HIDE_FINISHED_RUNS) {
    return ['FINISHED', 'FAILED', 'KILLED'].includes(runStatus ?? '');
  }
  return false;
};
