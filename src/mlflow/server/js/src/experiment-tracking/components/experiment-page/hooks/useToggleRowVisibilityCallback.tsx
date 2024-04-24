import { useCallback, useRef } from 'react';
import { useUpdateExperimentViewUIState } from '../contexts/ExperimentPageUIStateContext';
import { RUNS_VISIBILITY_MODE } from '../models/ExperimentPageUIStateV2';
import type { RunRowType } from '../utils/experimentPage.row-types';

export const useToggleRowVisibilityCallback = (tableRows: RunRowType[]) => {
  const updateUIState = useUpdateExperimentViewUIState();

  // We're going to use current state of the table rows to determine which rows are hidden.
  // Since we're interested only in the latest data, we avoid using state here to avoid unnecessary re-renders.
  const immediateTableRows = useRef(tableRows);
  immediateTableRows.current = tableRows;

  const toggleRowVisibility = useCallback(
    (mode: RUNS_VISIBILITY_MODE, groupOrRunUuid?: string) => {
      updateUIState((currentUIState) => {
        if (mode === RUNS_VISIBILITY_MODE.SHOWALL) {
          // Case #1: Showing all runs
          return {
            ...currentUIState,
            runsHiddenMode: RUNS_VISIBILITY_MODE.SHOWALL,
            runsHidden: [],
          };
        } else if (mode === RUNS_VISIBILITY_MODE.HIDEALL) {
          // Case #2: Hiding all runs
          return {
            ...currentUIState,
            runsHiddenMode: RUNS_VISIBILITY_MODE.HIDEALL,
            runsHidden: [],
          };
        } else if (mode === RUNS_VISIBILITY_MODE.FIRST_10_RUNS) {
          // Case #3: Showing only first 10 runs
          return {
            ...currentUIState,
            runsHiddenMode: RUNS_VISIBILITY_MODE.FIRST_10_RUNS,
            runsHidden: [],
          };
        } else if (mode === RUNS_VISIBILITY_MODE.FIRST_20_RUNS) {
          // Case #4: Showing only first 20 runs
          return {
            ...currentUIState,
            runsHiddenMode: RUNS_VISIBILITY_MODE.FIRST_20_RUNS,
            runsHidden: [],
          };
        }

        // Case #5: Custom visibility mode enabled by manually toggling visibility of a run or a group
        if (groupOrRunUuid) {
          // Determine which runs are hidden at the moment
          const currentlyHiddenRows = immediateTableRows.current
            .filter(({ hidden }) => hidden)
            .map(({ groupParentInfo, rowUuid, runUuid }) => (groupParentInfo ? rowUuid : runUuid));

          // Toggle visibility of a run/group by either adding or removing from the array
          const newHiddenRows = currentlyHiddenRows.includes(groupOrRunUuid)
            ? currentlyHiddenRows.filter((currentGroupOrRunUuid) => currentGroupOrRunUuid !== groupOrRunUuid)
            : [...currentlyHiddenRows, groupOrRunUuid];

          return {
            ...currentUIState,
            // Set mode to "custom"
            runsHiddenMode: RUNS_VISIBILITY_MODE.CUSTOM,
            runsHidden: newHiddenRows,
          };
        }

        return currentUIState;
      });
    },
    [updateUIState],
  );

  return toggleRowVisibility;
};
