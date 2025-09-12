import { useCallback, useRef } from 'react';
import { useUpdateExperimentViewUIState } from '../contexts/ExperimentPageUIStateContext';
import { RUNS_VISIBILITY_MODE } from '../models/ExperimentPageUIState';
import type { RunRowType } from '../utils/experimentPage.row-types';
import {
  shouldEnableToggleIndividualRunsInGroups,
  shouldUseRunRowsVisibilityMap,
} from '../../../../common/utils/FeatureUtils';

export const useToggleRowVisibilityCallback = (tableRows: RunRowType[], useGroupedValuesInCharts = true) => {
  const updateUIState = useUpdateExperimentViewUIState();

  // We're going to use current state of the table rows to determine which rows are hidden.
  // Since we're interested only in the latest data, we avoid using state here to avoid unnecessary re-renders.
  const immediateTableRows = useRef(tableRows);
  immediateTableRows.current = tableRows;

  const toggleRowUsingVisibilityMap = useCallback(
    (mode: RUNS_VISIBILITY_MODE, groupOrRunUuid?: string, isCurrentlyVisible?: boolean) => {
      updateUIState((currentUIState) => {
        // If user has toggled a run or a group manually, we need to update the visibility map
        if (mode === RUNS_VISIBILITY_MODE.CUSTOM && groupOrRunUuid) {
          const newRunsVisibilityMap = {
            ...currentUIState.runsVisibilityMap,
          };

          // Check if the toggles row is a run group
          const currentToggledGroupInfo = immediateTableRows.current.find(
            ({ rowUuid, groupParentInfo }) => rowUuid === groupOrRunUuid && groupParentInfo,
          )?.groupParentInfo;

          // If we're toggling a group and we're not using grouped values in charts,
          // then toggle all runs in the group
          if (
            currentToggledGroupInfo &&
            shouldEnableToggleIndividualRunsInGroups() &&
            useGroupedValuesInCharts === false
          ) {
            for (const runUuid of currentToggledGroupInfo.runUuids) {
              newRunsVisibilityMap[runUuid] = !isCurrentlyVisible;
            }
          } else {
            newRunsVisibilityMap[groupOrRunUuid] = !isCurrentlyVisible;
          }

          return {
            ...currentUIState,
            runsVisibilityMap: newRunsVisibilityMap,
          };
        }
        // Otherwise, we're toggling a predefined visibility mode
        // and clearing the visibility map
        if (
          [
            RUNS_VISIBILITY_MODE.SHOWALL,
            RUNS_VISIBILITY_MODE.HIDEALL,
            RUNS_VISIBILITY_MODE.FIRST_10_RUNS,
            RUNS_VISIBILITY_MODE.FIRST_20_RUNS,
            RUNS_VISIBILITY_MODE.HIDE_FINISHED_RUNS,
          ].includes(mode)
        ) {
          return {
            ...currentUIState,
            runsHiddenMode: mode,
            runsHidden: [],
            runsVisibilityMap: {},
          };
        }

        return currentUIState;
      });
    },
    [updateUIState, useGroupedValuesInCharts],
  );

  /**
   * @deprecated `toggleRowUsingVisibilityMap` replaces this function.
   * This one should be removed after ramping up `runsVisibility` field.
   */
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
        } else if (mode === RUNS_VISIBILITY_MODE.HIDE_FINISHED_RUNS) {
          // Case #5: Hiding finished runs
          return {
            ...currentUIState,
            runsHiddenMode: RUNS_VISIBILITY_MODE.HIDE_FINISHED_RUNS,
            runsHidden: [],
          };
        }

        // Case #6: Custom visibility mode enabled by manually toggling visibility of a run or a group
        if (groupOrRunUuid) {
          // Determine which runs are hidden at the moment
          const currentlyHiddenRows = immediateTableRows.current
            .filter(({ hidden }) => hidden)
            .map(({ groupParentInfo, rowUuid, runUuid }) => (groupParentInfo ? rowUuid : runUuid));

          // Check if the toggles row is a run group
          const currentToggledGroupInfo = immediateTableRows.current.find(
            ({ rowUuid, groupParentInfo }) => rowUuid === groupOrRunUuid && groupParentInfo,
          )?.groupParentInfo;

          // If we're toggling a group and we're not using grouped values in charts,
          // then toggle all runs in the group
          if (
            currentToggledGroupInfo &&
            shouldEnableToggleIndividualRunsInGroups() &&
            useGroupedValuesInCharts === false
          ) {
            let newHiddenRows: string[] = [];

            // Depending on the current state of the group, we either show all runs or hide all runs
            if (currentToggledGroupInfo.allRunsHidden) {
              newHiddenRows = currentlyHiddenRows.filter(
                (currentGroupOrRunUuid) => !currentToggledGroupInfo.runUuids.includes(currentGroupOrRunUuid),
              );
            } else {
              newHiddenRows = currentlyHiddenRows.concat(
                currentToggledGroupInfo.runUuids.filter((runUuid) => !currentlyHiddenRows.includes(runUuid)),
              );
            }
            return {
              ...currentUIState,
              // Set mode to "custom"
              runsHiddenMode: RUNS_VISIBILITY_MODE.CUSTOM,
              runsHidden: newHiddenRows,
            };
          }

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
    [updateUIState, useGroupedValuesInCharts],
  );

  return shouldUseRunRowsVisibilityMap() ? toggleRowUsingVisibilityMap : toggleRowVisibility;
};
