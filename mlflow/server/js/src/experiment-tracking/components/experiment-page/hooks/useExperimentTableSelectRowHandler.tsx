import type { GridApi, RowSelectedEvent, SelectionChangedEvent } from '@ag-grid-community/core';
import { useCallback, useRef } from 'react';
import type { ExperimentPageViewState } from '../models/ExperimentPageViewState';
import type { RunRowType } from '../utils/experimentPage.row-types';
import { uniqBy } from 'lodash';

/**
 * Helper function that select particular run rows in the ag-grid.
 */
const agGridSelectRunRows = (runUuids: string[], isSelected: boolean, gridApi: GridApi) => {
  gridApi.forEachNode((node) => {
    if (node.data?.isLoadMoreRow) {
      return;
    }
    const { runInfo, runDateAndNestInfo: childRunDateInfo } = node.data as RunRowType;

    if (!runInfo) {
      return;
    }

    const childrenRunUuid = runInfo.runUuid;
    if (runUuids.includes(childrenRunUuid)) {
      // If we found children being parents, mark their children
      // to be selected as well.
      if (childRunDateInfo?.childrenIds) {
        runUuids.push(...childRunDateInfo.childrenIds);
      }

      node.setSelected(isSelected, false, true);
    }
  });
};

/**
 * Helper function that select particular group rows in the ag-grid.
 */
const agGridSelectGroupRows = (rowData: RunRowType[], gridApi: GridApi) => {
  gridApi.forEachNode((node) => {
    const data: RunRowType = node.data;
    if (!data.groupParentInfo) {
      return;
    }

    // If all runs belonging to the group are selected, select the group
    if (data.groupParentInfo.runUuids.every((runUuid) => rowData.some((row) => row.runUuid === runUuid))) {
      node.setSelected(true, false, true);
    }

    // If none of the runs belonging to the group are selected, deselect the group
    if (!data.groupParentInfo.runUuids.some((runUuid) => rowData.some((row) => row.runUuid === runUuid))) {
      node.setSelected(false, false, true);
    }
  });
};

/**
 * Returns handlers for row selection in the experiment runs table.
 * Supports groups, nested runs and regular flat hierarchy.
 */
export const useExperimentTableSelectRowHandler = (
  updateViewState: (newPartialViewState: Partial<ExperimentPageViewState>) => void,
) => {
  const onSelectionChange = useCallback(
    ({ api }: SelectionChangedEvent) => {
      const selectedUUIDs: string[] = api
        .getSelectedRows()
        // Filter out "load more" and group rows
        .filter((row) => row.runInfo)
        .map(({ runInfo }) => runInfo.runUuid);
      updateViewState({
        runsSelected: selectedUUIDs.reduce((aggregate, curr) => ({ ...aggregate, [curr]: true }), {}),
      });
    },
    [updateViewState],
  );

  const handleRowSelected = useCallback((event: RowSelectedEvent) => {
    // Let's check if the actual number of selected rows have changed
    // to avoid empty runs
    const isSelected = Boolean(event.node.isSelected());

    // We will continue only if the selected row has properly set runDateInfo
    const { runDateAndNestInfo, runInfo, groupParentInfo } = event.data as RunRowType;

    if (groupParentInfo) {
      agGridSelectRunRows(groupParentInfo.runUuids, isSelected, event.api);
    }

    if (!runDateAndNestInfo) {
      return;
    }
    const { isParent, expanderOpen, childrenIds } = runDateAndNestInfo;

    // We will continue only if the selected row is a parent containing
    // children and is actually expanded
    if (isParent && expanderOpen && childrenIds) {
      const childrenIdsToSelect = childrenIds;
      agGridSelectRunRows(childrenIdsToSelect, isSelected, event.api);
    } else if (runInfo) {
      // If we are selecting a run row, we need to select other runs with the same UUID
      agGridSelectRunRows([runInfo.runUuid], isSelected, event.api);

      // Next, we need to (de)select the group row if all runs belonging to the group are (de)selected
      const selectedRunRows = uniqBy(
        event.api.getSelectedRows().filter((row) => Boolean(row.runUuid)),
        'runUuid',
      );
      agGridSelectGroupRows(selectedRunRows, event.api);
    }
  }, []);

  return { handleRowSelected, onSelectionChange };
};
