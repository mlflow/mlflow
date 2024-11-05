import React, { useMemo } from 'react';
import { RUNS_VISIBILITY_MODE } from '../../models/ExperimentPageUIState';

const ExperimentViewRunsTableHeaderContext = React.createContext({
  runsHiddenMode: RUNS_VISIBILITY_MODE.FIRST_10_RUNS,
  useGroupedValuesInCharts: true,
  usingCustomVisibility: false,
  allRunsHidden: false,
});

/**
 * A tiny context that passes the current runs hidden mode to the header cell renderer.
 *
 * This is needed because ag-grid context is unreliable and does not always update when the
 * runs hidden mode changes. The solution below is way more performant than recreating column definitions
 * based on a new value.
 */
export const ExperimentViewRunsTableHeaderContextProvider = ({
  children,
  runsHiddenMode,
  useGroupedValuesInCharts,
  usingCustomVisibility,
  allRunsHidden,
}: {
  children: React.ReactNode;
  runsHiddenMode: RUNS_VISIBILITY_MODE;
  useGroupedValuesInCharts?: boolean;
  /**
   * Whether the user is using custom visibility settings (at least one row is configured manually)
   */
  usingCustomVisibility?: boolean;
  /**
   * Whether all runs are hidden
   */
  allRunsHidden?: boolean;
}) => {
  const contextValue = useMemo(
    () => ({
      runsHiddenMode,
      useGroupedValuesInCharts: useGroupedValuesInCharts ?? true,
      usingCustomVisibility: usingCustomVisibility ?? false,
      allRunsHidden: allRunsHidden ?? false,
    }),
    [runsHiddenMode, useGroupedValuesInCharts, usingCustomVisibility, allRunsHidden],
  );
  return (
    <ExperimentViewRunsTableHeaderContext.Provider value={contextValue}>
      {children}
    </ExperimentViewRunsTableHeaderContext.Provider>
  );
};

export const useExperimentViewRunsTableHeaderContext = () => React.useContext(ExperimentViewRunsTableHeaderContext);
