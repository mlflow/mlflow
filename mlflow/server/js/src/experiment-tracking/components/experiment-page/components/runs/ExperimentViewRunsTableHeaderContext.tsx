import React, { useMemo } from 'react';
import { RUNS_VISIBILITY_MODE } from '../../models/ExperimentPageUIState';

const ExperimentViewRunsTableHeaderContext = React.createContext({
  runsHiddenMode: RUNS_VISIBILITY_MODE.FIRST_10_RUNS,
  useGroupedValuesInCharts: true,
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
}: {
  children: React.ReactNode;
  runsHiddenMode: RUNS_VISIBILITY_MODE;
  useGroupedValuesInCharts?: boolean;
}) => {
  const contextValue = useMemo(
    () => ({ runsHiddenMode, useGroupedValuesInCharts: useGroupedValuesInCharts ?? true }),
    [runsHiddenMode, useGroupedValuesInCharts],
  );
  return (
    <ExperimentViewRunsTableHeaderContext.Provider value={contextValue}>
      {children}
    </ExperimentViewRunsTableHeaderContext.Provider>
  );
};

export const useExperimentViewRunsTableHeaderContext = () => React.useContext(ExperimentViewRunsTableHeaderContext);
