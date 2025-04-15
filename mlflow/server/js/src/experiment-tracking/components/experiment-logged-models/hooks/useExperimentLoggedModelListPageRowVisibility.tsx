import { createContext, useCallback, useContext, useMemo, useRef } from 'react';
import { RUNS_VISIBILITY_MODE } from '../../experiment-page/models/ExperimentPageUIState';
import { determineIfRowIsHidden } from '../../experiment-page/utils/experimentPage.common-row-utils';
import { isUndefined } from 'lodash';

const ExperimentLoggedModelListPageRowVisibilityContext = createContext<{
  isRowHidden: (rowUuid: string, rowIndex: number) => boolean;
  setRowVisibilityMode: (visibilityMode: RUNS_VISIBILITY_MODE) => void;
  toggleRowVisibility: (rowUuid: string, rowIndex: number) => void;
  visibilityMode: RUNS_VISIBILITY_MODE;
  usingCustomVisibility: boolean;
}>({
  isRowHidden: () => false,
  setRowVisibilityMode: () => {},
  toggleRowVisibility: () => {},
  visibilityMode: RUNS_VISIBILITY_MODE.FIRST_10_RUNS,
  usingCustomVisibility: false,
});

// Utility function that determines if a particular table row should be hidden,
// based on the selected mode, position on the list and the visibility map.
export const isLoggedModelRowHidden = (
  rowsVisibilityMode: RUNS_VISIBILITY_MODE,
  runUuid: string,
  rowIndex: number,
  runsVisibilityMap: Record<string, boolean>,
) => {
  // If using rows visibility map, we should always use it to determine visibility
  if (!isUndefined(runsVisibilityMap[runUuid])) {
    return !runsVisibilityMap[runUuid];
  }
  if (rowsVisibilityMode === RUNS_VISIBILITY_MODE.HIDEALL) {
    return true;
  }
  if (rowsVisibilityMode === RUNS_VISIBILITY_MODE.FIRST_10_RUNS) {
    return rowIndex >= 10;
  }
  if (rowsVisibilityMode === RUNS_VISIBILITY_MODE.FIRST_20_RUNS) {
    return rowIndex >= 20;
  }

  return false;
};

export const ExperimentLoggedModelListPageRowVisibilityContextProvider = ({
  children,
  visibilityMap = {},
  visibilityMode,
  setRowVisibilityMode,
  toggleRowVisibility,
}: {
  visibilityMap?: Record<string, boolean>;
  visibilityMode: RUNS_VISIBILITY_MODE;
  children: React.ReactNode;
  setRowVisibilityMode: (visibilityMode: RUNS_VISIBILITY_MODE) => void;
  toggleRowVisibility: (rowUuid: string, rowIndex: number) => void;
}) => {
  const isRowHidden = useCallback(
    (rowUuid: string, rowIndex: number) => isLoggedModelRowHidden(visibilityMode, rowUuid, rowIndex, visibilityMap),
    [visibilityMap, visibilityMode],
  );

  const usingCustomVisibility = useMemo(() => Object.keys(visibilityMap).length > 0, [visibilityMap]);

  const contextValue = useMemo(
    () => ({ isRowHidden, setRowVisibilityMode, toggleRowVisibility, visibilityMode, usingCustomVisibility }),
    [isRowHidden, setRowVisibilityMode, toggleRowVisibility, visibilityMode, usingCustomVisibility],
  );

  return (
    <ExperimentLoggedModelListPageRowVisibilityContext.Provider value={contextValue}>
      {children}
    </ExperimentLoggedModelListPageRowVisibilityContext.Provider>
  );
};

export const useExperimentLoggedModelListPageRowVisibilityContext = () =>
  useContext(ExperimentLoggedModelListPageRowVisibilityContext);
