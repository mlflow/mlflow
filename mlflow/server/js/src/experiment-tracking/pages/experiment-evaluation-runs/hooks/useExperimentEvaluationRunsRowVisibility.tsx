import { createContext, useCallback, useContext, useMemo, useState } from 'react';
import { RUNS_VISIBILITY_MODE } from '../../../components/experiment-page/models/ExperimentPageUIState';

interface ExperimentEvaluationRunsRowVisibilityContextValue {
  isRowHidden: (rowUuid: string, rowIndex: number, runStatus?: string) => boolean;
  toggleRowVisibility: (rowUuid: string, rowIndex: number, runStatus?: string) => void;
  setVisibilityMode: (mode: RUNS_VISIBILITY_MODE) => void;
  visibilityMode: RUNS_VISIBILITY_MODE;
  usingCustomVisibility: boolean;
}

const ExperimentEvaluationRunsRowVisibilityContext = createContext<ExperimentEvaluationRunsRowVisibilityContextValue>({
  isRowHidden: () => false,
  toggleRowVisibility: () => {},
  setVisibilityMode: () => {},
  visibilityMode: RUNS_VISIBILITY_MODE.SHOWALL,
  usingCustomVisibility: false,
});

const isVisibleByMode = (mode: RUNS_VISIBILITY_MODE, rowIndex: number, runStatus?: string): boolean => {
  if (mode === RUNS_VISIBILITY_MODE.HIDEALL) return false;
  if (mode === RUNS_VISIBILITY_MODE.FIRST_10_RUNS) return rowIndex < 10;
  if (mode === RUNS_VISIBILITY_MODE.FIRST_20_RUNS) return rowIndex < 20;
  if (mode === RUNS_VISIBILITY_MODE.HIDE_FINISHED_RUNS) {
    return !['FINISHED', 'FAILED', 'KILLED'].includes(runStatus ?? '');
  }
  return true;
};

export const ExperimentEvaluationRunsRowVisibilityProvider = ({ children }: { children: React.ReactNode }) => {
  const [rowVisibilityOverrides, setRowVisibilityOverrides] = useState<Record<string, boolean>>({});
  const [visibilityMode, setVisibilityModeState] = useState<RUNS_VISIBILITY_MODE>(RUNS_VISIBILITY_MODE.SHOWALL);

  const isRowHidden = useCallback(
    (rowUuid: string, rowIndex: number, runStatus?: string) => {
      // Check explicit override - if set, use it directly
      if (rowVisibilityOverrides[rowUuid] !== undefined) {
        return !rowVisibilityOverrides[rowUuid];
      }

      return !isVisibleByMode(visibilityMode, rowIndex, runStatus);
    },
    [rowVisibilityOverrides, visibilityMode],
  );

  const toggleRowVisibility = useCallback(
    (rowUuid: string, rowIndex: number, runStatus?: string) => {
      setRowVisibilityOverrides((prev) => {
        const newVisibilityOverrides = { ...prev };

        if (prev[rowUuid] !== undefined) {
          // Clear override - go back to mode's decision
          delete newVisibilityOverrides[rowUuid];
        } else {
          const visibleByMode = isVisibleByMode(visibilityMode, rowIndex, runStatus);
          newVisibilityOverrides[rowUuid] = !visibleByMode;
        }

        return newVisibilityOverrides;
      });
    },
    [visibilityMode],
  );

  const setVisibilityMode = useCallback((mode: RUNS_VISIBILITY_MODE) => {
    setVisibilityModeState(mode);
    setRowVisibilityOverrides({});
  }, []);

  const usingCustomVisibility = Object.keys(rowVisibilityOverrides).length > 0;

  const value = useMemo(
    () => ({
      isRowHidden,
      toggleRowVisibility,
      setVisibilityMode,
      visibilityMode,
      usingCustomVisibility,
    }),
    [isRowHidden, toggleRowVisibility, setVisibilityMode, visibilityMode, usingCustomVisibility],
  );

  return (
    <ExperimentEvaluationRunsRowVisibilityContext.Provider value={value}>
      {children}
    </ExperimentEvaluationRunsRowVisibilityContext.Provider>
  );
};

export const useExperimentEvaluationRunsRowVisibility = () => {
  return useContext(ExperimentEvaluationRunsRowVisibilityContext);
};
