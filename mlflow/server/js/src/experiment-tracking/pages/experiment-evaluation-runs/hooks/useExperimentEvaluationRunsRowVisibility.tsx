import { createContext, useCallback, useContext, useMemo, useState } from 'react';
import { RUNS_VISIBILITY_MODE } from '../../../components/experiment-page/models/ExperimentPageUIState';

interface ExperimentEvaluationRunsRowVisibilityContextValue {
  isRowHidden: (rowUuid: string, rowIndex: number, runStatus?: string) => boolean;
  toggleRowVisibility: (rowUuid: string) => void;
  setVisibilityMode: (mode: RUNS_VISIBILITY_MODE) => void;
  visibilityMode: RUNS_VISIBILITY_MODE;
  usingCustomVisibility: boolean;
  allRunsHidden: boolean;
}

const ExperimentEvaluationRunsRowVisibilityContext = createContext<ExperimentEvaluationRunsRowVisibilityContextValue>({
  isRowHidden: () => false,
  toggleRowVisibility: () => {},
  setVisibilityMode: () => {},
  visibilityMode: RUNS_VISIBILITY_MODE.SHOWALL,
  usingCustomVisibility: false,
  allRunsHidden: false,
});

export const ExperimentEvaluationRunsRowVisibilityProvider = ({ children }: { children: React.ReactNode }) => {
  const [hiddenRuns, setHiddenRuns] = useState<Set<string>>(new Set());
  const [visibilityMode, setVisibilityModeState] = useState<RUNS_VISIBILITY_MODE>(RUNS_VISIBILITY_MODE.SHOWALL);

  const isRowHidden = useCallback(
    (rowUuid: string, rowIndex: number, runStatus?: string) => {
      // If custom mode, check the hiddenRuns set
      if (visibilityMode === RUNS_VISIBILITY_MODE.CUSTOM) {
        return hiddenRuns.has(rowUuid);
      }

      // For other modes, apply the mode logic
      if (visibilityMode === RUNS_VISIBILITY_MODE.HIDEALL) {
        return true;
      }

      if (visibilityMode === RUNS_VISIBILITY_MODE.FIRST_10_RUNS) {
        return rowIndex >= 10;
      }

      if (visibilityMode === RUNS_VISIBILITY_MODE.FIRST_20_RUNS) {
        return rowIndex >= 20;
      }

      if (visibilityMode === RUNS_VISIBILITY_MODE.HIDE_FINISHED_RUNS) {
        return ['FINISHED', 'FAILED', 'KILLED'].includes(runStatus ?? '');
      }

      // SHOWALL mode - show everything
      return false;
    },
    [hiddenRuns, visibilityMode],
  );

  const toggleRowVisibility = useCallback((rowUuid: string) => {
    setHiddenRuns((prevHiddenRuns) => {
      const newHiddenRuns = new Set(prevHiddenRuns);
      if (newHiddenRuns.has(rowUuid)) {
        newHiddenRuns.delete(rowUuid);
      } else {
        newHiddenRuns.add(rowUuid);
      }
      return newHiddenRuns;
    });
    // Switch to custom mode when manually toggling
    setVisibilityModeState(RUNS_VISIBILITY_MODE.CUSTOM);
  }, []);

  const setVisibilityMode = useCallback((mode: RUNS_VISIBILITY_MODE) => {
    setVisibilityModeState(mode);
    // Clear custom hidden runs when switching to a predefined mode
    if (mode !== RUNS_VISIBILITY_MODE.CUSTOM) {
      setHiddenRuns(new Set());
    }
  }, []);

  const usingCustomVisibility = visibilityMode === RUNS_VISIBILITY_MODE.CUSTOM && hiddenRuns.size > 0;
  const allRunsHidden = visibilityMode === RUNS_VISIBILITY_MODE.HIDEALL;

  const value = useMemo(
    () => ({
      isRowHidden,
      toggleRowVisibility,
      setVisibilityMode,
      visibilityMode,
      usingCustomVisibility,
      allRunsHidden,
    }),
    [isRowHidden, toggleRowVisibility, setVisibilityMode, visibilityMode, usingCustomVisibility, allRunsHidden],
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
