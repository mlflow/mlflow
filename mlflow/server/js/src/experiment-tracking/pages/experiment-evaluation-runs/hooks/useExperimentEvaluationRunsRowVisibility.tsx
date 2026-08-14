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
  const [toggledRuns, setToggledRuns] = useState<Set<string>>(new Set());
  const [visibilityMode, setVisibilityModeState] = useState<RUNS_VISIBILITY_MODE>(RUNS_VISIBILITY_MODE.SHOWALL);

  const isRowHidden = useCallback(
    (rowUuid: string, rowIndex: number, runStatus?: string) => {
      let hiddenByMode = false;

      if (visibilityMode === RUNS_VISIBILITY_MODE.HIDEALL) {
        hiddenByMode = true;
      } else if (visibilityMode === RUNS_VISIBILITY_MODE.FIRST_10_RUNS) {
        hiddenByMode = rowIndex >= 10;
      } else if (visibilityMode === RUNS_VISIBILITY_MODE.FIRST_20_RUNS) {
        hiddenByMode = rowIndex >= 20;
      } else if (visibilityMode === RUNS_VISIBILITY_MODE.HIDE_FINISHED_RUNS) {
        hiddenByMode = ['FINISHED', 'FAILED', 'KILLED'].includes(runStatus ?? '');
      }

      // Then apply overrides
      if (toggledRuns.has(rowUuid)) {
        return !hiddenByMode;
      }

      return hiddenByMode;
    },
    [toggledRuns, visibilityMode],
  );

  const toggleRowVisibility = useCallback((rowUuid: string) => {
    setToggledRuns((prevToggledRuns) => {
      const newToggledRuns = new Set(prevToggledRuns);
      if (newToggledRuns.has(rowUuid)) {
        newToggledRuns.delete(rowUuid);
      } else {
        newToggledRuns.add(rowUuid);
      }
      return newToggledRuns;
    });
  }, []);

  const setVisibilityMode = useCallback((mode: RUNS_VISIBILITY_MODE) => {
    setVisibilityModeState(mode);
    setToggledRuns(new Set());
  }, []);

  const usingCustomVisibility = toggledRuns.size > 0;
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
