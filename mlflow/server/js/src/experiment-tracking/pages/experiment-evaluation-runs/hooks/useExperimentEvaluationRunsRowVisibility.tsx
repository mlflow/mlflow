import { createContext, useCallback, useContext, useMemo, useState } from 'react';

const ExperimentEvaluationRunsRowVisibilityContext = createContext<{
  isRowHidden: (rowUuid: string) => boolean;
  toggleRowVisibility: (rowUuid: string) => void;
}>({
  isRowHidden: () => false,
  toggleRowVisibility: () => {},
});

export const ExperimentEvaluationRunsRowVisibilityProvider = ({ children }: { children: React.ReactNode }) => {
  const [hiddenRuns, setHiddenRuns] = useState<Set<string>>(new Set());

  const isRowHidden = useCallback(
    (rowUuid: string) => {
      return hiddenRuns.has(rowUuid);
    },
    [hiddenRuns],
  );

  const toggleRowVisibility = useCallback(
    (rowUuid: string) => {
      setHiddenRuns((prevHiddenRuns) => {
        const newHiddenRuns = new Set(prevHiddenRuns);
        if (newHiddenRuns.has(rowUuid)) {
          newHiddenRuns.delete(rowUuid);
        } else {
          newHiddenRuns.add(rowUuid);
        }
        return newHiddenRuns;
      });
    },
    [setHiddenRuns],
  );

  const value = useMemo(() => ({ isRowHidden, toggleRowVisibility }), [isRowHidden, toggleRowVisibility]);

  return (
    <ExperimentEvaluationRunsRowVisibilityContext.Provider value={value}>
      {children}
    </ExperimentEvaluationRunsRowVisibilityContext.Provider>
  );
};

export const useExperimentEvaluationRunsRowVisibility = () => {
  return useContext(ExperimentEvaluationRunsRowVisibilityContext);
};
