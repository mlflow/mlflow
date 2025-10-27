import React, { useCallback, useContext, useMemo, useState } from 'react';
import type { RunRowType } from '../utils/experimentPage.row-types';
import { EvaluationCreatePromptRunModal } from '../../evaluation-artifacts-compare/EvaluationCreatePromptRunModal';
import { shouldEnablePromptLab } from '../../../../common/utils/FeatureUtils';

const CreateNewRunContext = React.createContext<{
  createNewRun: (runToDuplicate?: RunRowType) => void;
}>({
  createNewRun: () => {},
});

/**
 * A thin context wrapper dedicated to invoke "create run" modal in various areas of the experiment runs page UI
 */
export const CreateNewRunContextProvider = ({
  children,
  visibleRuns,
  refreshRuns,
}: {
  children: React.ReactNode;
  visibleRuns: RunRowType[];
  refreshRuns: (() => Promise<never[]>) | (() => Promise<any> | null) | (() => void);
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [runBeingDuplicated, setRunBeingDuplicated] = useState<RunRowType | null>(null);

  const contextValue = useMemo(
    () => ({
      createNewRun: (runToDuplicate?: RunRowType) => {
        setIsOpen(true);
        setRunBeingDuplicated(runToDuplicate || null);
      },
    }),
    [],
  );

  return (
    <CreateNewRunContext.Provider value={contextValue}>
      {children}
      {shouldEnablePromptLab() && (
        <EvaluationCreatePromptRunModal
          visibleRuns={visibleRuns}
          isOpen={isOpen}
          closeModal={() => setIsOpen(false)}
          runBeingDuplicated={runBeingDuplicated}
          refreshRuns={refreshRuns}
        />
      )}
    </CreateNewRunContext.Provider>
  );
};

export const useCreateNewRun = () => useContext(CreateNewRunContext);
