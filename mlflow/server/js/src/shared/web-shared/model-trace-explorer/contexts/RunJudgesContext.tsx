import React, { useMemo } from 'react';

export interface ModelTraceExplorerRunJudgeConfig {
  renderRunJudgeButton?: ({ traceId, trigger }: { traceId: string; trigger: React.ReactNode }) => React.ReactNode;
}

const ModelTraceExplorerRunJudgesContext = React.createContext<ModelTraceExplorerRunJudgeConfig>({
  renderRunJudgeButton: undefined,
});

/**
 * Provides context for running judges on traces.
 * Contains:
 * - a function to render a button to run a judge on a trace
 * - TODO: logic for monitoring and updating the status of the judge runs
 */
export const ModelTraceExplorerRunJudgesContextProvider = ({
  children,
  renderRunJudgeButton,
}: ModelTraceExplorerRunJudgeConfig & {
  children: React.ReactNode;
}) => {
  const contextValue = useMemo(() => ({ renderRunJudgeButton }), [renderRunJudgeButton]);
  return (
    <ModelTraceExplorerRunJudgesContext.Provider value={contextValue}>
      {children}
    </ModelTraceExplorerRunJudgesContext.Provider>
  );
};

export const useModelTraceExplorerRunJudgesContext = () => React.useContext(ModelTraceExplorerRunJudgesContext);
