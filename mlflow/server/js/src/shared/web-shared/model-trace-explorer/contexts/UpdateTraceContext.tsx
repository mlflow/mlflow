import React, { useMemo } from 'react';

import type { ModelTrace, ModelTraceInfoV3 } from '../ModelTrace.types';

const ModelTraceExplorerUpdateTraceContext = React.createContext<{
  sqlWarehouseId?: string;
  modelTraceInfo?: ModelTrace['info'];
  invalidateTraceQuery?: (traceId?: string) => void;
  chatSessionId?: string;
  runJudgeContext?: {
    renderRunJudgeButton: ({
      traceId,
      onRunJudgeFinishedCallback,
      disabled,
    }: {
      traceId: string;
      onRunJudgeFinishedCallback: () => void;
      disabled?: boolean;
    }) => React.ReactNode;
    judgeExecutionState: { isLoading: boolean; scorerInProgress: string | undefined };
  };
}>({
  sqlWarehouseId: undefined,
  modelTraceInfo: undefined,
  invalidateTraceQuery: undefined,
  chatSessionId: undefined,
});

/**
 * Provides configuration context used to update trace data (assessments, tags).
 * Contains:
 * - an ID of the SQL warehouse to use for queries
 * - info of the currently selected model trace
 */
export const ModelTraceExplorerUpdateTraceContextProvider = ({
  sqlWarehouseId,
  modelTraceInfo,
  children,
  invalidateTraceQuery,
  chatSessionId,
  runJudgeContext,
}: {
  sqlWarehouseId?: string;
  modelTraceInfo?: ModelTrace['info'];
  children: React.ReactNode;
  invalidateTraceQuery?: (traceId?: string) => void;
  chatSessionId?: string;
  runJudgeContext?: {
    renderRunJudgeButton: ({
      traceId,
      onRunJudgeFinishedCallback,
      disabled,
    }: {
      traceId: string;
      onRunJudgeFinishedCallback: () => void;
      disabled?: boolean;
    }) => React.ReactNode;
    judgeExecutionState: { isLoading: boolean; scorerInProgress: string | undefined };
  };
}) => {
  const contextValue = useMemo(
    () => ({ sqlWarehouseId, modelTraceInfo, invalidateTraceQuery, chatSessionId, runJudgeContext }),
    [sqlWarehouseId, modelTraceInfo, invalidateTraceQuery, chatSessionId, runJudgeContext],
  );
  return (
    <ModelTraceExplorerUpdateTraceContext.Provider value={contextValue}>
      {children}
    </ModelTraceExplorerUpdateTraceContext.Provider>
  );
};

export const useModelTraceExplorerUpdateTraceContext = () => React.useContext(ModelTraceExplorerUpdateTraceContext);
