import React, { useMemo } from 'react';
import { FeedbackAssessment, ModelTrace } from '../ModelTrace.types';

interface TraceScorerJudgeEvaluationResult {
  trace: ModelTrace | null;
  results: FeedbackAssessment[];
  error: string | null;
}

interface SessionScorerJudgeEvaluationResult {
  sessionId: string;
  traces: ModelTrace[] | null;
  results: FeedbackAssessment[];
  error: string | null;
}

export interface ScorerFinishedEvent {
  requestKey: string;
  status: string;
  results?: (TraceScorerJudgeEvaluationResult | SessionScorerJudgeEvaluationResult)[];
  error?: Error | null;
}

export interface ModelTraceExplorerRunJudgeConfig {
  renderRunJudgeModal?: ({
    itemId,
    visible,
    onClose,
  }: {
    /**
     * The ID of the trace or session to run the judge on.
     */
    itemId: string;
    visible: boolean;
    onClose: () => void;
  }) => React.ReactNode;
  /** Current state of all active evaluations */
  evaluations?: Record<
    string,
    {
      requestKey: string;
      tracesData?: Record<string, ModelTrace>;
      results?: (TraceScorerJudgeEvaluationResult | SessionScorerJudgeEvaluationResult)[];
      isLoading: boolean;
      label: string;
      error?: Error | null;
    }
  >;
  /** Subscribe to scorer update events. Returns unsubscribe function. */
  subscribeToScorerFinished?: (callback: (event: ScorerFinishedEvent) => void) => () => void;
  scope?: 'sessions' | 'traces';
}

const ModelTraceExplorerRunJudgesContext = React.createContext<ModelTraceExplorerRunJudgeConfig>({
  renderRunJudgeModal: undefined,
  evaluations: undefined,
  subscribeToScorerFinished: undefined,
});

/**
 * Provides context for running judges on traces.
 * Contains:
 * - a function to render a button to run a judge on a trace
 * - current state of all active evaluations
 * - a function to subscribe to scorer update events
 */
export const ModelTraceExplorerRunJudgesContextProvider = ({
  children,
  renderRunJudgeModal,
  evaluations,
  subscribeToScorerFinished,
  scope,
}: ModelTraceExplorerRunJudgeConfig & {
  children: React.ReactNode;
}) => {
  const contextValue = useMemo(
    () => ({ renderRunJudgeModal, evaluations, subscribeToScorerFinished, scope }),
    [renderRunJudgeModal, evaluations, subscribeToScorerFinished, scope],
  );
  return (
    <ModelTraceExplorerRunJudgesContext.Provider value={contextValue}>
      {children}
    </ModelTraceExplorerRunJudgesContext.Provider>
  );
};

export const useModelTraceExplorerRunJudgesContext = () => React.useContext(ModelTraceExplorerRunJudgesContext);
