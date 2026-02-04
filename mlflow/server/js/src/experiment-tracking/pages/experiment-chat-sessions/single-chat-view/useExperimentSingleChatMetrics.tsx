import { getTraceTokenUsage, type ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';
import { first, last } from 'lodash';
import { useMemo } from 'react';

const SIMULATION_GOAL_KEY = 'mlflow.simulation.goal';
const SIMULATION_PERSONA_KEY = 'mlflow.simulation.persona';

type TraceTokenUsage = ReturnType<typeof getTraceTokenUsage>;
export interface ExperimentSingleChatMetrics {
  sessionTokens: TraceTokenUsage;
  sessionLatency: number | undefined;
  goal?: string;
  persona?: string;
  perTurnMetrics?: {
    tokens: TraceTokenUsage;
    latency: string | undefined;
  }[];
}

const emptyMetrics: ExperimentSingleChatMetrics = {
  sessionTokens: { input_tokens: 0, output_tokens: 0 },
  sessionLatency: 0,
  goal: undefined,
  persona: undefined,
  perTurnMetrics: [],
};

export const useExperimentSingleChatMetrics = ({
  traceInfos,
}: {
  traceInfos?: ModelTraceInfoV3[];
}): ExperimentSingleChatMetrics =>
  useMemo(() => {
    const lastTurn = last(traceInfos);
    if (!lastTurn) {
      return emptyMetrics;
    }
    const sessionTokens = getTraceTokenUsage(lastTurn);

    const firstTurn = first(traceInfos);
    const goal = firstTurn?.trace_metadata?.[SIMULATION_GOAL_KEY];
    const persona = firstTurn?.trace_metadata?.[SIMULATION_PERSONA_KEY];
    const { sessionLatency, perTurnMetrics } =
      traceInfos?.reduce<{
        sessionLatency: number;
        perTurnMetrics: {
          tokens: TraceTokenUsage;
          latency: string | undefined;
        }[];
      }>(
        (aggregate, turnTraceInfo) => {
          const turnTimeInSeconds = parseFloat(turnTraceInfo.execution_duration || '0');
          return {
            sessionLatency: aggregate.sessionLatency + turnTimeInSeconds,
            perTurnMetrics: [
              ...aggregate.perTurnMetrics,
              {
                tokens: getTraceTokenUsage(turnTraceInfo),
                latency: turnTraceInfo.execution_duration,
              },
            ],
          };
        },
        {
          sessionLatency: 0,
          perTurnMetrics: [],
        },
      ) ?? {};

    return {
      sessionTokens,
      sessionLatency,
      goal,
      persona,
      perTurnMetrics,
    };
  }, [traceInfos]);
