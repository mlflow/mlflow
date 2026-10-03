import { getTraceCost, getTraceTokenUsage, type ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';
import { SIMULATION_GOAL_KEY, SIMULATION_PERSONA_KEY } from '@databricks/web-shared/genai-traces-table';
import { first, last } from 'lodash';
import { useMemo } from 'react';

type TraceTokenUsage = ReturnType<typeof getTraceTokenUsage>;
export interface ExperimentSingleChatMetrics {
  sessionTokens: TraceTokenUsage;
  sessionCost?: {
    input_cost: number;
    output_cost: number;
    total_cost: number;
  };
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

const sumFiniteValues = (values: Array<number | undefined>): number | undefined => {
  const finiteValues = values.filter((value): value is number => typeof value === 'number' && Number.isFinite(value));
  return finiteValues.length > 0 ? finiteValues.reduce((sum, value) => sum + value, 0) : undefined;
};

export const useExperimentSingleChatMetrics = ({
  traceInfos,
}: {
  traceInfos?: ModelTraceInfoV3[];
}): ExperimentSingleChatMetrics =>
  useMemo(() => {
    const availableTraceInfos = traceInfos ?? [];
    const lastTurn = last(availableTraceInfos);
    if (!lastTurn) {
      return emptyMetrics;
    }
    const tokenUsageByTurn = availableTraceInfos.map(getTraceTokenUsage);
    const inputTokens = sumFiniteValues(tokenUsageByTurn.map((usage) => usage?.input_tokens));
    const outputTokens = sumFiniteValues(tokenUsageByTurn.map((usage) => usage?.output_tokens));
    const totalTokens = sumFiniteValues(
      tokenUsageByTurn.map((usage) =>
        usage?.total_tokens !== undefined
          ? usage.total_tokens
          : sumFiniteValues([usage?.input_tokens, usage?.output_tokens]),
      ),
    );
    const cacheReadInputTokens = sumFiniteValues(tokenUsageByTurn.map((usage) => usage?.cache_read_input_tokens));
    const cacheCreationInputTokens = sumFiniteValues(
      tokenUsageByTurn.map((usage) => usage?.cache_creation_input_tokens),
    );
    const sessionTokens = {
      ...(inputTokens !== undefined ? { input_tokens: inputTokens } : {}),
      ...(outputTokens !== undefined ? { output_tokens: outputTokens } : {}),
      ...(totalTokens !== undefined ? { total_tokens: totalTokens } : {}),
      ...(cacheReadInputTokens !== undefined ? { cache_read_input_tokens: cacheReadInputTokens } : {}),
      ...(cacheCreationInputTokens !== undefined ? { cache_creation_input_tokens: cacheCreationInputTokens } : {}),
    };
    const costByTurn = availableTraceInfos.map(getTraceCost);
    const inputCost = sumFiniteValues(costByTurn.map((cost) => cost?.input_cost));
    const outputCost = sumFiniteValues(costByTurn.map((cost) => cost?.output_cost));
    const totalCost = sumFiniteValues(
      costByTurn.map((cost) =>
        cost?.total_cost !== undefined ? cost.total_cost : sumFiniteValues([cost?.input_cost, cost?.output_cost]),
      ),
    );
    const sessionCost =
      inputCost !== undefined || outputCost !== undefined || totalCost !== undefined
        ? {
            input_cost: inputCost ?? 0,
            output_cost: outputCost ?? 0,
            total_cost: totalCost ?? (inputCost ?? 0) + (outputCost ?? 0),
          }
        : undefined;

    const firstTurn = first(availableTraceInfos);
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
      sessionCost,
      sessionLatency,
      goal,
      persona,
      perTurnMetrics,
    };
  }, [traceInfos]);
