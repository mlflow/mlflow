import { TOKEN_USAGE_METADATA_KEY, type ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';
import { last } from 'lodash';
import { useMemo } from 'react';

interface TraceTokenUsage {
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
}

export interface ExperimentSingleChatMetrics {
  sessionTokens: TraceTokenUsage;
  sessionLatency: number | undefined;
  perTurnMetrics?: {
    tokens: TraceTokenUsage;
    latency: string | undefined;
  }[];
}

const emptyMetrics: ExperimentSingleChatMetrics = {
  sessionTokens: { input_tokens: 0, output_tokens: 0, total_tokens: 0 },
  sessionLatency: 0,
  perTurnMetrics: [],
};

const getTraceTokenUsage = (traceInfo: ModelTraceInfoV3): TraceTokenUsage => {
  const tokenUsage = traceInfo?.trace_metadata?.[TOKEN_USAGE_METADATA_KEY];
  try {
    const parsed = tokenUsage ? JSON.parse(tokenUsage) : {};
    return {
      input_tokens: parsed.input_tokens ?? 0,
      output_tokens: parsed.output_tokens ?? 0,
      total_tokens: parsed.total_tokens ?? 0,
    };
  } catch {
    return { input_tokens: 0, output_tokens: 0, total_tokens: 0 };
  }
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
      perTurnMetrics,
    };
  }, [traceInfos]);
