import { useMemo } from 'react';

import { ModelTraceExplorerLinkedPromptsTable } from './ModelTraceExplorerLinkedPromptsTable';
import { MLFLOW_LINKED_PROMPTS_TAG } from './utils';
import type { ModelTrace } from '../ModelTrace.types';
import { isV3ModelTraceInfo } from '../ModelTraceExplorer.utils';

export const ModelTraceExplorerLinkedPromptsView = ({ modelTraceInfo }: { modelTraceInfo: ModelTrace['info'] }) => {
  const traceInfo = modelTraceInfo;
  const traceInfoV3 = isV3ModelTraceInfo(traceInfo) ? traceInfo : undefined;

  const experimentId =
    traceInfoV3?.trace_location.type === 'MLFLOW_EXPERIMENT'
      ? traceInfoV3.trace_location.mlflow_experiment.experiment_id
      : undefined;

  const linkedPromptsTagValue = traceInfoV3?.tags?.[MLFLOW_LINKED_PROMPTS_TAG];
  const rawLinkedPrompts: { name: string; version: string }[] = useMemo(() => {
    try {
      return JSON.parse(linkedPromptsTagValue ?? '[]');
    } catch (e) {
      // fail gracefully, just don't show any linked prompts
      return [];
    }
  }, [linkedPromptsTagValue]);

  const data = useMemo(
    () => (experimentId ? rawLinkedPrompts.map((prompt) => ({ ...prompt, experimentId })) : []),
    [rawLinkedPrompts, experimentId],
  );

  return <ModelTraceExplorerLinkedPromptsTable data={data} />;
};
