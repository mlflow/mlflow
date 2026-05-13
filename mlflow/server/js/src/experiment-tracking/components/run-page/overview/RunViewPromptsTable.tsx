import { useMemo } from 'react';

import type { KeyValueEntity } from '../../../../common/types';
import { ExperimentLinkedPromptsTable } from '../../experiment-prompts/ExperimentLinkedPromptsTable';
import { MLFLOW_LINKED_PROMPTS_TAG } from '../../../constants';

export const RunViewPromptsTable = ({
  runTags,
  experimentId,
}: {
  runTags: Record<string, KeyValueEntity>;
  experimentId?: string | null;
}) => {
  const linkedPromptsTagValue = runTags[MLFLOW_LINKED_PROMPTS_TAG]?.value;

  const rawLinkedPrompts: { name: string; version: string }[] = useMemo(() => {
    if (!linkedPromptsTagValue) {
      return [];
    }
    try {
      return JSON.parse(linkedPromptsTagValue ?? '[]');
    } catch (e) {
      // fail gracefully, just don't show any linked prompts
      return [];
    }
  }, [linkedPromptsTagValue]);

  const data = useMemo(
    () => rawLinkedPrompts.map((prompt) => ({ ...prompt, experimentId: experimentId ?? '' })),
    [rawLinkedPrompts, experimentId],
  );

  return <ExperimentLinkedPromptsTable data={data} />;
};
