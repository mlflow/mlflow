import { useMemo } from 'react';

import type { LoggedModelProto } from '../../types';
import { ExperimentLinkedPromptsTable } from '../experiment-prompts/ExperimentLinkedPromptsTable';
import { MLFLOW_LINKED_PROMPTS_TAG } from '../../constants';

interface Props {
  loggedModel?: LoggedModelProto;
}

export const ExperimentLoggedModelDetailsPageLinkedPromptsTable = ({ loggedModel }: Props) => {
  const tags = loggedModel?.info?.tags;
  const experimentId = loggedModel?.info?.experiment_id ?? '';
  const linkedPromptsTag = tags?.find(({ key }) => key === MLFLOW_LINKED_PROMPTS_TAG);
  const rawLinkedPrompts: { name: string; version: string }[] = useMemo(() => {
    try {
      return JSON.parse(linkedPromptsTag?.value ?? '[]');
    } catch (e) {
      // fail gracefully, just don't show any linked prompts
      return [];
    }
  }, [linkedPromptsTag]);

  const data = useMemo(
    () => rawLinkedPrompts.map((prompt) => ({ ...prompt, experimentId })),
    [rawLinkedPrompts, experimentId],
  );

  return <ExperimentLinkedPromptsTable data={data} />;
};
