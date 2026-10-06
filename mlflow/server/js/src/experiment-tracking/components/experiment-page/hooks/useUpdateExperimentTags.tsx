import { useMutation } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { useEditKeyValueTagsModal } from '../../../../common/hooks/useEditKeyValueTagsModal';
import { useCallback } from 'react';
import { diffCurrentAndNewTags, isUserFacingTag } from '../../../../common/utils/TagUtils';
import { MlflowService } from '../../../sdk/MlflowService';
import { ErrorWrapper } from '../../../../common/utils/ErrorWrapper';
import type { ExperimentEntity } from '../../../types';

type UpdateTagsPayload = {
  experimentId: string;
  toAdd: { key: string; value: string }[];
  toDelete: { key: string }[];
};

/** The server's reason for one rejected tag write, or a generic fallback. */
const reasonOf = (reason: unknown): string => {
  const raw =
    reason instanceof ErrorWrapper
      ? reason.getUserVisibleError()?.message
      : reason instanceof Error
        ? reason.message
        : undefined;
  // A server message is usually a full sentence ending in a period, and this one gets
  // composed into a longer sentence, so trim the terminator rather than emit "denied.. ".
  return raw?.trim().replace(/\.+$/, '') || 'the request was rejected';
};

/**
 * Describe the outcome of a tag save in which at least one write was rejected.
 *
 * A tag save is one request PER TAG, so a save is not atomic: when some are permitted and
 * some are not, the permitted ones land. `Promise.all` rejects on the first failure, which
 * reported the whole save as failed while leaving the successful writes in place -- the
 * user saw "Permission denied", reopened the modal, and found some of their edits applied.
 *
 * There is no server API to make this atomic (`SetExperimentTag` is one tag per call) and
 * a pre-flight check would be a guess, so the honest fix is to say exactly what happened
 * rather than to imply all-or-nothing.
 */
export const describeTagSaveOutcome = (refused: { key: string; reason: unknown }[], savedKeys: string[]): string => {
  const reasons = new Set(refused.map(({ reason }) => reasonOf(reason)));
  // One shared reason is the common case (a condition refusing several keys), so state it
  // once rather than repeating it per key.
  const why =
    reasons.size === 1 ? [...reasons][0] : refused.map(({ key, reason }) => `${key}: ${reasonOf(reason)}`).join('; ');
  const refusedList = refused.map(({ key }) => key).join(', ');
  const head = reasons.size === 1 ? `Could not save ${refusedList}: ${why}` : `Could not save some tags — ${why}`;
  if (savedKeys.length === 0) return `${head}. No changes were made.`;
  return `${head}. These changes were saved: ${savedKeys.join(', ')}.`;
};

export const useUpdateExperimentTags = ({ onSuccess }: { onSuccess?: () => void }) => {
  const updateMutation = useMutation<unknown, Error, UpdateTagsPayload>({
    mutationFn: async ({ toAdd, toDelete, experimentId }) => {
      const operations = [
        ...toAdd.map(({ key, value }) => ({
          key,
          send: () => MlflowService.setExperimentTag({ experiment_id: experimentId, key, value }),
        })),
        ...toDelete.map(({ key }) => ({
          key,
          send: () => MlflowService.deleteExperimentTag({ experiment_id: experimentId, key }),
        })),
      ];

      // `allSettled`, not `all`: every request is in flight already, so the successes land
      // whatever the failures do. Waiting for all of them is what makes the report true.
      const results = await Promise.allSettled(operations.map(({ send }) => send()));
      const refused = results.flatMap((result, i) =>
        result.status === 'rejected' ? [{ key: operations[i].key, reason: result.reason }] : [],
      );
      if (refused.length === 0) return results;

      const savedKeys = operations.filter((_, i) => results[i].status === 'fulfilled').map(({ key }) => key);
      throw new Error(describeTagSaveOutcome(refused, savedKeys));
    },
  });

  const { EditTagsModal, showEditTagsModal, isLoading } = useEditKeyValueTagsModal<
    Pick<ExperimentEntity, 'experimentId' | 'name' | 'tags'>
  >({
    valueRequired: true,
    saveTagsHandler: (experiment, currentTags, newTags) => {
      const { addedOrModifiedTags, deletedTags } = diffCurrentAndNewTags(currentTags, newTags);

      return new Promise<void>((resolve, reject) => {
        if (!experiment) {
          return reject();
        }
        // Send all requests to the mutation
        updateMutation.mutate(
          {
            experimentId: experiment.experimentId,
            toAdd: addedOrModifiedTags,
            toDelete: deletedTags,
          },
          {
            onSuccess: () => {
              resolve();
              onSuccess?.();
            },
            onError: reject,
          },
        );
      });
    },
  });

  const showEditExperimentTagsModal = useCallback(
    (experiment: ExperimentEntity) =>
      showEditTagsModal({
        experimentId: experiment.experimentId,
        name: experiment.name,
        tags: experiment.tags.filter((tag) => isUserFacingTag(tag.key)),
      }),
    [showEditTagsModal],
  );

  return { EditTagsModal, showEditExperimentTagsModal, isLoading };
};
