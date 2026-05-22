import { useCallback, useMemo, useRef, useState } from 'react';
import { Button, PlusIcon, Tag, Tooltip, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { useQueryClient } from '@databricks/web-shared/query-client';
import { KeyValueTagFullViewModal } from '../../experiment-evaluation-datasets/components/KeyValueTagFullViewModal';
import type { DatasetRecord } from '../hooks/useDatasetsQueries';
import {
  listDatasetRecordsQueryKey,
  useUpsertDatasetRecordsMutation,
} from '../hooks/useDatasetsQueries';

interface TagsCellProps {
  record: DatasetRecord;
  datasetId: string;
  /** Called when a tag mutation rejects. Optimistic rollback already happens via the mutation. */
  onSaveError?: (error: unknown) => void;
}

interface TagModalState {
  open: boolean;
  /** Empty key when adding a new tag. */
  key: string;
  value: string;
}

type Tags = NonNullable<DatasetRecord['tags']>;

/**
 * Returns the live tags for `recordId` straight out of the react-query cache. Reading the
 * cache (instead of a closure-captured snapshot) is one half of the concurrency story; the
 * other half is the per-record serialization in `saveTagsUpdate` below — together they
 * ensure two near-simultaneous writes compose correctly instead of clobbering each other.
 */
const readLatestTags = (
  queryClient: ReturnType<typeof useQueryClient>,
  datasetId: string,
  recordId: string,
  fallback: Tags,
): Tags => {
  const cached = queryClient.getQueryData<DatasetRecord[] | undefined>(listDatasetRecordsQueryKey(datasetId));
  const fresh = cached?.find((r) => r.dataset_record_id === recordId)?.tags;
  return fresh ?? fallback;
};

export const TagsCell = ({ record, datasetId, onSaveError }: TagsCellProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const queryClient = useQueryClient();
  const upsertMutation = useUpsertDatasetRecordsMutation(datasetId);

  const [modalState, setModalState] = useState<TagModalState>({ open: false, key: '', value: '' });
  const tags = useMemo<Tags>(() => record.tags ?? {}, [record.tags]);
  const entries = useMemo(() => Object.entries(tags), [tags]);

  // Per-cell promise queue: chains each `saveTagsUpdate` onto the previous one so the
  // cache read for the next mutation always runs *after* the prior mutation's `onMutate`
  // has written its optimistic state. Without serialization, two rapid pill-close clicks
  // would each read the same pre-mutation cache and the second mutation's `nextTags` would
  // re-introduce the tag the first mutation just removed.
  const pendingQueueRef = useRef<Promise<unknown>>(Promise.resolve());

  const saveTagsUpdate = useCallback(
    (computeNext: (current: Tags) => Tags): Promise<void> => {
      const next = pendingQueueRef.current
        .catch(() => {
          // A failed predecessor must not poison the queue — subsequent edits still run.
        })
        .then(async () => {
          const current = readLatestTags(queryClient, datasetId, record.dataset_record_id, tags);
          const nextTags = computeNext(current);
          try {
            await upsertMutation.mutateAsync([
              {
                recordId: record.dataset_record_id,
                updates: { tags: nextTags },
                updateMask: { tags: nextTags },
              },
            ]);
          } catch (err) {
            // Optimistic rollback is handled by the mutation's onError; this surfaces the
            // failure to the user. We rethrow so the modal stays open and the caller can react.
            onSaveError?.(err);
            throw err;
          }
        });
      // Swallow rejections in the cached chain so a failed earlier write doesn't surface
      // as an unhandled rejection in the next caller; we still return the unswallowed
      // promise to the immediate caller for its own error handling.
      pendingQueueRef.current = next.catch(() => undefined);
      return next;
    },
    [queryClient, datasetId, record.dataset_record_id, tags, upsertMutation, onSaveError],
  );

  const handleSaveTag = useCallback(
    async (key: string, value: string): Promise<void> => {
      await saveTagsUpdate((current) => {
        const next = { ...current };
        // Editing renames a key — drop the old one if needed.
        if (modalState.key && modalState.key !== key) {
          delete next[modalState.key];
        }
        next[key] = value;
        return next;
      });
    },
    [saveTagsUpdate, modalState.key],
  );

  const handleDeleteTag = useCallback(
    async (key: string): Promise<void> => {
      await saveTagsUpdate((current) => {
        const next = { ...current };
        delete next[key];
        return next;
      });
    },
    [saveTagsUpdate],
  );

  const handlePillClose = useCallback(
    // Close-icon click is a fast path for removing without opening the modal. The user has
    // no other surface for the error here, so we route it through onSaveError unconditionally.
    (key: string) => {
      saveTagsUpdate((current) => {
        const next = { ...current };
        delete next[key];
        return next;
      }).catch(() => {
        // saveTagsUpdate already notifies via onSaveError; swallow here so we don't surface
        // an unhandled rejection.
      });
    },
    [saveTagsUpdate],
  );

  return (
    <div
      css={{
        display: 'flex',
        flexWrap: 'wrap',
        alignItems: 'center',
        gap: theme.spacing.xs,
      }}
    >
      {entries.map(([key, value]) => (
        <Tooltip key={key} componentId="mlflow.eval-datasets-v2.records.tag.pill-tooltip" content={`${key}: ${value}`}>
          <Tag
            componentId="mlflow.eval-datasets-v2.records.tag.pill"
            color="default"
            closable
            onClose={() => handlePillClose(key)}
            onClick={() => setModalState({ open: true, key, value })}
            css={{ cursor: 'pointer', maxWidth: 220 }}
          >
            <Typography.Text ellipsis css={{ maxWidth: 180 }}>
              {key}: {value}
            </Typography.Text>
          </Tag>
        </Tooltip>
      ))}
      <Button
        componentId="mlflow.eval-datasets-v2.records.tag.add"
        size="small"
        type="tertiary"
        icon={<PlusIcon />}
        onClick={() => setModalState({ open: true, key: '', value: '' })}
        aria-label={intl.formatMessage({
          defaultMessage: 'Add tag',
          description: 'Aria label for the add-tag button in the V2 dataset records table',
        })}
      >
        {entries.length === 0 ? (
          <FormattedMessage
            defaultMessage="Add tag"
            description="Button text for adding a tag when no tags exist on a dataset record"
          />
        ) : null}
      </Button>
      <KeyValueTagFullViewModal
        tagKey={modalState.key}
        tagValue={modalState.value}
        isKeyValueTagFullViewModalVisible={modalState.open}
        setIsKeyValueTagFullViewModalVisible={(visible) => setModalState((prev) => ({ ...prev, open: visible }))}
        onSave={handleSaveTag}
        onDelete={modalState.key ? handleDeleteTag : undefined}
      />
    </div>
  );
};
