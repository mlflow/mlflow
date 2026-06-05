import { useMemo, useState } from 'react';

import {
  Alert,
  Button,
  Empty,
  GearIcon,
  Modal,
  PlusIcon,
  SearchIcon,
  TableSkeleton,
  Tooltip,
  TrashIcon,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

import { useListLabelSchemasQuery } from '../../components/label-schemas';
import { useParams } from '../../../common/utils/RoutingUtils';
import { CreateReviewQueueModal } from './CreateReviewQueueModal';
import { FocusedReview } from './FocusedReview';
import { ManageQuestionsModal } from './ManageQuestionsModal';
import { ReviewQueueList } from './ReviewQueueList';
import { useDeleteReviewQueueMutation } from './hooks/useDeleteReviewQueueMutation';
import { useListReviewQueueTracesQuery } from './hooks/useListReviewQueueTracesQuery';
import { useListReviewQueuesQuery } from './hooks/useListReviewQueuesQuery';
import { useReviewer } from './hooks/useReviewer';
import { useSetReviewQueueTraceStatusMutation } from './hooks/useSetReviewQueueTraceStatusMutation';
import type { ReviewQueue, ReviewStatus } from './types';

const CID = 'mlflow.experiment-review-queue.page';

/**
 * Review tab — a reviewer works a queue's traces and answers its questions.
 *
 * Lists the experiment's review queues, shows the selected queue's traces in
 * a sortable table, and opens a focused 3-panel review (queue rail | trace |
 * question widgets). Answering writes Feedback/Expectation assessments; the
 * complete / decline / reopen actions drive the shared-pool status.
 */
const ExperimentReviewQueuePage = () => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { experimentId } = useParams<{ experimentId: string }>();
  const reviewer = useReviewer();

  const [selectedQueueId, setSelectedQueueId] = useState<string | null>(null);
  const [openTargetId, setOpenTargetId] = useState<string | null>(null);
  const [manageOpen, setManageOpen] = useState(false);
  const [createOpen, setCreateOpen] = useState(false);
  // The custom queue pending deletion confirmation.
  const [pendingDelete, setPendingDelete] = useState<ReviewQueue | null>(null);

  const { reviewQueues, isLoading: queuesLoading } = useListReviewQueuesQuery({
    experimentId: experimentId ?? '',
    // Scope to the current reviewer so the rail shows their queues (and the
    // single default queue on a no-auth server), not everyone's.
    user: reviewer,
  });
  const { labelSchemas } = useListLabelSchemasQuery({ experimentId: experimentId ?? '' });
  const { setReviewQueueTraceStatusAsync, isSettingStatus } = useSetReviewQueueTraceStatusMutation();
  const { deleteReviewQueue, isDeletingQueue, error: deleteError, reset: resetDelete } = useDeleteReviewQueueMutation();

  // Clear any prior delete error and open the confirm dialog for `queue`.
  const promptDelete = (queue: ReviewQueue) => {
    resetDelete();
    setPendingDelete(queue);
  };

  const cancelDelete = () => {
    resetDelete();
    setPendingDelete(null);
  };

  const selectedQueue = useMemo(
    () => reviewQueues.find((q) => q.queue_id === selectedQueueId) ?? reviewQueues[0] ?? null,
    [reviewQueues, selectedQueueId],
  );

  const { items, isLoading: tracesLoading } = useListReviewQueueTracesQuery({
    queueId: selectedQueue?.queue_id ?? '',
    enabled: Boolean(selectedQueue),
  });

  // A user queue inherits all of the experiment's schemas; a custom queue
  // uses its explicit subset.
  const questionSchemas = useMemo(() => {
    if (!selectedQueue) {
      return [];
    }
    if (selectedQueue.queue_type === 'USER') {
      return labelSchemas;
    }
    const ids = new Set(selectedQueue.schema_ids ?? []);
    return labelSchemas.filter((s) => ids.has(s.schema_id));
  }, [selectedQueue, labelSchemas]);

  const openItem = useMemo(() => items.find((i) => i.target_id === openTargetId) ?? null, [items, openTargetId]);

  const nowMs = Date.now();

  const setOpenStatus = async (status: ReviewStatus) => {
    if (!selectedQueue || !openItem) {
      return;
    }
    await setReviewQueueTraceStatusAsync({
      queue_id: selectedQueue.queue_id,
      target_id: openItem.target_id,
      status,
      // Attribution only applies to the terminal states; reopen clears it.
      completed_by: status === 'PENDING' ? undefined : reviewer,
    });
  };

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.md,
        height: '100%',
        padding: theme.spacing.md,
      }}
    >
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
        <Typography.Title level={2} withoutMargins>
          <FormattedMessage defaultMessage="Review" description="Review queue tab title" />
        </Typography.Title>
        <Tooltip
          componentId={`${CID}.edit-questions-tooltip`}
          content={intl.formatMessage({
            defaultMessage: 'Edit review questions for this experiment',
            description: 'Review queue: edit-questions gear tooltip',
          })}
        >
          <Button
            componentId={`${CID}.edit-questions`}
            icon={<GearIcon />}
            aria-label={intl.formatMessage({
              defaultMessage: 'Edit review questions',
              description: 'Review queue: edit-questions gear aria label',
            })}
            onClick={() => setManageOpen(true)}
          />
        </Tooltip>
        <div css={{ flex: 1 }} />
        <Button componentId={`${CID}.new-queue`} icon={<PlusIcon />} onClick={() => setCreateOpen(true)}>
          <FormattedMessage defaultMessage="New queue" description="Review queue: create-queue button" />
        </Button>
      </div>

      {queuesLoading ? (
        <TableSkeleton lines={5} />
      ) : reviewQueues.length === 0 ? (
        <div
          css={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            height: '100%',
            minHeight: 400,
            width: '100%',
            '& > div': { height: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'center' },
          }}
        >
          <Empty
            description={
              <FormattedMessage
                defaultMessage="No review queues yet. Flag traces for review to create one."
                description="Review queue: empty state when no queues exist"
              />
            }
            image={<SearchIcon />}
          />
        </div>
      ) : (
        <>
          {!openItem && (
            <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
              <Typography.Text color="secondary">
                <FormattedMessage defaultMessage="Queue:" description="Review queue: queue selector label" />
              </Typography.Text>
              {reviewQueues.map((q) => (
                <div key={q.queue_id} css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
                  <Button
                    componentId={`${CID}.select-queue`}
                    size="small"
                    type={q.queue_id === selectedQueue?.queue_id ? 'primary' : undefined}
                    onClick={() => {
                      setSelectedQueueId(q.queue_id);
                      setOpenTargetId(null);
                    }}
                  >
                    {q.name}
                  </Button>
                  {/* Personal USER queues stay put; only custom queues are deletable here. */}
                  {q.queue_type === 'CUSTOM' && (
                    <Button
                      componentId={`${CID}.delete-queue`}
                      size="small"
                      icon={<TrashIcon />}
                      aria-label={intl.formatMessage({
                        defaultMessage: 'Delete queue',
                        description: 'Review queue: delete-queue button aria label',
                      })}
                      onClick={() => promptDelete(q)}
                    />
                  )}
                </div>
              ))}
            </div>
          )}

          <div css={{ flex: 1, minHeight: 0, overflow: 'auto' }}>
            {openItem ? (
              <FocusedReview
                // Remount per trace so answer state never bleeds across traces.
                key={openItem.target_id}
                item={openItem}
                items={items}
                schemas={questionSchemas}
                completedBy={reviewer}
                isSettingStatus={isSettingStatus}
                onBack={() => setOpenTargetId(null)}
                onSelect={setOpenTargetId}
                onSetStatus={setOpenStatus}
              />
            ) : tracesLoading ? (
              <TableSkeleton lines={5} />
            ) : (
              <ReviewQueueList items={items} onOpen={(item) => setOpenTargetId(item.target_id)} nowMs={nowMs} />
            )}
          </div>
        </>
      )}

      {manageOpen && experimentId && (
        <ManageQuestionsModal experimentId={experimentId} onClose={() => setManageOpen(false)} />
      )}

      {createOpen && experimentId && (
        <CreateReviewQueueModal
          experimentId={experimentId}
          onClose={() => setCreateOpen(false)}
          onCreated={(queue) => setSelectedQueueId(queue.queue_id)}
        />
      )}

      {pendingDelete && (
        <Modal
          componentId={`${CID}.delete-queue-confirm`}
          visible
          title={
            <FormattedMessage defaultMessage="Delete queue?" description="Review queue: delete confirmation title" />
          }
          okText={<FormattedMessage defaultMessage="Delete" description="Review queue: confirm delete" />}
          okButtonProps={{ danger: true, loading: isDeletingQueue }}
          cancelText={<FormattedMessage defaultMessage="Cancel" description="Review queue: cancel delete" />}
          onCancel={cancelDelete}
          onOk={() =>
            deleteReviewQueue(
              { queue_id: pendingDelete.queue_id },
              {
                // Close only on success; on failure keep the dialog open and
                // surface the error below so the action isn't silently lost.
                onSuccess: () => {
                  // Drop the selection if we just deleted the active queue so
                  // the page falls back to the first remaining queue.
                  if (selectedQueueId === pendingDelete.queue_id) {
                    setSelectedQueueId(null);
                    setOpenTargetId(null);
                  }
                  setPendingDelete(null);
                },
              },
            )
          }
        >
          <FormattedMessage
            defaultMessage='Deleting "{name}" removes the queue and its trace assignments. The traces and their assessments are not deleted. This cannot be undone.'
            description="Review queue: delete confirmation body"
            values={{ name: pendingDelete.name }}
          />
          {deleteError && (
            <Alert
              componentId={`${CID}.delete-queue-error`}
              type="error"
              closable={false}
              css={{ marginTop: theme.spacing.sm }}
              message={intl.formatMessage({
                defaultMessage: 'Failed to delete the queue.',
                description: 'Review queue: delete error alert title',
              })}
              description={deleteError.message}
            />
          )}
        </Modal>
      )}
    </div>
  );
};

export default ExperimentReviewQueuePage;
