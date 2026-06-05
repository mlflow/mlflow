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
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

import { useListLabelSchemasQuery } from '../../components/label-schemas';
import { useParams } from '../../../common/utils/RoutingUtils';
import { CreateReviewQueueModal } from './CreateReviewQueueModal';
import { FocusedReview } from './FocusedReview';
import { ManageQuestionsModal } from './ManageQuestionsModal';
import { ReviewQueueSection } from './ReviewQueueSection';
import { useDeleteReviewQueueMutation } from './hooks/useDeleteReviewQueueMutation';
import { useListReviewQueueTracesQuery } from './hooks/useListReviewQueueTracesQuery';
import { useListReviewQueuesQuery } from './hooks/useListReviewQueuesQuery';
import { useReviewer } from './hooks/useReviewer';
import { useSetReviewQueueTraceStatusMutation } from './hooks/useSetReviewQueueTraceStatusMutation';
import type { ReviewQueue, ReviewStatus } from './types';

const CID = 'mlflow.experiment-review-queue.page';

/**
 * Review tab — a reviewer works their queues' traces and answers the questions.
 *
 * Each queue the reviewer is assigned to renders as a collapsible section over
 * its own trace table, so several queues show at once and can be collapsed to
 * focus on one. Opening a trace takes over with a focused 3-panel review (queue
 * rail | trace | question widgets); answering writes Feedback/Expectation
 * assessments and the complete / decline / reopen actions drive the
 * shared-pool status.
 */
const ExperimentReviewQueuePage = () => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { experimentId } = useParams<{ experimentId: string }>();
  const reviewer = useReviewer();

  // Collapsed (not expanded) queue ids. Empty == every section expanded.
  const [collapsedIds, setCollapsedIds] = useState<Set<string>>(new Set());
  // The trace currently open in focused review, with the queue it belongs to.
  const [openTrace, setOpenTrace] = useState<{ queueId: string; targetId: string } | null>(null);
  const [manageOpen, setManageOpen] = useState(false);
  const [createOpen, setCreateOpen] = useState(false);
  // The custom queue pending deletion confirmation.
  const [pendingDelete, setPendingDelete] = useState<ReviewQueue | null>(null);

  const { reviewQueues, isLoading: queuesLoading } = useListReviewQueuesQuery({
    experimentId: experimentId ?? '',
    // Scope to the current reviewer so the tab shows their queues (and the
    // single default queue on a no-auth server), not everyone's.
    user: reviewer,
  });
  const { labelSchemas } = useListLabelSchemasQuery({ experimentId: experimentId ?? '' });
  const { setReviewQueueTraceStatusAsync, isSettingStatus } = useSetReviewQueueTraceStatusMutation();
  const { deleteReviewQueue, isDeletingQueue, error: deleteError, reset: resetDelete } = useDeleteReviewQueueMutation();

  // Traces for the queue whose trace is open in focused review. (Each section
  // fetches its own list for the collapsed view; this drives the takeover.)
  const { items: openItems, isLoading: openItemsLoading } = useListReviewQueueTracesQuery({
    queueId: openTrace?.queueId ?? '',
    enabled: Boolean(openTrace),
  });

  const openQueue = useMemo(
    () => reviewQueues.find((q) => q.queue_id === openTrace?.queueId) ?? null,
    [reviewQueues, openTrace],
  );
  const openItem = useMemo(
    () => openItems.find((i) => i.target_id === openTrace?.targetId) ?? null,
    [openItems, openTrace],
  );

  // A user queue inherits all of the experiment's schemas; a custom queue uses
  // its explicit subset.
  const questionSchemas = useMemo(() => {
    if (!openQueue) {
      return [];
    }
    if (openQueue.queue_type === 'USER') {
      return labelSchemas;
    }
    const ids = new Set(openQueue.schema_ids ?? []);
    return labelSchemas.filter((s) => ids.has(s.schema_id));
  }, [openQueue, labelSchemas]);

  const nowMs = Date.now();

  const toggleQueue = (queueId: string) => {
    setCollapsedIds((prev) => {
      const next = new Set(prev);
      if (next.has(queueId)) {
        next.delete(queueId);
      } else {
        next.add(queueId);
      }
      return next;
    });
  };

  const promptDelete = (queue: ReviewQueue) => {
    resetDelete();
    setPendingDelete(queue);
  };

  const cancelDelete = () => {
    resetDelete();
    setPendingDelete(null);
  };

  const setOpenStatus = async (status: ReviewStatus) => {
    if (!openTrace || !openItem) {
      return;
    }
    await setReviewQueueTraceStatusAsync({
      queue_id: openTrace.queueId,
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
      ) : openTrace ? (
        <div css={{ flex: 1, minHeight: 0, overflow: 'auto' }}>
          {openItem ? (
            <FocusedReview
              // Remount per trace so answer state never bleeds across traces.
              key={openItem.target_id}
              item={openItem}
              items={openItems}
              schemas={questionSchemas}
              completedBy={reviewer}
              isSettingStatus={isSettingStatus}
              onBack={() => setOpenTrace(null)}
              onSelect={(targetId) => setOpenTrace({ queueId: openTrace.queueId, targetId })}
              onSetStatus={setOpenStatus}
            />
          ) : openItemsLoading ? (
            <TableSkeleton lines={5} />
          ) : null}
        </div>
      ) : (
        <div
          css={{
            flex: 1,
            minHeight: 0,
            overflow: 'auto',
            display: 'flex',
            flexDirection: 'column',
            gap: theme.spacing.sm,
          }}
        >
          {reviewQueues.map((q) => (
            <ReviewQueueSection
              key={q.queue_id}
              queue={q}
              expanded={!collapsedIds.has(q.queue_id)}
              onToggle={() => toggleQueue(q.queue_id)}
              onOpenTrace={(item) => setOpenTrace({ queueId: q.queue_id, targetId: item.target_id })}
              onDelete={() => promptDelete(q)}
              nowMs={nowMs}
            />
          ))}
        </div>
      )}

      {manageOpen && experimentId && (
        <ManageQuestionsModal experimentId={experimentId} onClose={() => setManageOpen(false)} />
      )}

      {createOpen && experimentId && (
        <CreateReviewQueueModal experimentId={experimentId} onClose={() => setCreateOpen(false)} />
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
                  // Leave focused review if we just deleted the open trace's queue.
                  if (openTrace?.queueId === pendingDelete.queue_id) {
                    setOpenTrace(null);
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
