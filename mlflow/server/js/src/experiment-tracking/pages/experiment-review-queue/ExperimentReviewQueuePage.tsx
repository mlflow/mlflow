import { useEffect, useMemo, useState } from 'react';

import { Empty, SearchIcon, TableSkeleton, useDesignSystemTheme } from '@databricks/design-system';
import { ModelTraceExplorerResizablePane } from '@databricks/web-shared/model-trace-explorer';
import { FormattedMessage, useIntl } from 'react-intl';

import { LabelSchemaFormModal, useListLabelSchemasQuery } from '../../components/label-schemas';
import type { LabelSchema } from '../../components/label-schemas';
import { useParams } from '../../../common/utils/RoutingUtils';
import { useIsAuthAvailable } from '../../../account/hooks';
import { CreateReviewQueueModal } from './CreateReviewQueueModal';
import { EditReviewQueueModal } from './EditReviewQueueModal';
import { FocusedReview } from './FocusedReview';
import { ManageQuestionsModal } from './ManageQuestionsModal';
import { ReviewQueueList } from './ReviewQueueList';
import { ReviewQueueSidebar } from './ReviewQueueSidebar';
import { useCanManageReviews } from './hooks/useCanManageReviews';
import { useDeleteReviewQueueMutation } from './hooks/useDeleteReviewQueueMutation';
import { useGetOrCreateDefaultQueueMutation } from './hooks/useGetOrCreateDefaultQueueMutation';
import { useListReviewQueueTracesQuery } from './hooks/useListReviewQueueTracesQuery';
import { useListReviewQueuesQuery } from './hooks/useListReviewQueuesQuery';
import { useRemoveTracesFromReviewQueueMutation } from './hooks/useRemoveTracesFromReviewQueueMutation';
import { displayUser, useReviewer } from './hooks/useReviewer';
import { useSetReviewQueueTraceStatusMutation } from './hooks/useSetReviewQueueTraceStatusMutation';
import { canManageQueue } from './queuePermissions';
import type { ReviewStatus } from './types';

/**
 * Review tab — a master/detail surface modeled on the labeling-session page:
 * the reviewer's queues on the left, the selected queue's traces on the right.
 * Clicking a trace swaps the right panel to the focused question-answering view
 * (with a "Back to traces" control); the left queue list stays put.
 *
 * The left list is grouped on authenticated servers (queues others assigned to
 * me vs. queues I created); a no-auth server shows one list. See
 * `ReviewQueueSidebar`.
 */
const ExperimentReviewQueuePage = () => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { experimentId } = useParams<{ experimentId: string }>();
  const reviewer = useReviewer();
  const authAvailable = useIsAuthAvailable();
  // Gate management controls (create / edit / delete queue, edit questions) on
  // EDIT+; reviewing stays available to everyone assigned. See useCanManageReviews.
  const canManage = useCanManageReviews(experimentId ?? '');

  const [selectedQueueIdState, setSelectedQueueIdState] = useState<string>();
  // The trace open in focused review (null = show the queue's trace table).
  const [openTargetId, setOpenTargetId] = useState<string | null>(null);
  const [manageOpen, setManageOpen] = useState(false);
  const [createOpen, setCreateOpen] = useState(false);
  // A question (label schema) being edited from the sidebar's questions list.
  const [editingQuestion, setEditingQuestion] = useState<LabelSchema | null>(null);
  // A queue being edited (members / delete) from the sidebar gear.
  const [editingQueueId, setEditingQueueId] = useState<string>();
  const [paneWidth, setPaneWidth] = useState(320);

  const { reviewQueues, isLoading: queuesLoading } = useListReviewQueuesQuery({
    experimentId: experimentId ?? '',
    // Scope to the current reviewer (the single default queue on a no-auth server).
    user: reviewer,
  });
  const { labelSchemas } = useListLabelSchemasQuery({ experimentId: experimentId ?? '' });
  const { setReviewQueueTraceStatusAsync, isSettingStatus } = useSetReviewQueueTraceStatusMutation();
  const { removeTracesFromReviewQueue, isRemovingTraces } = useRemoveTracesFromReviewQueueMutation();
  const { deleteReviewQueue } = useDeleteReviewQueueMutation();
  const { getOrCreateDefaultQueueAsync, isResolvingDefaultQueue } = useGetOrCreateDefaultQueueMutation();

  // The default queue is a no-auth-only catch-all; ensure it exists when the
  // Review tab loads (idempotent, so a repeat call is a no-op). Authenticated
  // MLflow has no default queue — reviewers use their own user/custom queues.
  useEffect(() => {
    if (authAvailable || queuesLoading || isResolvingDefaultQueue || !experimentId) {
      return;
    }
    if (!reviewQueues.some((q) => q.is_default)) {
      getOrCreateDefaultQueueAsync({ experiment_id: experimentId, created_by: reviewer }).catch(() => {
        // Non-fatal: the tab still works without it; the next load retries.
      });
    }
  }, [
    authAvailable,
    queuesLoading,
    isResolvingDefaultQueue,
    experimentId,
    reviewer,
    reviewQueues,
    getOrCreateDefaultQueueAsync,
  ]);

  // No queue is selected until the reviewer picks one (the right panel prompts
  // them to). Auto-selecting the first queue would land on a no-work queue and
  // force the "No work to do" group open.
  const selectedQueueId = selectedQueueIdState;
  const selectedQueue = useMemo(
    () => reviewQueues.find((q) => q.queue_id === selectedQueueId) ?? null,
    [reviewQueues, selectedQueueId],
  );
  const editingQueue = useMemo(
    () => (editingQueueId ? (reviewQueues.find((q) => q.queue_id === editingQueueId) ?? null) : null),
    [reviewQueues, editingQueueId],
  );
  // Whether the reviewer may manage the selected queue (remove traces) — a
  // CUSTOM queue they created, or any on a no-auth server.
  const canManageSelectedQueue = selectedQueue
    ? canManageQueue(selectedQueue, reviewer, authAvailable, canManage)
    : false;

  const handleDeleteQueue = (queueId: string) =>
    deleteReviewQueue(
      { queue_id: queueId },
      {
        onSuccess: () => {
          // Drop the selection if the queue that was open got deleted.
          if (selectedQueueId === queueId) {
            setSelectedQueueIdState(undefined);
            setOpenTargetId(null);
          }
        },
      },
    );

  const { items: traces, isLoading: tracesLoading } = useListReviewQueueTracesQuery({
    queueId: selectedQueueId ?? '',
    enabled: Boolean(selectedQueueId),
  });

  // A user queue (and the default queue) inherits all of the experiment's
  // schemas; any other custom queue uses its explicit subset.
  const questionSchemas = useMemo(() => {
    if (!selectedQueue) {
      return [];
    }
    if (selectedQueue.queue_type === 'USER' || selectedQueue.is_default) {
      return labelSchemas;
    }
    const ids = new Set(selectedQueue.schema_ids ?? []);
    return labelSchemas.filter((s) => ids.has(s.schema_id));
  }, [selectedQueue, labelSchemas]);
  const latestQuestionCreatedAtMs = questionSchemas.reduce((max, s) => Math.max(max, s.created_at ?? 0), 0);

  const openItem = useMemo(() => traces.find((t) => t.target_id === openTargetId) ?? null, [traces, openTargetId]);
  const nowMs = Date.now();

  const selectQueue = (queueId: string) => {
    setSelectedQueueIdState(queueId);
    setOpenTargetId(null);
  };

  const setOpenStatus = async (status: ReviewStatus) => {
    if (!selectedQueueId || !openItem) {
      return;
    }
    await setReviewQueueTraceStatusAsync({
      queue_id: selectedQueueId,
      target_id: openItem.target_id,
      status,
      // Attribution only applies to the terminal states; reopen clears it.
      completed_by: status === 'PENDING' ? undefined : reviewer,
    });
  };

  const centeredEmpty = (description: React.ReactNode) => (
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
      <Empty description={description} image={<SearchIcon />} />
    </div>
  );

  let rightContent: React.ReactNode;
  if (queuesLoading) {
    rightContent = <TableSkeleton lines={6} />;
  } else if (reviewQueues.length === 0) {
    rightContent = centeredEmpty(
      <FormattedMessage
        defaultMessage="No review queues yet. Flag traces for review to create one."
        description="Review queue: empty state when no queues exist"
      />,
    );
  } else if (!selectedQueue) {
    rightContent = centeredEmpty(
      <FormattedMessage
        defaultMessage="Select a queue to review its traces."
        description="Review queue: prompt to pick a queue"
      />,
    );
  } else if (openTargetId && openItem) {
    rightContent = (
      <FocusedReview
        // Remount per trace so answer state never bleeds across traces.
        key={openItem.target_id}
        item={openItem}
        items={traces}
        schemas={questionSchemas}
        completedBy={reviewer}
        isSettingStatus={isSettingStatus}
        onBack={() => setOpenTargetId(null)}
        onSelect={(targetId) => setOpenTargetId(targetId)}
        onSetStatus={setOpenStatus}
      />
    );
  } else if (tracesLoading) {
    rightContent = <TableSkeleton lines={5} />;
  } else {
    rightContent = (
      <ReviewQueueList
        // Remount per queue so the trace selection (and expanded/collapsed
        // groups) reset instead of leaking stale target ids across queues.
        key={selectedQueue.queue_id}
        title={selectedQueue.queue_type === 'USER' ? displayUser(selectedQueue.name, intl) : selectedQueue.name}
        items={traces}
        onOpen={(item) => setOpenTargetId(item.target_id)}
        nowMs={nowMs}
        latestQuestionCreatedAtMs={latestQuestionCreatedAtMs}
        onRemoveTraces={
          canManageSelectedQueue
            ? (targetIds) => removeTracesFromReviewQueue({ queue_id: selectedQueue.queue_id, target_ids: targetIds })
            : undefined
        }
        isRemovingTraces={isRemovingTraces}
      />
    );
  }

  return (
    <div css={{ display: 'flex', flexDirection: 'column', height: '100%', minHeight: 0, padding: theme.spacing.md }}>
      <div css={{ display: 'flex', flex: 1, minHeight: 0, overflow: 'hidden' }}>
        <ModelTraceExplorerResizablePane
          initialRatio={0.26}
          paneWidth={paneWidth}
          setPaneWidth={setPaneWidth}
          leftMinWidth={260}
          rightMinWidth={480}
          leftChild={
            <div css={{ width: '100%', height: '100%', minHeight: 0 }}>
              <ReviewQueueSidebar
                queues={reviewQueues}
                selectedQueueId={selectedQueueId}
                reviewer={reviewer}
                authAvailable={authAvailable}
                canManage={canManage}
                selectedQueueQuestions={questionSchemas}
                onSelect={selectQueue}
                onDeselectQueue={() => {
                  setSelectedQueueIdState(undefined);
                  setOpenTargetId(null);
                }}
                onDeleteQueue={handleDeleteQueue}
                onEditQueue={setEditingQueueId}
                onEditQuestion={canManage ? setEditingQuestion : undefined}
                onNewQueue={() => setCreateOpen(true)}
                onManageQuestions={() => setManageOpen(true)}
              />
            </div>
          }
          rightChild={
            <div
              css={{
                width: '100%',
                height: '100%',
                minHeight: 0,
                overflow: 'hidden',
                display: 'flex',
                flexDirection: 'column',
                paddingLeft: theme.spacing.md,
                borderLeft: `1px solid ${theme.colors.border}`,
              }}
            >
              {rightContent}
            </div>
          }
        />
      </div>

      {manageOpen && experimentId && (
        <ManageQuestionsModal experimentId={experimentId} onClose={() => setManageOpen(false)} />
      )}

      {createOpen && experimentId && (
        <CreateReviewQueueModal experimentId={experimentId} onClose={() => setCreateOpen(false)} />
      )}

      {editingQueue && (
        <EditReviewQueueModal
          queue={editingQueue}
          onClose={() => setEditingQueueId(undefined)}
          onDeleted={() => {
            if (selectedQueueId === editingQueue.queue_id) {
              setSelectedQueueIdState(undefined);
              setOpenTargetId(null);
            }
          }}
        />
      )}

      {editingQuestion && experimentId && (
        <LabelSchemaFormModal
          experimentId={experimentId}
          editingSchema={editingQuestion}
          visible
          onClose={() => setEditingQuestion(null)}
        />
      )}
    </div>
  );
};

export default ExperimentReviewQueuePage;
