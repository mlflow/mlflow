import { useEffect, useMemo, useRef, useState } from 'react';

import { Empty, Modal, SearchIcon, TableSkeleton, useDesignSystemTheme } from '@databricks/design-system';
import { ModelTraceExplorerResizablePane, useGetTracesById } from '@databricks/web-shared/model-trace-explorer';
import { FormattedMessage, useIntl } from 'react-intl';

import { useListLabelSchemasQuery } from '../../components/label-schemas';
import { useParams } from '../../../common/utils/RoutingUtils';
import { useMlflowSidebar } from '../../../common/contexts/MlflowSidebarContext';
import { useIsAuthAvailable } from '../../../account/hooks';
import { CreateReviewQueueModal } from './CreateReviewQueueModal';
import { FocusedReview } from './FocusedReview';
import { ManageQuestionsModal } from './ManageQuestionsModal';
import { QueueSettingsModal } from './QueueSettingsModal';
import { ReviewQueueList } from './ReviewQueueList';
import { ReviewQueueSidebar } from './ReviewQueueSidebar';
import { useCanManageReviews } from './hooks/useCanManageReviews';
import { useDeleteReviewQueueMutation } from './hooks/useDeleteReviewQueueMutation';
import { useListReviewQueueTracesQuery } from './hooks/useListReviewQueueTracesQuery';
import { useListReviewQueuesQuery } from './hooks/useListReviewQueuesQuery';
import { useUpdateReviewQueueMutation } from './hooks/useUpdateReviewQueueMutation';
import { useRemoveTracesFromReviewQueueMutation } from './hooks/useRemoveTracesFromReviewQueueMutation';
import { displayUser, useReviewer } from './hooks/useReviewer';
import { useSetReviewQueueTraceStatusMutation } from './hooks/useSetReviewQueueTraceStatusMutation';
import { canDeleteQueue, canManageQueue, sameUser } from './queuePermissions';
import type { ReviewQueueItem, ReviewStatus } from './types';

/**
 * Review tab — a master/detail surface modeled on the labeling-session page:
 * the reviewer's queues on the left, the selected queue's traces on the right.
 * Clicking a trace opens the full-page focused question-answering view (the
 * queue list collapses), with a "Back" control to return to the list.
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
  // The queue open in "Manage queue" (questions + members), from the right-pane gear.
  const [editingQueueId, setEditingQueueId] = useState<string>();
  // The queue pending delete confirmation, from the right-pane gear.
  const [confirmDeleteQueueId, setConfirmDeleteQueueId] = useState<string>();
  const [paneWidth, setPaneWidth] = useState(320);

  const { reviewQueues, isLoading: queuesLoading } = useListReviewQueuesQuery({
    experimentId: experimentId ?? '',
    // Don't scope by reviewer: a manager must see every queue, and the server's
    // visibility filter (`filter_list_review_queues`) narrows the list to assigned
    // queues for non-managers (admins / no-auth see all).
    // No-auth only: the server seeds the experiment's protected default queue
    // while listing. Authenticated MLflow has no default queue — reviewers use
    // their own user/custom queues.
    ensureDefault: !authAvailable,
  });
  const { labelSchemas } = useListLabelSchemasQuery({ experimentId: experimentId ?? '' });
  const { setReviewQueueTraceStatusAsync, isSettingStatus } = useSetReviewQueueTraceStatusMutation();
  const { removeTracesFromReviewQueue, isRemovingTraces } = useRemoveTracesFromReviewQueueMutation();
  const { deleteReviewQueue } = useDeleteReviewQueueMutation();
  const { updateReviewQueueAsync, isUpdatingQueue } = useUpdateReviewQueueMutation();

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
  const confirmDeleteQueue = useMemo(
    () => (confirmDeleteQueueId ? (reviewQueues.find((q) => q.queue_id === confirmDeleteQueueId) ?? null) : null),
    [reviewQueues, confirmDeleteQueueId],
  );
  // Whether the reviewer may manage the selected queue (settings / remove traces)
  // — any CUSTOM queue when they can manage reviews (MANAGE).
  const canManageSelectedQueue = selectedQueue ? canManageQueue(selectedQueue, canManage) : false;
  // Whether the right-pane gear (manage settings / delete) shows — editable
  // non-default custom queues only (never USER queues or the default queue).
  const canEditSelectedQueue = selectedQueue ? canDeleteQueue(selectedQueue, canManage) : false;
  // Whether the reviewer may submit reviews in the selected queue: always on a
  // no-auth server; otherwise only if they're in the queue's assigned-user pool
  // (the server enforces this on set-status). A manager viewing a queue they're
  // not assigned to gets a view-only pane with a self-assign affordance.
  const canReviewSelectedQueue =
    !authAvailable || (selectedQueue ? (selectedQueue.users ?? []).some((u) => sameUser(u, reviewer)) : false);
  const handleAssignSelf =
    authAvailable &&
    canManage &&
    !canReviewSelectedQueue &&
    selectedQueue &&
    selectedQueue.queue_type === 'CUSTOM' &&
    !selectedQueue.is_default
      ? () => {
          void updateReviewQueueAsync({
            queue_id: selectedQueue.queue_id,
            users: [...(selectedQueue.users ?? []), reviewer],
          });
        }
      : undefined;

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
  // Each trace's own creation time, to order the queue by when traces were
  // produced (not when they were added to the queue). Shares the cache with the
  // trace-list preview fetch.
  const { data: orderingTraceData } = useGetTracesById(traces.map((t) => t.target_id));
  const traceCreatedMsById = useMemo(() => {
    const map = new Map<string, number>();
    (orderingTraceData ?? []).forEach((t) => {
      const id = t?.info?.trace_id;
      const ms = t?.info?.request_time ? Date.parse(t.info.request_time) : NaN;
      if (id && !Number.isNaN(ms)) {
        map.set(id, ms);
      }
    });
    return map;
  }, [orderingTraceData]);
  // Queue order for review: completed (terminal) traces first, then the to-do
  // (pending) traces, each by trace creation time (newest first). This puts the
  // to-do traces contiguous at the end so "Start review" lands on the first
  // to-do and Next walks the rest through to the end. The same order drives both
  // the list display and the focused view's prev/next.
  const orderedTraces = useMemo(() => {
    const byTraceNewest = (a: ReviewQueueItem, b: ReviewQueueItem) =>
      (traceCreatedMsById.get(b.target_id) ?? 0) - (traceCreatedMsById.get(a.target_id) ?? 0);
    const done = traces.filter((t) => t.status !== 'PENDING').sort(byTraceNewest);
    const todo = traces.filter((t) => t.status === 'PENDING').sort(byTraceNewest);
    return [...done, ...todo];
  }, [traces, traceCreatedMsById]);

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

  // Focus mode (a trace open) dedicates the full page to the review: the queue
  // list (left pane) collapses below, and the app-shell sidebar collapses too.
  // We save the sidebar's state on entering and restore it on exit/unmount, so a
  // sidebar the user had already collapsed stays collapsed when they go back.
  const inFocusMode = Boolean(openTargetId && openItem);
  const sidebar = useMlflowSidebar();
  // Read the sidebar context through a ref so the effect can collapse/restore it
  // without listing `sidebar` as a dependency — `setShowSidebar` changes the
  // context value on every toggle, and depending on it would re-run the effect
  // mid-focus-mode and fight its own write. The effect keys only on `inFocusMode`;
  // the cleanup restores the saved state on exit and on unmount.
  const sidebarRef = useRef(sidebar);
  sidebarRef.current = sidebar;
  const savedSidebarRef = useRef<boolean | null>(null);
  useEffect(() => {
    const current = sidebarRef.current;
    if (!current) {
      return undefined;
    }
    if (inFocusMode && savedSidebarRef.current === null) {
      savedSidebarRef.current = current.showSidebar;
      current.setShowSidebar(false);
    }
    return () => {
      const latest = sidebarRef.current;
      if (latest && savedSidebarRef.current !== null) {
        latest.setShowSidebar(savedSidebarRef.current);
        savedSidebarRef.current = null;
      }
    };
  }, [inFocusMode]);

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
        items={orderedTraces}
        schemas={questionSchemas}
        completedBy={reviewer}
        isSettingStatus={isSettingStatus}
        canReview={canReviewSelectedQueue}
        onAssignSelf={handleAssignSelf}
        isAssigningSelf={isUpdatingQueue}
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
        questionCount={questionSchemas.length}
        items={orderedTraces}
        onOpen={(item) => setOpenTargetId(item.target_id)}
        nowMs={nowMs}
        latestQuestionCreatedAtMs={latestQuestionCreatedAtMs}
        onRemoveTraces={
          canManageSelectedQueue
            ? (targetIds) => removeTracesFromReviewQueue({ queue_id: selectedQueue.queue_id, target_ids: targetIds })
            : undefined
        }
        isRemovingTraces={isRemovingTraces}
        // Gear menu (manage / delete) only for editable non-default custom queues.
        onManageQueue={canEditSelectedQueue ? () => setEditingQueueId(selectedQueue.queue_id) : undefined}
        onDeleteQueue={canEditSelectedQueue ? () => setConfirmDeleteQueueId(selectedQueue.queue_id) : undefined}
      />
    );
  }

  return (
    <div css={{ display: 'flex', flexDirection: 'column', height: '100%', minHeight: 0, padding: theme.spacing.md }}>
      <div css={{ display: 'flex', flex: 1, minHeight: 0, overflow: 'hidden' }}>
        {inFocusMode ? (
          <div
            css={{
              width: '100%',
              height: '100%',
              minHeight: 0,
              overflow: 'hidden',
              display: 'flex',
              flexDirection: 'column',
            }}
          >
            {rightContent}
          </div>
        ) : (
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
                  canManage={canManage}
                  onSelect={selectQueue}
                  onDeselectQueue={() => {
                    setSelectedQueueIdState(undefined);
                    setOpenTargetId(null);
                  }}
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
        )}
      </div>

      {manageOpen && experimentId && (
        <ManageQuestionsModal experimentId={experimentId} onClose={() => setManageOpen(false)} />
      )}

      {createOpen && experimentId && (
        <CreateReviewQueueModal experimentId={experimentId} onClose={() => setCreateOpen(false)} />
      )}

      {editingQueue && <QueueSettingsModal queue={editingQueue} onClose={() => setEditingQueueId(undefined)} />}

      {confirmDeleteQueue && (
        <Modal
          componentId="mlflow.experiment-review-queue.delete-queue-confirm"
          visible
          title={<FormattedMessage defaultMessage="Delete queue?" description="Delete review queue: confirm title" />}
          okText={<FormattedMessage defaultMessage="Delete" description="Delete review queue: confirm button" />}
          okButtonProps={{ danger: true }}
          cancelText={<FormattedMessage defaultMessage="Cancel" description="Delete review queue: cancel button" />}
          onOk={() => {
            handleDeleteQueue(confirmDeleteQueue.queue_id);
            setConfirmDeleteQueueId(undefined);
          }}
          onCancel={() => setConfirmDeleteQueueId(undefined)}
        >
          <FormattedMessage
            defaultMessage='Permanently delete "{name}" and remove its traces from review? This cannot be undone.'
            description="Delete review queue: confirm body"
            values={{ name: confirmDeleteQueue.name }}
          />
        </Modal>
      )}
    </div>
  );
};

export default ExperimentReviewQueuePage;
