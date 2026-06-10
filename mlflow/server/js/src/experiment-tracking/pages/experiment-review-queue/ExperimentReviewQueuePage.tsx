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
import { useGetOrCreateUserQueueMutation } from './hooks/useGetOrCreateUserQueueMutation';
import { useListReviewQueueItemsQuery } from './hooks/useListReviewQueueItemsQuery';
import { useListReviewQueuesQuery } from './hooks/useListReviewQueuesQuery';
import { useRemoveItemsFromReviewQueueMutation } from './hooks/useRemoveItemsFromReviewQueueMutation';
import { DEFAULT_REVIEWER, displayUser, useIsReviewerResolved, useReviewer } from './hooks/useReviewer';
import { useSetReviewQueueItemStatusMutation } from './hooks/useSetReviewQueueItemStatusMutation';
import { canManageQueue } from './queuePermissions';
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
  // Completion stamps `completed_by` with the reviewer, so block it until the
  // identity is settled (an in-flight /users/current load reads as `default`).
  const reviewerResolved = useIsReviewerResolved();
  const authAvailable = useIsAuthAvailable();
  // Gate management controls (create / edit / delete queue, edit questions) on
  // EDIT+; reviewing stays available to everyone assigned. See useCanManageReviews.
  const canManage = useCanManageReviews(experimentId ?? '');

  const [selectedQueueIdState, setSelectedQueueIdState] = useState<string>();
  // The trace open in focused review (null = show the queue's trace table).
  const [openItemId, setOpenItemId] = useState<string | null>(null);
  const [manageOpen, setManageOpen] = useState(false);
  const [createOpen, setCreateOpen] = useState(false);
  // The queue open in "Manage queue" (questions + members), from the right-pane gear.
  const [editingQueueId, setEditingQueueId] = useState<string>();
  // The queue pending delete confirmation, from the right-pane gear.
  const [confirmDeleteQueueId, setConfirmDeleteQueueId] = useState<string>();
  const [paneWidth, setPaneWidth] = useState(320);

  const { reviewQueues, isLoading: queuesLoading } = useListReviewQueuesQuery({
    experimentId: experimentId ?? '',
    // Scope to the current reviewer (the reserved `default` user on a no-auth server).
    user: reviewer,
  });
  const { labelSchemas } = useListLabelSchemasQuery({ experimentId: experimentId ?? '' });
  const { setReviewQueueItemStatusAsync, isSettingStatus } = useSetReviewQueueItemStatusMutation();
  const { removeItemsFromReviewQueue, isRemovingItems } = useRemoveItemsFromReviewQueueMutation();
  const { deleteReviewQueue } = useDeleteReviewQueueMutation();
  const { getOrCreateUserQueueAsync } = useGetOrCreateUserQueueMutation();

  // No-auth catch-all: the reviewer's reserved `default` user queue. Ensure it
  // once on load (idempotent) so it appears in the sidebar before any item is
  // flagged. Best-effort — the mutation invalidates the queue list on success,
  // so the sidebar refreshes on its own; a failure just means the default queue
  // appears once the first item is flagged into it (the modal re-resolves it
  // and surfaces any error there). Authenticated MLflow has no default queue.
  useEffect(() => {
    if (!authAvailable && experimentId) {
      getOrCreateUserQueueAsync({ experiment_id: experimentId, user: DEFAULT_REVIEWER, created_by: reviewer }).catch(
        () => {},
      );
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [authAvailable, experimentId]);

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
  // Whether the reviewer may manage the selected queue — removing traces and
  // the right-pane gear (manage settings / delete) share one permission: a
  // CUSTOM queue they created, or any on a no-auth server (never USER queues).
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
            setOpenItemId(null);
          }
        },
      },
    );

  const { items: traces, isLoading: itemsLoading } = useListReviewQueueItemsQuery({
    queueId: selectedQueueId ?? '',
    enabled: Boolean(selectedQueueId),
  });
  // Each trace's own creation time, to order the queue by when traces were
  // produced (not when they were added to the queue). Shares the cache with the
  // trace-list preview fetch.
  const { data: orderingTraceData } = useGetTracesById(traces.map((t) => t.item_id));
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
      (traceCreatedMsById.get(b.item_id) ?? 0) - (traceCreatedMsById.get(a.item_id) ?? 0);
    const done = traces.filter((t) => t.status !== 'PENDING').sort(byTraceNewest);
    const todo = traces.filter((t) => t.status === 'PENDING').sort(byTraceNewest);
    return [...done, ...todo];
  }, [traces, traceCreatedMsById]);

  // A user queue inherits all of the experiment's schemas; a custom queue uses
  // its explicit subset.
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
  const latestQuestionCreatedAtMs = questionSchemas.reduce((max, s) => Math.max(max, s.created_at ?? 0), 0);

  const openItem = useMemo(() => traces.find((t) => t.item_id === openItemId) ?? null, [traces, openItemId]);
  const nowMs = Date.now();

  // Focus mode (a trace open) dedicates the full page to the review: the queue
  // list (left pane) collapses below, and the app-shell sidebar collapses too.
  // We save the sidebar's state on entering and restore it on exit/unmount, so a
  // sidebar the user had already collapsed stays collapsed when they go back.
  const inFocusMode = Boolean(openItemId && openItem);
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
    setOpenItemId(null);
  };

  const setOpenStatus = async (status: ReviewStatus) => {
    if (!selectedQueueId || !openItem) {
      return;
    }
    await setReviewQueueItemStatusAsync({
      queue_id: selectedQueueId,
      item_id: openItem.item_id,
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
  } else if (openItemId && openItem) {
    rightContent = (
      <FocusedReview
        // Remount per trace so answer state never bleeds across traces.
        key={openItem.item_id}
        item={openItem}
        items={orderedTraces}
        schemas={questionSchemas}
        completedBy={reviewer}
        // Treat an unresolved reviewer like an in-flight write so the complete /
        // decline controls stay disabled until `completed_by` is trustworthy.
        isSettingStatus={isSettingStatus || !reviewerResolved}
        onBack={() => setOpenItemId(null)}
        onSelect={(itemId) => setOpenItemId(itemId)}
        onSetStatus={setOpenStatus}
      />
    );
  } else if (itemsLoading) {
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
        onOpen={(item) => setOpenItemId(item.item_id)}
        nowMs={nowMs}
        latestQuestionCreatedAtMs={latestQuestionCreatedAtMs}
        onRemoveItems={
          canManageSelectedQueue
            ? (itemIds) => removeItemsFromReviewQueue({ queue_id: selectedQueue.queue_id, item_ids: itemIds })
            : undefined
        }
        isRemovingItems={isRemovingItems}
        // Gear menu (manage / delete) only for editable non-default custom queues.
        onManageQueue={canManageSelectedQueue ? () => setEditingQueueId(selectedQueue.queue_id) : undefined}
        onDeleteQueue={canManageSelectedQueue ? () => setConfirmDeleteQueueId(selectedQueue.queue_id) : undefined}
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
                  reviewer={reviewer}
                  authAvailable={authAvailable}
                  canManage={canManage}
                  onSelect={selectQueue}
                  onDeselectQueue={() => {
                    setSelectedQueueIdState(undefined);
                    setOpenItemId(null);
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
