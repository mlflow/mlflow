import { useMemo, useState } from 'react';

import {
  Alert,
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxEmpty,
  DialogComboboxOptionList,
  DialogComboboxOptionListCheckboxItem,
  DialogComboboxOptionListSearch,
  DialogComboboxSectionHeader,
  DialogComboboxTrigger,
  Modal,
  PlusIcon,
  TableSkeleton,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import type { ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';
import { FormattedMessage, useIntl } from 'react-intl';

import { useListLabelSchemasQuery } from '../../components/label-schemas';
import { useCurrentUserIsAdmin, useCurrentUserIsWorkspaceAdmin, useIsAuthAvailable } from '../../../account/hooks';
import Utils from '../../../common/utils/Utils';
import { generatePath, Link } from '../../../common/utils/RoutingUtils';
import { RoutePaths } from '../../routes';
import { CreateReviewQueueModal } from './CreateReviewQueueModal';
import { getQueueAssignability } from './queueAssignability';
import { sameUser } from './queuePermissions';
import { useAddItemsToReviewQueueMutation } from './hooks/useAddItemsToReviewQueueMutation';
import { useAssignableUsersQuery } from './hooks/useAssignableUsersQuery';
import { useGetOrCreateUserQueueMutation } from './hooks/useGetOrCreateUserQueueMutation';
import { useListReviewQueuesQuery } from './hooks/useListReviewQueuesQuery';
import { DEFAULT_REVIEWER, useIsReviewerResolved, useReviewer } from './hooks/useReviewer';
import type { ReviewQueue } from './types';

const CID = 'mlflow.experiment-review-queue.add-to-queue';

// Distinct, non-empty error messages from a batch of rejected promises, joined
// for display. May be empty if every rejection carried a blank message, so
// callers must decide whether the batch failed from the rejection count, not
// from this string.
const collectRejectionMessages = (rejections: PromiseRejectedResult[]): string =>
  Array.from(
    new Set(rejections.map((r) => (r.reason instanceof Error ? r.reason.message : String(r.reason))).filter(Boolean)),
  ).join('; ');

// Custom queues shown before the reviewer searches; the rest are reachable by
// name through the dropdown's search box.
const COLLAPSED_QUEUE_COUNT = 3;
// Cap on user matches surfaced per search so a large roster can't flood the list.
const MAX_USER_MATCHES = 20;

/**
 * Picker for routing one or more traces into review queues. Shown both from the
 * Traces table bulk action and the trace-detail "Flag for review" button
 * (injected into the shared trace UI via the same render-prop mechanism as
 * "Add to evaluation dataset").
 *
 * Destinations are multi-select via a searchable dropdown with two separate
 * sections: shared CUSTOM "Queues" (the first few shown up front, the rest found
 * by name) and, on an authenticated server where the caller can list users,
 * "Users" — each routing into that reviewer's personal queue, resolved
 * (get-or-create) on confirm. The experiment's default queue is pinned at the
 * top. Nothing is selected by default. Queues that wouldn't
 * present any questions are disabled (see `getQueueAssignability`), and a new
 * queue can be created inline.
 */
export const AddToReviewQueueModal = ({
  experimentId,
  visible,
  setVisible,
  selectedTraceInfos,
}: {
  experimentId: string;
  visible: boolean;
  setVisible: (visible: boolean) => void;
  selectedTraceInfos: ModelTraceInfoV3[];
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const reviewer = useReviewer();
  // Don't let a write stamp `created_by` until the reviewer identity is settled.
  const reviewerResolved = useIsReviewerResolved();
  const authAvailable = useIsAuthAvailable();
  const isAdmin = useCurrentUserIsAdmin();
  const isWorkspaceAdmin = useCurrentUserIsWorkspaceAdmin();
  // The user roster is workspace-admin gated server-side; only fetch it when the
  // caller can actually list users (and the modal is open).
  const canListUsers = authAvailable && (isAdmin || isWorkspaceAdmin);

  const [search, setSearch] = useState('');
  // The experiment's default queue. Nothing is selected by default; the caller
  // picks where to route the traces.
  const [defaultQueueSelected, setDefaultQueueSelected] = useState(false);
  const [selectedQueueIds, setSelectedQueueIds] = useState<Set<string>>(new Set());
  const [selectedUsers, setSelectedUsers] = useState<Set<string>>(new Set());
  const [createOpen, setCreateOpen] = useState(false);

  const {
    reviewQueues,
    isLoading: queuesLoading,
    error: queuesError,
  } = useListReviewQueuesQuery({ experimentId, enabled: visible });
  const { labelSchemas } = useListLabelSchemasQuery({ experimentId, enabled: visible });
  const { users, isLoading: usersLoading } = useAssignableUsersQuery({ enabled: visible && canListUsers });
  const { addItemsToReviewQueueAsync, isAddingItems, reset: resetAdd } = useAddItemsToReviewQueueMutation();
  const { getOrCreateUserQueueAsync, isResolvingUserQueue, reset: resetResolve } = useGetOrCreateUserQueueMutation();
  // Errors are tracked locally rather than read off the mutation hooks: a single
  // `useMutation` instance is reused across every per-destination call, so its
  // `error` slot only retains the last-settled call and a mid-batch failure
  // would be lost. `handleAdd` collects every failure from the batch instead.
  const [submitError, setSubmitError] = useState<string | null>(null);
  // Tracks the whole add operation locally. The mutation hooks' `isLoading`
  // reflects only their last-settled call (same shared-instance issue as their
  // `error` slot), so it can flip false mid-batch and momentarily re-enable
  // Add; this flag stays true for the entire batch to block a double-submit.
  const [isSubmitting, setIsSubmitting] = useState(false);

  const itemIds = useMemo(
    () => selectedTraceInfos.map((info) => info.trace_id).filter((id): id is string => Boolean(id)),
    [selectedTraceInfos],
  );

  // Shared queues anyone can route into. The no-auth catch-all (the reserved
  // `default` user queue) is surfaced through the pinned option instead of the
  // list; other users' personal queues are reached through the "Users" section.
  const customQueues = useMemo(() => reviewQueues.filter((q) => q.queue_type === 'CUSTOM'), [reviewQueues]);

  // USER queues (including the `default` catch-all) present every experiment
  // schema, so they are assignable as soon as the experiment has a question.
  const inheritAllAssignable = labelSchemas.length > 0;

  const assignabilityById = useMemo(() => {
    const map = new Map<string, ReturnType<typeof getQueueAssignability>>();
    customQueues.forEach((q) => map.set(q.queue_id, getQueueAssignability(q, labelSchemas)));
    return map;
  }, [customQueues, labelSchemas]);

  const query = search.trim().toLowerCase();

  // No search: show the first few custom queues, plus any selected ones (e.g. a
  // queue just created inline) so a checked queue is always visible. Searching:
  // filter the full set.
  const visibleQueues = useMemo(() => {
    if (query) {
      return customQueues.filter((q) => q.name.toLowerCase().includes(query));
    }
    const head = customQueues.slice(0, COLLAPSED_QUEUE_COUNT);
    const headIds = new Set(head.map((q) => q.queue_id));
    const selectedExtras = customQueues.filter((q) => selectedQueueIds.has(q.queue_id) && !headIds.has(q.queue_id));
    return [...head, ...selectedExtras];
  }, [customQueues, query, selectedQueueIds]);
  const hasMoreQueues = !query && customQueues.length > visibleQueues.length;

  // Users are search-driven; the experiment default queue is the pinned option.
  const visibleUsers = useMemo(() => {
    if (!query) {
      return [];
    }
    return users
      .filter((u) => !sameUser(u.username, reviewer) && u.username.toLowerCase().includes(query))
      .slice(0, MAX_USER_MATCHES);
  }, [users, query, reviewer]);

  // The default queue is a no-auth-only catch-all, so it's only offered here on
  // a no-auth server; authenticated MLflow routes via custom queues / users.
  const defaultQueueChecked = !authAvailable && defaultQueueSelected && inheritAllAssignable;
  const selectedCount = (defaultQueueChecked ? 1 : 0) + selectedQueueIds.size + selectedUsers.size;

  const isWorking = isAddingItems || isResolvingUserQueue || isSubmitting;
  const canAdd = selectedCount > 0 && itemIds.length > 0 && !isWorking && reviewerResolved;

  const triggerValue = useMemo(
    () =>
      selectedCount > 0
        ? [
            intl.formatMessage(
              {
                defaultMessage: '{count, plural, one {# queue} other {# queues}} selected',
                description: 'Add to review queue: destination dropdown selected-count summary',
              },
              { count: selectedCount },
            ),
          ]
        : [],
    [selectedCount, intl],
  );

  const toggleQueue = (queueId: string) =>
    setSelectedQueueIds((prev) => {
      const next = new Set(prev);
      if (next.has(queueId)) {
        next.delete(queueId);
      } else {
        next.add(queueId);
      }
      return next;
    });

  const toggleUser = (username: string) =>
    setSelectedUsers((prev) => {
      const next = new Set(prev);
      if (next.has(username)) {
        next.delete(username);
      } else {
        next.add(username);
      }
      return next;
    });

  const handleClose = () => {
    setSearch('');
    setDefaultQueueSelected(false);
    setSelectedQueueIds(new Set());
    setSelectedUsers(new Set());
    setCreateOpen(false);
    setSubmitError(null);
    setIsSubmitting(false);
    resetAdd();
    resetResolve();
    setVisible(false);
  };

  const handleCreated = (queue: ReviewQueue) => {
    setSelectedQueueIds((prev) => new Set(prev).add(queue.queue_id));
  };

  const handleAdd = async () => {
    if (!canAdd) {
      return;
    }
    // Block re-entry for the whole batch (the mutation hooks' isLoading can't —
    // see the `isSubmitting` declaration); cleared in `finally` on every path.
    setIsSubmitting(true);
    setSubmitError(null);
    try {
      // Fallback so a failure whose message is blank still surfaces something.
      const genericError = intl.formatMessage({
        defaultMessage: 'Please try again.',
        description: 'Add to review queue: fallback error detail when a failure carries no message',
      });
      // Resolve every USER-queue destination to a queue id: the no-auth catch-all
      // (the reserved `default` user queue) and each selected user's personal
      // queue, all via get-or-create. `allSettled` so one failure doesn't hide the
      // others; we collect every rejection rather than relying on the mutation's
      // single shared `error` slot, which only keeps the last-settled call.
      const usersToResolve = [...(defaultQueueChecked ? [DEFAULT_REVIEWER] : []), ...selectedUsers];
      const resolved = await Promise.allSettled(
        usersToResolve.map((user) =>
          getOrCreateUserQueueAsync({ experiment_id: experimentId, user, created_by: reviewer }).then(
            (res) => res.review_queue.queue_id,
          ),
        ),
      );
      // Gate on the rejection count (not the joined message, which can be empty),
      // and narrow the fulfilled results by status rather than casting. A failed
      // resolution aborts the whole add (including selected CUSTOM queues): a
      // requested destination we can't even resolve fails the operation visibly,
      // and the idempotent attach makes a corrected retry safe.
      const resolveRejections = resolved.filter((r): r is PromiseRejectedResult => r.status === 'rejected');
      if (resolveRejections.length > 0) {
        setSubmitError(collectRejectionMessages(resolveRejections) || genericError);
        return;
      }
      const resolvedQueueIds = resolved
        .filter((r): r is PromiseFulfilledResult<string> => r.status === 'fulfilled')
        .map((r) => r.value);
      // Attach the traces to every distinct destination, again accumulating every
      // failure. Re-attaching is idempotent server-side (an already-attached item
      // is a no-op that keeps its status), so retrying after a partial failure
      // safely re-sends to the destinations that already succeeded.
      const queueIds = Array.from(new Set([...selectedQueueIds, ...resolvedQueueIds]));
      const added = await Promise.allSettled(
        queueIds.map((queue_id) => addItemsToReviewQueueAsync({ queue_id, item_ids: itemIds })),
      );
      const addRejections = added.filter((r): r is PromiseRejectedResult => r.status === 'rejected');
      if (addRejections.length > 0) {
        setSubmitError(collectRejectionMessages(addRejections) || genericError);
        return;
      }
      // Confirm the add with a global toast — it must be global to survive this
      // modal unmounting on close. The notification holder renders inside the
      // router, so a <Link> resolves correctly for any deployment. The message
      // text is pre-built via `intl`. `white-space: nowrap` + the notification's
      // `width: 'auto'` keep it on one line (widening past the default fixed
      // width) instead of wrapping.
      const reviewQueuePath = generatePath(RoutePaths.experimentPageTabReviewQueue, { experimentId });
      Utils.displayGlobalInfoNotification(
        <span css={{ whiteSpace: 'nowrap' }}>
          {intl.formatMessage(
            {
              defaultMessage: 'Added {count, plural, one {# trace} other {# traces}} to review.',
              description: 'Add to review queue: success toast after traces are added',
            },
            { count: itemIds.length },
          )}{' '}
          <Link componentId={`${CID}.toast-view-queue`} to={reviewQueuePath}>
            {intl.formatMessage({
              defaultMessage: 'View review queue',
              description: 'Add to review queue: success toast link to the review queue page',
            })}
          </Link>
        </span>,
        undefined,
        { width: 'auto' },
      );
      handleClose();
    } finally {
      setIsSubmitting(false);
    }
  };

  const reasonText = (reason: ReturnType<typeof getQueueAssignability>['reason']) => {
    switch (reason) {
      case 'no-experiment-schemas':
        return intl.formatMessage({
          defaultMessage: 'No questions defined for this experiment yet.',
          description: 'Add to review queue: user queue disabled reason',
        });
      case 'no-resolvable-schemas':
      default:
        return intl.formatMessage({
          defaultMessage: 'This queue has no questions attached.',
          description: 'Add to review queue: custom queue disabled reason',
        });
    }
  };

  const defaultQueueLabel = intl.formatMessage({
    defaultMessage: 'Default queue',
    description: 'Add to review queue: the experiment default queue option',
  });

  return (
    <>
      <Modal
        componentId={`${CID}.modal`}
        // Hide while a child form (new queue / add question) is open rather than
        // stacking modals; it reopens when the child form closes (with the new
        // queue selected, in the create-queue case).
        visible={visible && !createOpen}
        destroyOnClose
        title={
          <FormattedMessage
            defaultMessage="Add {count, plural, one {# trace} other {# traces}} to review queues"
            description="Add to review queue modal title"
            values={{ count: itemIds.length }}
          />
        }
        okText={
          selectedCount > 0 ? (
            <FormattedMessage
              defaultMessage="Add to {count, plural, one {# queue} other {# queues}}"
              description="Add to review queue: confirm button with queue count"
              values={{ count: selectedCount }}
            />
          ) : (
            <FormattedMessage defaultMessage="Add" description="Add to review queue: confirm button" />
          )
        }
        okButtonProps={{ disabled: !canAdd, loading: isWorking }}
        cancelText={null}
        onOk={handleAdd}
        onCancel={handleClose}
      >
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
          {queuesError ? (
            <Alert
              componentId={`${CID}.load-error`}
              type="error"
              closable={false}
              message={intl.formatMessage({
                defaultMessage: 'Failed to load review queues.',
                description: 'Add to review queue: queue list load error',
              })}
              description={queuesError.message}
            />
          ) : queuesLoading ? (
            <TableSkeleton lines={4} />
          ) : (
            <DialogCombobox
              componentId={`${CID}.queue-picker`}
              label={intl.formatMessage({
                defaultMessage: 'Select or create review queues',
                description: 'Add to review queue: destination dropdown label',
              })}
              multiSelect
              value={triggerValue}
            >
              <DialogComboboxTrigger
                allowClear={false}
                placeholder={intl.formatMessage({
                  defaultMessage: 'Select review queues',
                  description: 'Add to review queue: destination dropdown placeholder',
                })}
              />
              <DialogComboboxContent
                matchTriggerWidth
                maxHeight={320}
                style={{ zIndex: theme.options.zIndexBase + 100 }}
              >
                <DialogComboboxOptionList>
                  <DialogComboboxOptionListSearch controlledValue={search} setControlledValue={setSearch}>
                    {!query && (
                      <Typography.Link
                        componentId={`${CID}.new-queue`}
                        css={{
                          display: 'inline-flex',
                          alignItems: 'center',
                          gap: theme.spacing.xs,
                          padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
                        }}
                        onClick={() => setCreateOpen(true)}
                      >
                        <PlusIcon css={{ paddingLeft: theme.spacing.xs, paddingRight: theme.spacing.xs }} />
                        <FormattedMessage
                          defaultMessage="New queue"
                          description="Add to review queue: create new queue link"
                        />
                      </Typography.Link>
                    )}

                    {/* Pinned default queue (no-auth only; the reserved `default` user queue). */}
                    {!authAvailable && !query && (
                      <DialogComboboxOptionListCheckboxItem
                        value={defaultQueueLabel}
                        checked={defaultQueueChecked}
                        disabled={!inheritAllAssignable}
                        disabledReason={inheritAllAssignable ? undefined : reasonText('no-experiment-schemas')}
                        onChange={() => setDefaultQueueSelected((prev) => !prev)}
                      >
                        {defaultQueueLabel}
                      </DialogComboboxOptionListCheckboxItem>
                    )}

                    {visibleQueues.length === 0 && query && (
                      <DialogComboboxEmpty
                        emptyText={
                          <FormattedMessage
                            defaultMessage="No matching queues"
                            description="Add to review queue: no custom queues match the search"
                          />
                        }
                      />
                    )}
                    {visibleQueues.map((q) => {
                      const assignability = assignabilityById.get(q.queue_id);
                      const disabled = !assignability?.assignable;
                      return (
                        <DialogComboboxOptionListCheckboxItem
                          key={q.queue_id}
                          value={q.name}
                          checked={selectedQueueIds.has(q.queue_id)}
                          disabled={disabled}
                          disabledReason={disabled ? reasonText(assignability?.reason) : undefined}
                          onChange={() => toggleQueue(q.queue_id)}
                        >
                          {q.name}
                        </DialogComboboxOptionListCheckboxItem>
                      );
                    })}
                    {hasMoreQueues && (
                      <DialogComboboxEmpty
                        emptyText={
                          <FormattedMessage
                            defaultMessage="Search to find more queues"
                            description="Add to review queue: hint that more custom queues are searchable"
                          />
                        }
                      />
                    )}

                    {canListUsers && (
                      <>
                        <DialogComboboxSectionHeader>
                          <FormattedMessage
                            defaultMessage="Users"
                            description="Add to review queue: per-user personal-queue section header"
                          />
                        </DialogComboboxSectionHeader>
                        {!query ? (
                          <DialogComboboxEmpty
                            emptyText={
                              <FormattedMessage
                                defaultMessage="Search by name to add a user's queue"
                                description="Add to review queue: prompt to search users"
                              />
                            }
                          />
                        ) : usersLoading ? (
                          <DialogComboboxEmpty
                            emptyText={
                              <FormattedMessage
                                defaultMessage="Loading users…"
                                description="Add to review queue: users loading hint"
                              />
                            }
                          />
                        ) : visibleUsers.length === 0 ? (
                          <DialogComboboxEmpty
                            emptyText={
                              <FormattedMessage
                                defaultMessage="No matching users"
                                description="Add to review queue: no users match the search"
                              />
                            }
                          />
                        ) : (
                          <>
                            {visibleUsers.map((u) => (
                              <DialogComboboxOptionListCheckboxItem
                                key={u.username}
                                value={u.username}
                                checked={selectedUsers.has(u.username)}
                                disabled={!inheritAllAssignable}
                                disabledReason={inheritAllAssignable ? undefined : reasonText('no-experiment-schemas')}
                                onChange={() => toggleUser(u.username)}
                              >
                                {u.username}
                              </DialogComboboxOptionListCheckboxItem>
                            ))}
                            {visibleUsers.length === MAX_USER_MATCHES && (
                              <DialogComboboxEmpty
                                emptyText={
                                  <FormattedMessage
                                    defaultMessage="Showing the first {count} matches — refine your search to narrow them."
                                    description="Add to review queue: hint that the user search results are capped"
                                    values={{ count: MAX_USER_MATCHES }}
                                  />
                                }
                              />
                            )}
                          </>
                        )}
                      </>
                    )}
                  </DialogComboboxOptionListSearch>
                </DialogComboboxOptionList>
              </DialogComboboxContent>
            </DialogCombobox>
          )}

          {submitError && (
            <Alert
              componentId={`${CID}.error`}
              type="error"
              closable={false}
              // Neutral title: the failure can come from the destination
              // resolution step or the attach step; the specific cause is in
              // `submitError` (the description).
              message={intl.formatMessage({
                defaultMessage: 'Something went wrong.',
                description: 'Add to review queue: error alert title',
              })}
              description={submitError}
            />
          )}
        </div>
      </Modal>

      {createOpen && (
        <CreateReviewQueueModal
          experimentId={experimentId}
          onClose={() => setCreateOpen(false)}
          onCreated={handleCreated}
        />
      )}
    </>
  );
};
