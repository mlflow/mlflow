import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import {
  Alert,
  DialogCombobox,
  DialogComboboxEmpty,
  DialogComboboxOptionList,
  DialogComboboxOptionListCheckboxItem,
  DialogComboboxOptionListSearch,
  DialogComboboxSectionHeader,
  PlusIcon,
  Popover,
  Spinner,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import type { ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';
import { FormattedMessage, useIntl } from 'react-intl';

import { useListLabelSchemasQuery } from '../../components/label-schemas';
import { useIsAuthAvailable } from '../../../account/hooks';
import { useCanEditReviews, useCanManageReviews } from './hooks/useCanManageReviews';
import Utils from '../../../common/utils/Utils';
import { Link } from '../../../common/utils/RoutingUtils';
import { CreateReviewQueueModal } from './CreateReviewQueueModal';
import { getQueueAssignability } from './queueAssignability';
import { canRemoveQueueItems, sameUser } from './queuePermissions';
import { useAddItemsToReviewQueueMutation } from './hooks/useAddItemsToReviewQueueMutation';
import { useAssignableUsersQuery } from './hooks/useAssignableUsersQuery';
import { useGetOrCreateUserQueueMutation } from './hooks/useGetOrCreateUserQueueMutation';
import { getReviewQueuePageRoute } from './hooks/useReviewQueueSearchParams';
import { useListReviewQueuesQuery } from './hooks/useListReviewQueuesQuery';
import { useRemoveItemsFromReviewQueueMutation } from './hooks/useRemoveItemsFromReviewQueueMutation';
import { DEFAULT_REVIEWER, useIsReviewerResolved, useReviewer } from './hooks/useReviewer';
import type { ReviewQueue } from './types';

const CID = 'mlflow.experiment-review-queue.add-to-queue';

// Custom queues shown before the reviewer searches; the rest are reachable by
// name through the dropdown's search box.
const COLLAPSED_QUEUE_COUNT = 3;
// Cap on user matches surfaced per search so a large roster can't flood the list.
const MAX_USER_MATCHES = 20;
const DROPDOWN_WIDTH = 320;
const LIST_MAX_HEIGHT = 280;

/**
 * Dropdown picker for routing one or more traces into review queues.
 *
 * Wraps `children` (the trigger element) with a {@link Popover} whose content
 * lets the caller multi-select destinations — shared CUSTOM queues and, on an
 * authenticated server, per-user personal queues — then confirm with a single
 * button.  Supports both controlled (`open` / `onOpenChange`) and uncontrolled
 * modes so it can replace a menu item (Actions dropdown) or simply decorate a
 * button ("Flag for review").
 *
 * Destinations are multi-select via a searchable list with two separate
 * sections: shared CUSTOM "Queues" (the first few shown up front, the rest found
 * by name) and, on an authenticated server where the caller can list users,
 * "Users" — each routing into that reviewer's personal queue, resolved
 * (get-or-create) on confirm. The experiment's default queue is pinned at the
 * top. Nothing is selected by default. Queues that wouldn't
 * present any questions are disabled (see `getQueueAssignability`), and a new
 * queue can be created inline.
 */
export const AddToReviewQueueDropdown = ({
  experimentId,
  selectedTraceInfos,
  children,
  open: controlledOpen,
  onOpenChange: controlledOnOpenChange,
  popoverAlign = 'end',
}: {
  experimentId: string;
  selectedTraceInfos: ModelTraceInfoV3[];
  children: React.ReactNode;
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
  popoverAlign?: 'start' | 'end';
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const reviewer = useReviewer();
  // Don't let a write stamp `created_by` until the reviewer identity is settled.
  const reviewerResolved = useIsReviewerResolved();
  const authAvailable = useIsAuthAvailable();
  // Flagging traces into a queue is an EDIT capability; fetch the roster for the
  // per-user-queue picker only when an editor opens the dropdown.
  const canEdit = useCanEditReviews(experimentId);
  const canManage = useCanManageReviews(experimentId);
  const canListUsers = authAvailable && canEdit;

  // -- open state (controlled or uncontrolled) --
  const [internalOpen, setInternalOpen] = useState(false);
  const isControlled = controlledOpen !== undefined;
  const isOpen = isControlled ? controlledOpen : internalOpen;
  const setOpen = useCallback(
    (next: boolean) => {
      if (isControlled) {
        controlledOnOpenChange?.(next);
      } else {
        setInternalOpen(next);
      }
    },
    [isControlled, controlledOnOpenChange],
  );

  // -- picker state --
  const [search, setSearch] = useState('');
  // Tracks queues/users that traces have already been added to in this session.
  const [addedQueueIds, setAddedQueueIds] = useState<Set<string>>(new Set());
  const [addedUsers, setAddedUsers] = useState<Set<string>>(new Set());
  const [createOpen, setCreateOpen] = useState(false);

  const {
    reviewQueues,
    isLoading: queuesLoading,
    error: queuesError,
  } = useListReviewQueuesQuery({ experimentId, enabled: isOpen || createOpen });
  const { labelSchemas } = useListLabelSchemasQuery({ experimentId, enabled: isOpen || createOpen });
  const { users, isLoading: usersLoading } = useAssignableUsersQuery({
    enabled: (isOpen || createOpen) && canListUsers,
  });
  const { addItemsToReviewQueueAsync, reset: resetAdd } = useAddItemsToReviewQueueMutation();
  const { removeItemsFromReviewQueueAsync } = useRemoveItemsFromReviewQueueMutation();
  const { getOrCreateUserQueueAsync, reset: resetResolve } = useGetOrCreateUserQueueMutation();
  // Tracks which queue IDs / usernames are currently being processed so only
  // those specific items are disabled — no full-list flash.
  const [busyIds, setBusyIds] = useState<Set<string>>(new Set());

  const itemIds = useMemo(
    () => selectedTraceInfos.map((info) => info.trace_id).filter((id): id is string => Boolean(id)),
    [selectedTraceInfos],
  );

  // For a single trace, reflect the queues it is already a member of. The
  // checked set is seeded from those memberships on open so a queue the reviewer
  // is allowed to remove from can be unchecked through the normal toggle path;
  // a queue they can't remove from is rendered checked-but-locked below. A bulk
  // selection has no single membership set, so this is skipped.
  const singleItemId = itemIds.length === 1 ? itemIds[0] : undefined;
  const { reviewQueues: memberQueues, isLoading: membersLoading } = useListReviewQueuesQuery({
    experimentId,
    itemId: singleItemId,
    enabled: (isOpen || createOpen) && Boolean(singleItemId),
  });
  const memberQueueIds = useMemo(
    () => new Set(memberQueues.filter((q) => q.queue_type === 'CUSTOM').map((q) => q.queue_id)),
    [memberQueues],
  );
  // USER-queue memberships, keyed by the queue name (== the username); covers
  // the no-auth `default` queue and any per-user queues the trace is in.
  const memberUserNames = useMemo(
    () => memberQueues.filter((q) => q.queue_type === 'USER').map((q) => q.name),
    [memberQueues],
  );
  const memberUserSet = useMemo(() => new Set(memberUserNames.map((n) => n.toLowerCase())), [memberUserNames]);
  // While a single trace's memberships are still loading we don't yet know which
  // rows are checked/locked, so disable toggling to avoid e.g. adding to a queue
  // the reviewer only meant to remove from. Bulk selections never query this.
  const membershipPending = Boolean(singleItemId) && membersLoading;
  const seededItemRef = useRef<string | null>(null);
  useEffect(() => {
    // Seed once per open (the ref guards against re-seeding on refetch), then
    // let toggles take over.
    if (!isOpen || !singleItemId || membersLoading || seededItemRef.current === singleItemId) {
      return;
    }
    seededItemRef.current = singleItemId;
    setAddedQueueIds(new Set(memberQueueIds));
    // Only seed user memberships that have a visible row to uncheck: the Users
    // section (canListUsers) on an auth server, or just the pinned `default`
    // queue on no-auth. Seeding an invisible user would be unreachable state.
    const seedableUsers = canListUsers ? memberUserNames : memberUserNames.filter((n) => sameUser(n, DEFAULT_REVIEWER));
    setAddedUsers(new Set(seedableUsers));
  }, [isOpen, singleItemId, membersLoading, memberQueueIds, memberUserNames, canListUsers]);

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
    // Keep added queues and existing memberships visible even if they'd fall
    // outside the collapsed head, so a checked row is never hidden.
    const extras = customQueues.filter(
      (q) => (addedQueueIds.has(q.queue_id) || memberQueueIds.has(q.queue_id)) && !headIds.has(q.queue_id),
    );
    return [...head, ...extras];
  }, [customQueues, query, addedQueueIds, memberQueueIds]);
  const hasMoreQueues = !query && customQueues.length > visibleQueues.length;

  // Users are search-driven; the experiment default queue is the pinned option.
  const visibleUsers = useMemo(() => {
    if (!query) {
      return [];
    }
    return users
      .filter(
        (u) =>
          !sameUser(u.username, reviewer) &&
          // Members are listed separately (checked) above the search results.
          !memberUserSet.has(u.username.toLowerCase()) &&
          u.username.toLowerCase().includes(query),
      )
      .slice(0, MAX_USER_MATCHES);
  }, [users, query, reviewer, memberUserSet]);

  // Existing memberships shown in the Users section, narrowed by the search.
  const visibleMemberUserNames = useMemo(
    () => (query ? memberUserNames.filter((n) => n.toLowerCase().includes(query)) : memberUserNames),
    [memberUserNames, query],
  );

  const resetState = useCallback(() => {
    setSearch('');
    setAddedQueueIds(new Set());
    setAddedUsers(new Set());
    setCreateOpen(false);
    setBusyIds(new Set());
    // Allow the next open to re-seed membership from the server.
    seededItemRef.current = null;
    resetAdd();
    resetResolve();
  }, [resetAdd, resetResolve]);

  const handleClose = useCallback(() => {
    resetState();
    setOpen(false);
  }, [resetState, setOpen]);

  const handleCreated = (queue: ReviewQueue) => {
    toggleCustomQueue(queue.queue_id);
  };

  const showSuccessToast = useCallback(
    (queueId: string) => {
      const reviewQueuePath = getReviewQueuePageRoute(experimentId, queueId);
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
        3,
      );
    },
    [experimentId, itemIds.length, intl],
  );

  const showErrorToast = useCallback(
    (e: unknown) =>
      Utils.displayGlobalErrorNotification(
        intl.formatMessage(
          {
            defaultMessage: 'Failed to update the review queue: {error}',
            description: 'Add to review queue: error toast when adding or removing traces fails',
          },
          { error: e instanceof Error ? e.message : String(e) },
        ),
      ),
    [intl],
  );

  const markBusy = (id: string) => setBusyIds((prev) => new Set(prev).add(id));
  const clearBusy = (id: string) =>
    setBusyIds((prev) => {
      const next = new Set(prev);
      next.delete(id);
      return next;
    });

  const toggleCustomQueue = async (queueId: string) => {
    if (busyIds.has(queueId) || !reviewerResolved || itemIds.length === 0) return;
    markBusy(queueId);
    const alreadyAdded = addedQueueIds.has(queueId);
    try {
      if (alreadyAdded) {
        await removeItemsFromReviewQueueAsync({ queue_id: queueId, item_ids: itemIds });
        setAddedQueueIds((prev) => {
          const next = new Set(prev);
          next.delete(queueId);
          return next;
        });
      } else {
        await addItemsToReviewQueueAsync({ queue_id: queueId, item_ids: itemIds });
        setAddedQueueIds((prev) => new Set(prev).add(queueId));
        showSuccessToast(queueId);
      }
    } catch (e) {
      showErrorToast(e);
    } finally {
      clearBusy(queueId);
    }
  };

  const toggleUserQueue = async (username: string) => {
    if (busyIds.has(username) || !reviewerResolved || itemIds.length === 0) return;
    markBusy(username);
    const alreadyAdded = addedUsers.has(username);
    try {
      const { review_queue } = await getOrCreateUserQueueAsync({
        experiment_id: experimentId,
        user: username,
        created_by: reviewer,
      });
      if (alreadyAdded) {
        await removeItemsFromReviewQueueAsync({ queue_id: review_queue.queue_id, item_ids: itemIds });
        setAddedUsers((prev) => {
          const next = new Set(prev);
          next.delete(username);
          return next;
        });
      } else {
        await addItemsToReviewQueueAsync({ queue_id: review_queue.queue_id, item_ids: itemIds });
        setAddedUsers((prev) => new Set(prev).add(username));
        showSuccessToast(review_queue.queue_id);
      }
    } catch (e) {
      showErrorToast(e);
    } finally {
      clearBusy(username);
    }
  };

  const toggleDefaultQueue = () => toggleUserQueue(DEFAULT_REVIEWER);

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
  // Reason shown on a checked-but-locked membership the reviewer can't remove.
  const lockedMemberReason = intl.formatMessage({
    defaultMessage: "This trace is already in this queue, and you don't have permission to remove it.",
    description: 'Add to review queue: queue option locked because the trace is a member the reviewer cannot remove',
  });

  return (
    <>
      <Popover.Root
        componentId={`${CID}.popover`}
        // Hide while a child form (new queue / add question) is open rather than
        // stacking; it reopens when the child form closes (with the new queue
        // selected, in the create-queue case).
        open={isOpen && !createOpen}
        onOpenChange={(next) => {
          if (next) {
            setOpen(true);
          } else {
            handleClose();
          }
        }}
      >
        <Popover.Trigger asChild>
          <div css={{ display: 'inline-flex' }}>{children}</div>
        </Popover.Trigger>
        <Popover.Content
          side="bottom"
          align={popoverAlign}
          collisionPadding={theme.spacing.sm}
          css={{
            width: DROPDOWN_WIDTH,
            minWidth: DROPDOWN_WIDTH,
            padding: 0,
            display: 'flex',
            flexDirection: 'column',
          }}
        >
          {queuesError ? (
            <div css={{ padding: theme.spacing.sm }}>
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
            </div>
          ) : queuesLoading ? (
            <div
              css={{
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                padding: theme.spacing.lg,
              }}
            >
              <Spinner />
            </div>
          ) : (
            <>
              <DialogCombobox
                componentId={`${CID}.queue-picker`}
                label={intl.formatMessage({
                  defaultMessage: 'Select or create review queues',
                  description: 'Add to review queue: destination dropdown label',
                })}
                multiSelect
                value={[]}
                open
              >
                <DialogComboboxOptionList css={{ maxHeight: LIST_MAX_HEIGHT, overflowY: 'auto', overflowX: 'hidden' }}>
                  <DialogComboboxOptionListSearch controlledValue={search} setControlledValue={setSearch}>
                    {/* Creating a queue (which you then own) requires EDIT. */}
                    {!query && canEdit && (
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
                        checked={addedUsers.has(DEFAULT_REVIEWER)}
                        disabled={!inheritAllAssignable || busyIds.has(DEFAULT_REVIEWER) || membershipPending}
                        disabledReason={inheritAllAssignable ? undefined : reasonText('no-experiment-schemas')}
                        onChange={() => toggleDefaultQueue()}
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
                      const notAssignable = !assignability?.assignable;
                      // A queue the trace already belongs to is locked unless the
                      // reviewer is allowed to remove from it (then unchecking
                      // removes the trace via the toggle path).
                      const lockedMember =
                        memberQueueIds.has(q.queue_id) && !canRemoveQueueItems(q, reviewer, canManage, canEdit);
                      return (
                        <DialogComboboxOptionListCheckboxItem
                          key={q.queue_id}
                          value={q.name}
                          checked={addedQueueIds.has(q.queue_id)}
                          disabled={lockedMember || notAssignable || busyIds.has(q.queue_id) || membershipPending}
                          disabledReason={
                            lockedMember
                              ? lockedMemberReason
                              : notAssignable
                                ? reasonText(assignability?.reason)
                                : undefined
                          }
                          onChange={() => toggleCustomQueue(q.queue_id)}
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
                        {/* Existing per-user memberships, shown checked. Still honor the search
                            filter so typing narrows the list. A personal queue is prunable only
                            by a manager, so otherwise it's locked. */}
                        {visibleMemberUserNames.map((username) => (
                          <DialogComboboxOptionListCheckboxItem
                            key={`member-${username}`}
                            value={username}
                            checked={addedUsers.has(username)}
                            disabled={!canManage || busyIds.has(username) || membershipPending}
                            disabledReason={!canManage ? lockedMemberReason : undefined}
                            onChange={() => toggleUserQueue(username)}
                          >
                            {username}
                          </DialogComboboxOptionListCheckboxItem>
                        ))}
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
                        ) : visibleUsers.length === 0 && visibleMemberUserNames.length === 0 ? (
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
                                checked={addedUsers.has(u.username)}
                                disabled={!inheritAllAssignable || busyIds.has(u.username) || membershipPending}
                                disabledReason={inheritAllAssignable ? undefined : reasonText('no-experiment-schemas')}
                                onChange={() => toggleUserQueue(u.username)}
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
              </DialogCombobox>
            </>
          )}
        </Popover.Content>
      </Popover.Root>

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
