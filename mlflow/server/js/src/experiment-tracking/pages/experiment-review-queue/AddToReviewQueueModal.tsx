import { useMemo, useState } from 'react';

import {
  Alert,
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxHintRow,
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
import { CreateReviewQueueModal } from './CreateReviewQueueModal';
import { getQueueAssignability } from './queueAssignability';
import { sameUser } from './queuePermissions';
import { useAddTracesToReviewQueueMutation } from './hooks/useAddTracesToReviewQueueMutation';
import { useAssignableUsersQuery } from './hooks/useAssignableUsersQuery';
import { useGetOrCreateDefaultQueueMutation } from './hooks/useGetOrCreateDefaultQueueMutation';
import { useGetOrCreateUserQueueMutation } from './hooks/useGetOrCreateUserQueueMutation';
import { useListReviewQueuesQuery } from './hooks/useListReviewQueuesQuery';
import { useReviewer } from './hooks/useReviewer';
import type { ReviewQueue } from './types';

const CID = 'mlflow.experiment-review-queue.add-to-queue';

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
  const {
    addTracesToReviewQueueAsync,
    isAddingTraces,
    error: addError,
    reset: resetAdd,
  } = useAddTracesToReviewQueueMutation();
  const {
    getOrCreateUserQueueAsync,
    isResolvingUserQueue,
    error: resolveError,
    reset: resetResolve,
  } = useGetOrCreateUserQueueMutation();
  const {
    getOrCreateDefaultQueueAsync,
    isResolvingDefaultQueue,
    error: resolveDefaultError,
    reset: resetResolveDefault,
  } = useGetOrCreateDefaultQueueMutation();

  const targetIds = useMemo(
    () => selectedTraceInfos.map((info) => info.trace_id).filter((id): id is string => Boolean(id)),
    [selectedTraceInfos],
  );

  // Shared queues anyone can route into. The experiment's default queue is
  // surfaced through the pinned option instead of the list; other users'
  // personal queues are reached through the "Users" section.
  const customQueues = useMemo(
    () => reviewQueues.filter((q) => q.queue_type === 'CUSTOM' && !q.is_default),
    [reviewQueues],
  );

  // The default queue and USER queues present every experiment schema, so they
  // are assignable as soon as the experiment has at least one question.
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

  const actionError = addError ?? resolveError ?? resolveDefaultError;
  const isWorking = isAddingTraces || isResolvingUserQueue || isResolvingDefaultQueue;
  const canAdd = selectedCount > 0 && targetIds.length > 0 && !isWorking;

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
    resetAdd();
    resetResolve();
    resetResolveDefault();
    setVisible(false);
  };

  const handleCreated = (queue: ReviewQueue) => {
    setSelectedQueueIds((prev) => new Set(prev).add(queue.queue_id));
  };

  const handleAdd = async () => {
    if (!canAdd) {
      return;
    }
    // Resolve the default queue (if picked) and each selected user's personal
    // queue via get-or-create, then attach the traces to each distinct
    // destination.
    const resolvedDefaultQueueIds = defaultQueueChecked
      ? [
          (await getOrCreateDefaultQueueAsync({ experiment_id: experimentId, created_by: reviewer })).review_queue
            .queue_id,
        ]
      : [];
    const resolvedUserQueueIds = await Promise.all(
      [...selectedUsers].map((user) =>
        getOrCreateUserQueueAsync({ experiment_id: experimentId, user, created_by: reviewer }).then(
          (res) => res.review_queue.queue_id,
        ),
      ),
    );
    const queueIds = Array.from(new Set([...selectedQueueIds, ...resolvedDefaultQueueIds, ...resolvedUserQueueIds]));
    await Promise.all(queueIds.map((queue_id) => addTracesToReviewQueueAsync({ queue_id, target_ids: targetIds })));
    handleClose();
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
        // Hide while the create form is open rather than stacking two modals; it
        // reopens with the new queue selected when the create modal closes.
        visible={visible && !createOpen}
        title={
          <FormattedMessage
            defaultMessage="Add {count, plural, one {# trace} other {# traces}} to review queues"
            description="Add to review queue modal title"
            values={{ count: targetIds.length }}
          />
        }
        okText={<FormattedMessage defaultMessage="Add" description="Add to review queue: confirm button" />}
        okButtonProps={{ disabled: !canAdd, loading: isWorking }}
        cancelText={<FormattedMessage defaultMessage="Cancel" description="Add to review queue: cancel button" />}
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
                defaultMessage: 'Review queues',
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
                    {/* Pinned default queue (no-auth only; resolved on confirm). */}
                    {!authAvailable && (
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

                    <DialogComboboxSectionHeader>
                      <FormattedMessage
                        defaultMessage="Queues"
                        description="Add to review queue: shared custom-queues section header"
                      />
                    </DialogComboboxSectionHeader>
                    {visibleQueues.length === 0 ? (
                      <DialogComboboxHintRow>
                        {query ? (
                          <FormattedMessage
                            defaultMessage="No matching queues"
                            description="Add to review queue: no custom queues match the search"
                          />
                        ) : (
                          <FormattedMessage
                            defaultMessage="No shared queues yet"
                            description="Add to review queue: no custom queues exist"
                          />
                        )}
                      </DialogComboboxHintRow>
                    ) : (
                      visibleQueues.map((q) => {
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
                      })
                    )}
                    {hasMoreQueues && (
                      <DialogComboboxHintRow>
                        <FormattedMessage
                          defaultMessage="Search to find more queues"
                          description="Add to review queue: hint that more custom queues are searchable"
                        />
                      </DialogComboboxHintRow>
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
                          <DialogComboboxHintRow>
                            <FormattedMessage
                              defaultMessage="Search by name to add a user's queue"
                              description="Add to review queue: prompt to search users"
                            />
                          </DialogComboboxHintRow>
                        ) : usersLoading ? (
                          <DialogComboboxHintRow>
                            <FormattedMessage
                              defaultMessage="Loading users…"
                              description="Add to review queue: users loading hint"
                            />
                          </DialogComboboxHintRow>
                        ) : visibleUsers.length === 0 ? (
                          <DialogComboboxHintRow>
                            <FormattedMessage
                              defaultMessage="No matching users"
                              description="Add to review queue: no users match the search"
                            />
                          </DialogComboboxHintRow>
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
                              <DialogComboboxHintRow>
                                <FormattedMessage
                                  defaultMessage="Showing the first {count} matches — refine your search to narrow them."
                                  description="Add to review queue: hint that the user search results are capped"
                                  values={{ count: MAX_USER_MATCHES }}
                                />
                              </DialogComboboxHintRow>
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

          <Typography.Link
            componentId={`${CID}.new-queue`}
            css={{ display: 'inline-flex', alignItems: 'center', gap: theme.spacing.xs, alignSelf: 'flex-start' }}
            onClick={() => setCreateOpen(true)}
          >
            <PlusIcon />
            <FormattedMessage defaultMessage="New queue" description="Add to review queue: create new queue link" />
          </Typography.Link>

          {actionError && (
            <Alert
              componentId={`${CID}.error`}
              type="error"
              closable={false}
              message={intl.formatMessage({
                defaultMessage: 'Failed to add traces to the review queue.',
                description: 'Add to review queue: error alert title',
              })}
              description={actionError.message}
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
