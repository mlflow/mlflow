import { useState } from 'react';

import {
  Button,
  ChevronDownIcon,
  ChevronRightIcon,
  GearIcon,
  PlusIcon,
  Popover,
  Tag,
  TrashIcon,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { useQueries } from '@databricks/web-shared/query-client';
import { FormattedMessage, useIntl } from 'react-intl';

import type { LabelSchema } from '../../components/label-schemas';
import { displayUser } from './hooks/useReviewer';
import { buildReviewQueueTracesQuery } from './hooks/useListReviewQueueTracesQuery';
import { canDeleteQueue, canManageQueue, sameUser } from './queuePermissions';
import type { ReviewQueueItem, ReviewQueue } from './types';

const CID = 'mlflow.experiment-review-queue.sidebar';

// Fixed widths so the "To do" count and the row action line up into columns across rows.
const COUNT_COL_WIDTH = 48;
const ACTION_COL_WIDTH = 32;
// Fixed type-tag column for the questions list so every question name truncates
// at the same point (wide enough for the longest tag, "Expectation").
const QUESTION_TAG_COL_WIDTH = 96;

const QueueRow = ({
  queue,
  selected,
  canEdit,
  canDelete,
  pending,
  onSelect,
  onEdit,
  onDelete,
}: {
  queue: ReviewQueue;
  selected: boolean;
  /** Show the gear (edit members / delete via the modal); auth servers only. */
  canEdit: boolean;
  canDelete: boolean;
  /** Count of still-to-review traces; `undefined` while the count loads. */
  pending: number | undefined;
  onSelect: () => void;
  onEdit: () => void;
  onDelete: () => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const label = queue.is_default
    ? intl.formatMessage({
        defaultMessage: 'Default queue',
        description: 'Review queue sidebar: label for the experiment default queue',
      })
    : queue.queue_type === 'USER'
      ? displayUser(queue.name, intl)
      : queue.name;

  return (
    <div
      role="button"
      tabIndex={0}
      onClick={onSelect}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          onSelect();
        }
      }}
      css={{
        display: 'flex',
        alignItems: 'center',
        gap: theme.spacing.sm,
        padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
        borderRadius: theme.borders.borderRadiusMd,
        cursor: 'pointer',
        backgroundColor: selected ? theme.colors.actionDefaultBackgroundPress : undefined,
        '&:hover': { backgroundColor: selected ? undefined : theme.colors.actionDefaultBackgroundHover },
      }}
    >
      <Typography.Text bold={selected} ellipsis css={{ flex: 1, minWidth: 0 }}>
        {label}
      </Typography.Text>
      <Typography.Text color="secondary" css={{ width: COUNT_COL_WIDTH, flexShrink: 0, textAlign: 'right' }}>
        {pending ?? ''}
      </Typography.Text>
      {/* Stop row selection on the wrapper, not the trigger: suppressing the
          trigger's own onClick would also swallow Radix's open-toggle. */}
      <div
        css={{ width: ACTION_COL_WIDTH, flexShrink: 0, display: 'flex', justifyContent: 'flex-end' }}
        onClick={(e) => e.stopPropagation()}
        onKeyDown={(e) => e.stopPropagation()}
      >
        {/* Auth: gear opens the edit-members modal (which owns delete). No-auth:
            there are no members to assign, so fall back to the delete popover. */}
        {canEdit ? (
          <Button
            componentId={`${CID}.edit-trigger`}
            size="small"
            icon={<GearIcon />}
            onClick={onEdit}
            aria-label={intl.formatMessage({
              defaultMessage: 'Edit queue',
              description: 'Review queue sidebar: edit-queue gear button aria label',
            })}
          />
        ) : canDelete ? (
          <Popover.Root componentId={`${CID}.delete-popover`}>
            <Popover.Trigger asChild>
              <Button
                componentId={`${CID}.delete-trigger`}
                size="small"
                icon={<TrashIcon />}
                aria-label={intl.formatMessage({
                  defaultMessage: 'Delete queue',
                  description: 'Review queue sidebar: delete-queue trash button aria label',
                })}
              />
            </Popover.Trigger>
            {/* Override the DuBois 220px min-width so the popover hugs the button. */}
            <Popover.Content align="end" css={{ minWidth: 'auto' }}>
              <Button componentId={`${CID}.delete-confirm`} danger icon={<TrashIcon />} onClick={onDelete}>
                <FormattedMessage
                  defaultMessage="Delete queue"
                  description="Review queue sidebar: confirm delete-queue button"
                />
              </Button>
            </Popover.Content>
          </Popover.Root>
        ) : null}
      </div>
    </div>
  );
};

const Group = ({ title, children }: { title: React.ReactNode; children: React.ReactNode }) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
      <Typography.Text size="sm" color="secondary" bold css={{ paddingLeft: theme.spacing.sm }}>
        {title}
      </Typography.Text>
      {children}
    </div>
  );
};

const CollapsibleGroup = ({
  title,
  count,
  open,
  onToggle,
  children,
}: {
  title: React.ReactNode;
  count: number;
  open: boolean;
  onToggle: () => void;
  children: React.ReactNode;
}) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
      <div
        role="button"
        tabIndex={0}
        onClick={onToggle}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            onToggle();
          }
        }}
        css={{
          display: 'flex',
          alignItems: 'center',
          gap: theme.spacing.xs,
          paddingLeft: theme.spacing.sm,
          cursor: 'pointer',
        }}
      >
        {open ? <ChevronDownIcon /> : <ChevronRightIcon />}
        <Typography.Text size="sm" color="secondary" bold>
          {title}
        </Typography.Text>
        <Typography.Text size="sm" color="secondary">
          ({count})
        </Typography.Text>
      </div>
      {open && children}
    </div>
  );
};

/**
 * Left panel of the Review tab: the reviewer's queues, each showing how many
 * traces are still to review. Queues with nothing left to review collapse into a
 * "No work to do" group. On an authenticated server the active queues are split
 * into "Feedback requested" (assigned by others — answer only) and "Created by
 * me" (deletable via the row trash action); a no-auth server shows one list.
 */
export const ReviewQueueSidebar = ({
  queues,
  selectedQueueId,
  reviewer,
  authAvailable,
  canManage,
  selectedQueueQuestions,
  onSelect,
  onDeselectQueue,
  onDeleteQueue,
  onEditQueue,
  onEditQuestion,
  onNewQueue,
  onManageQuestions,
}: {
  queues: ReviewQueue[];
  selectedQueueId: string | undefined;
  reviewer: string;
  authAvailable: boolean;
  canManage: boolean;
  /** Questions (label schemas) the selected queue asks, shown at the bottom. */
  selectedQueueQuestions: LabelSchema[];
  onSelect: (queueId: string) => void;
  onDeselectQueue: () => void;
  onDeleteQueue: (queueId: string) => void;
  /** Open the edit-members modal for a queue (sidebar gear). */
  onEditQueue: (queueId: string) => void;
  /** When provided, a question row is clickable to open its edit modal. */
  onEditQuestion?: (schema: LabelSchema) => void;
  onNewQueue: () => void;
  onManageQuestions: () => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [noWorkOpen, setNoWorkOpen] = useState(false);

  // One fetch per queue for its pending count; shares cache with the right
  // panel's trace list (same query config).
  const traceQueries = useQueries({
    queries: queues.map((q) => buildReviewQueueTracesQuery({ queueId: q.queue_id })),
  });
  // Pending count once loaded; absent (undefined) while a queue's count loads.
  const pendingByQueueId = new Map<string, number>();
  queues.forEach((q, idx) => {
    const result = traceQueries[idx];
    if (result && !result.isLoading) {
      const items = (result.data?.items ?? []) as ReviewQueueItem[];
      pendingByQueueId.set(q.queue_id, items.filter((i) => i.status === 'PENDING').length);
    }
  });

  // No-work == loaded with zero pending. Loading queues stay in the active list.
  const isNoWork = (q: ReviewQueue) => pendingByQueueId.get(q.queue_id) === 0;
  // The default queue flows through the normal grouping like any other queue,
  // but is surfaced first among the active queues when it has work to do.
  const active = queues
    .filter((q) => !isNoWork(q))
    .sort((a, b) => Number(Boolean(b.is_default)) - Number(Boolean(a.is_default)));
  const noWork = queues.filter(isNoWork);
  // Keep the selected queue visible even if it has no work — selecting a no-work
  // queue force-expands the group.
  const selectedInNoWork = noWork.some((q) => q.queue_id === selectedQueueId);
  const noWorkExpanded = noWorkOpen || selectedInNoWork;
  // Collapsing the group also drops a selected no-work queue, otherwise the
  // selection would force it back open and the collapse would appear to do nothing.
  const toggleNoWork = () => {
    if (noWorkExpanded) {
      if (selectedInNoWork) {
        onDeselectQueue();
      }
      setNoWorkOpen(false);
    } else {
      setNoWorkOpen(true);
    }
  };

  const renderRow = (queue: ReviewQueue) => (
    <QueueRow
      key={queue.queue_id}
      queue={queue}
      selected={queue.queue_id === selectedQueueId}
      canEdit={authAvailable && canManageQueue(queue, reviewer, authAvailable, canManage)}
      canDelete={canDeleteQueue(queue, reviewer, authAvailable, canManage)}
      pending={pendingByQueueId.get(queue.queue_id)}
      onSelect={() => onSelect(queue.queue_id)}
      onEdit={() => onEditQueue(queue.queue_id)}
      onDelete={() => onDeleteQueue(queue.queue_id)}
    />
  );

  // Auth: split active queues owned vs assigned-by-others. No-auth: one list.
  const created = authAvailable ? active.filter((q) => sameUser(q.created_by, reviewer)) : [];
  const requested = authAvailable ? active.filter((q) => !sameUser(q.created_by, reviewer)) : [];

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.md,
        height: '100%',
        minHeight: 0,
        paddingRight: theme.spacing.sm,
        overflow: 'auto',
      }}
    >
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
        <Typography.Title level={3} withoutMargins css={{ flex: 1 }}>
          <FormattedMessage defaultMessage="Review" description="Review queue tab title" />
        </Typography.Title>
        {canManage && (
          <Button componentId={`${CID}.manage-questions`} icon={<GearIcon />} onClick={onManageQuestions}>
            <FormattedMessage
              defaultMessage="Manage questions"
              description="Review queue sidebar: manage-questions button"
            />
          </Button>
        )}
        {canManage && (
          <Button componentId={`${CID}.new-queue`} icon={<PlusIcon />} onClick={onNewQueue}>
            <FormattedMessage defaultMessage="New queue" description="Review queue: create-queue button" />
          </Button>
        )}
      </div>

      {queues.length > 0 && (
        <div
          css={{
            display: 'flex',
            alignItems: 'center',
            gap: theme.spacing.sm,
            padding: `0 ${theme.spacing.sm}px`,
            borderBottom: `1px solid ${theme.colors.border}`,
            paddingBottom: theme.spacing.xs,
          }}
        >
          <Typography.Text size="sm" color="secondary" bold css={{ flex: 1, minWidth: 0 }}>
            <FormattedMessage defaultMessage="Queue" description="Review queue sidebar: queue-name column header" />
          </Typography.Text>
          <Typography.Text
            size="sm"
            color="secondary"
            bold
            css={{ width: COUNT_COL_WIDTH, flexShrink: 0, textAlign: 'right' }}
          >
            <FormattedMessage
              defaultMessage="To do"
              description="Review queue sidebar: still-to-review count column header"
            />
          </Typography.Text>
          <div css={{ width: ACTION_COL_WIDTH, flexShrink: 0 }} />
        </div>
      )}

      {authAvailable ? (
        <>
          {requested.length > 0 && (
            <Group
              title={
                <FormattedMessage
                  defaultMessage="Feedback requested"
                  description="Review queue sidebar: group of queues assigned to the reviewer by others"
                />
              }
            >
              {requested.map(renderRow)}
            </Group>
          )}
          {created.length > 0 && (
            <Group
              title={
                <FormattedMessage
                  defaultMessage="Created by me"
                  description="Review queue sidebar: group of queues the reviewer owns"
                />
              }
            >
              {created.map(renderRow)}
            </Group>
          )}
        </>
      ) : (
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>{active.map(renderRow)}</div>
      )}

      {noWork.length > 0 && (
        <CollapsibleGroup
          title={
            <FormattedMessage
              defaultMessage="No work to do"
              description="Review queue sidebar: collapsed group of queues with nothing left to review"
            />
          }
          count={noWork.length}
          open={noWorkExpanded}
          onToggle={toggleNoWork}
        >
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>{noWork.map(renderRow)}</div>
        </CollapsibleGroup>
      )}

      {/* Questions the selected queue asks, pinned to the bottom so reviewers
          can see what they'll be answering before opening a trace. */}
      {selectedQueueId && selectedQueueQuestions.length > 0 && (
        <div
          css={{
            marginTop: 'auto',
            paddingTop: theme.spacing.sm,
            borderTop: `1px solid ${theme.colors.border}`,
            display: 'flex',
            flexDirection: 'column',
            gap: theme.spacing.xs,
          }}
        >
          <Typography.Text size="sm" color="secondary" bold css={{ paddingLeft: theme.spacing.sm }}>
            <FormattedMessage
              defaultMessage="Questions in this queue"
              description="Review queue sidebar: header for the selected queue's questions"
            />
          </Typography.Text>
          {selectedQueueQuestions.map((q) => {
            const editable = Boolean(onEditQuestion);
            return (
              <div
                key={q.schema_id}
                role={editable ? 'button' : undefined}
                tabIndex={editable ? 0 : undefined}
                onClick={editable ? () => onEditQuestion?.(q) : undefined}
                onKeyDown={
                  editable
                    ? (e) => {
                        if (e.key === 'Enter' || e.key === ' ') {
                          e.preventDefault();
                          onEditQuestion?.(q);
                        }
                      }
                    : undefined
                }
                css={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: theme.spacing.sm,
                  padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
                  borderRadius: theme.borders.borderRadiusMd,
                  cursor: editable ? 'pointer' : 'default',
                  '&:hover': editable ? { backgroundColor: theme.colors.actionDefaultBackgroundHover } : undefined,
                }}
              >
                <Typography.Text ellipsis css={{ flex: 1, minWidth: 0 }}>
                  {q.name}
                </Typography.Text>
                <div
                  css={{ width: QUESTION_TAG_COL_WIDTH, flexShrink: 0, display: 'flex', justifyContent: 'flex-end' }}
                >
                  <Tag componentId={`${CID}.question-type`} color={q.type === 'EXPECTATION' ? 'turquoise' : 'lime'}>
                    {q.type === 'EXPECTATION' ? (
                      <FormattedMessage defaultMessage="Expectation" description="Label schema type: expectation" />
                    ) : (
                      <FormattedMessage defaultMessage="Feedback" description="Label schema type: feedback" />
                    )}
                  </Tag>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
