import { useState } from 'react';

import {
  Button,
  ChevronDownIcon,
  ChevronRightIcon,
  GearIcon,
  PlusIcon,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { useQueries } from '@databricks/web-shared/query-client';
import { FormattedMessage, useIntl } from 'react-intl';

import { displayUser } from './hooks/useReviewer';
import { buildReviewQueueItemsQuery } from './hooks/useListReviewQueueItemsQuery';
import { sameUser } from './queuePermissions';
import type { ReviewQueueItem, ReviewQueue } from './types';

const CID = 'mlflow.experiment-review-queue.sidebar';

// Fixed width so the "To do" count lines up into a column across rows.
const COUNT_COL_WIDTH = 48;

const QueueRow = ({
  queue,
  selected,
  pending,
  onSelect,
}: {
  queue: ReviewQueue;
  selected: boolean;
  /** Count of still-to-review traces; `undefined` while the count loads. */
  pending: number | undefined;
  onSelect: () => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const label = queue.queue_type === 'USER' ? displayUser(queue.name, intl) : queue.name;

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
        {/* Blank for a zero count (queues with no work sit under "No work to do",
            where a "0" is just noise) and while the count is still loading. */}
        {pending ? pending : ''}
      </Typography.Text>
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
 * into "Feedback requested" (assigned by others) and "Created by me"; a no-auth
 * server shows one list. The selected queue's questions and per-queue actions
 * (manage / delete) live in the right pane's header, not here.
 */
export const ReviewQueueSidebar = ({
  queues,
  selectedQueueId,
  reviewer,
  authAvailable,
  canManage,
  onSelect,
  onDeselectQueue,
  onNewQueue,
  onManageQuestions,
}: {
  queues: ReviewQueue[];
  selectedQueueId: string | undefined;
  reviewer: string;
  authAvailable: boolean;
  canManage: boolean;
  onSelect: (queueId: string) => void;
  onDeselectQueue: () => void;
  onNewQueue: () => void;
  onManageQuestions: () => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const [noWorkOpen, setNoWorkOpen] = useState(false);

  // One fetch per queue for its pending count; shares cache with the right
  // panel's trace list (same query config).
  const traceQueries = useQueries({
    queries: queues.map((q) => buildReviewQueueItemsQuery({ queueId: q.queue_id })),
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
  const active = queues.filter((q) => !isNoWork(q));
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
      pending={pendingByQueueId.get(queue.queue_id)}
      onSelect={() => onSelect(queue.queue_id)}
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
    </div>
  );
};
