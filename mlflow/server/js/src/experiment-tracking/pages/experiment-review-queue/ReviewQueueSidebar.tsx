import { useState } from 'react';

import {
  ArrowDownIcon,
  ArrowUpIcon,
  Button,
  GearIcon,
  PlusIcon,
  SegmentedControlButton,
  SegmentedControlGroup,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { useQueries } from '@databricks/web-shared/query-client';
import { FormattedMessage, useIntl } from 'react-intl';

import { useIsAuthAvailable } from '../../../account/hooks';
import { buildReviewQueueItemsQuery } from './hooks/useListReviewQueueItemsQuery';
import { displayUser } from './hooks/useReviewer';
import { canInspectQueue, isQueueOwner } from './queuePermissions';
import type { ReviewQueueItem, ReviewQueue } from './types';

const CID = 'mlflow.experiment-review-queue.sidebar';

// Fixed widths so the owner and "To do" columns line up across rows.
const OWNER_COL_WIDTH = 120;
const COUNT_COL_WIDTH = 48;

type SortKey = 'name' | 'owner' | 'todo';
type SortDir = 'asc' | 'desc';

const SortHeader = ({
  label,
  active,
  dir,
  width,
  align,
  onClick,
}: {
  label: React.ReactNode;
  active: boolean;
  dir: SortDir;
  /** Fixed pixel width; omit to flex-fill the remaining space. */
  width?: number;
  align: 'left' | 'right';
  onClick: () => void;
}) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div
      role="button"
      tabIndex={0}
      onClick={onClick}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          onClick();
        }
      }}
      css={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: align === 'right' ? 'flex-end' : 'flex-start',
        gap: theme.spacing.xs,
        cursor: 'pointer',
        ...(width != null ? { width, flexShrink: 0 } : { flex: 1, minWidth: 0 }),
      }}
    >
      <Typography.Text size="sm" color="secondary" bold ellipsis>
        {label}
      </Typography.Text>
      {active && (dir === 'asc' ? <ArrowUpIcon /> : <ArrowDownIcon />)}
    </div>
  );
};

const QueueRow = ({
  label,
  owner,
  showOwner,
  pending,
  inspectable,
  selected,
  onSelect,
}: {
  label: string;
  owner: string;
  showOwner: boolean;
  /** Count of still-to-review traces; `undefined` while loading or not inspectable. */
  pending: number | undefined;
  inspectable: boolean;
  selected: boolean;
  onSelect: () => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const noAccessHint = intl.formatMessage({
    defaultMessage: "You don't have access to this queue.",
    description: 'Review queue sidebar: tooltip for a queue the reviewer cannot open',
  });

  return (
    <div
      {...(inspectable
        ? {
            role: 'button',
            tabIndex: 0,
            onClick: onSelect,
            onKeyDown: (e: React.KeyboardEvent) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                onSelect();
              }
            },
          }
        : { 'aria-disabled': true, title: noAccessHint })}
      css={{
        display: 'flex',
        alignItems: 'center',
        gap: theme.spacing.sm,
        padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
        borderRadius: theme.borders.borderRadiusMd,
        cursor: inspectable ? 'pointer' : 'default',
        // Greyed: an EDIT user can see every queue in the list but may only open
        // ones they own or are assigned to (mirrors the server detail-tier gate).
        opacity: inspectable ? 1 : 0.5,
        backgroundColor: selected ? theme.colors.actionDefaultBackgroundPress : undefined,
        '&:hover': inspectable
          ? { backgroundColor: selected ? undefined : theme.colors.actionDefaultBackgroundHover }
          : undefined,
      }}
    >
      <Typography.Text bold={selected} ellipsis css={{ flex: 1, minWidth: 0 }}>
        {label}
      </Typography.Text>
      {showOwner && (
        <Typography.Text color="secondary" ellipsis css={{ width: OWNER_COL_WIDTH, flexShrink: 0 }}>
          {owner}
        </Typography.Text>
      )}
      <Typography.Text color="secondary" css={{ width: COUNT_COL_WIDTH, flexShrink: 0, textAlign: 'right' }}>
        {/* Blank for a zero count (no work is just noise as a "0"), while the
            count loads, and for queues the reviewer can't inspect. */}
        {pending ? pending : ''}
      </Typography.Text>
    </div>
  );
};

/**
 * Left panel of the Review tab: a flat, sortable list of the reviewer's visible
 * queues with each queue's owner and how many traces are still to review. Which
 * queues are visible is decided server-side — managers and editors see every
 * queue; read-only reviewers see only the queues they're assigned to. Editors
 * see queues they don't own greyed out (listed but not openable, matching the
 * server's detail-tier gate); the "My queues" filter narrows to owned queues.
 * The selected queue's questions and per-queue actions (manage / delete) live in
 * the right pane's header, not here.
 */
export const ReviewQueueSidebar = ({
  queues,
  selectedQueueId,
  canManage,
  canEdit,
  canCreateQueue,
  reviewer,
  onSelect,
  onNewQueue,
  onManageQuestions,
}: {
  queues: ReviewQueue[];
  selectedQueueId: string | undefined;
  canManage: boolean;
  canEdit: boolean;
  canCreateQueue: boolean;
  reviewer: string;
  onSelect: (queueId: string) => void;
  onNewQueue: () => void;
  onManageQuestions: () => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const authAvailable = useIsAuthAvailable();
  const [sortKey, setSortKey] = useState<SortKey>('name');
  const [sortDir, setSortDir] = useState<SortDir>('asc');
  const [mineOnly, setMineOnly] = useState(false);

  // Owner is only meaningful on an auth server (no-auth leaves `created_by`
  // unset). The "My queues" filter only helps users who can see queues they
  // don't own — read-only reviewers already see just their assigned queues.
  const showOwner = authAvailable;
  const showFilter = authAvailable && canEdit;

  const inspectable = (q: ReviewQueue) => canInspectQueue(q, reviewer, canManage, canEdit);

  // One fetch per inspectable queue for its pending count; shares cache with the
  // right panel's trace list (same query config). Non-inspectable queues are
  // skipped — their item list would 403 — and render a blank count.
  const traceQueries = useQueries({
    queries: queues.map((q) => ({
      ...buildReviewQueueItemsQuery({ queueId: q.queue_id }),
      enabled: Boolean(q.queue_id) && inspectable(q),
    })),
  });
  const pendingByQueueId = new Map<string, number>();
  queues.forEach((q, idx) => {
    const result = traceQueries[idx];
    if (result && !result.isLoading && result.data) {
      const items = (result.data.items ?? []) as ReviewQueueItem[];
      pendingByQueueId.set(q.queue_id, items.filter((i) => i.status === 'PENDING').length);
    }
  });

  const labelOf = (q: ReviewQueue) => (q.queue_type === 'USER' ? displayUser(q.name, intl) : q.name);
  const ownerOf = (q: ReviewQueue) => q.created_by ?? '';

  const toggleSort = (key: SortKey) => {
    if (key === sortKey) {
      setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortKey(key);
      setSortDir('asc');
    }
  };

  const dirMul = sortDir === 'asc' ? 1 : -1;
  const byLabel = (a: ReviewQueue, b: ReviewQueue) =>
    labelOf(a).localeCompare(labelOf(b), undefined, { sensitivity: 'base' });
  const compare = (a: ReviewQueue, b: ReviewQueue): number => {
    if (sortKey === 'todo') {
      const pa = pendingByQueueId.get(a.queue_id);
      const pb = pendingByQueueId.get(b.queue_id);
      // Unknown counts (still loading or not inspectable) always sort last,
      // regardless of direction.
      if (pa == null && pb == null) return byLabel(a, b);
      if (pa == null) return 1;
      if (pb == null) return -1;
      return pa !== pb ? dirMul * (pa - pb) : byLabel(a, b);
    }
    const va = sortKey === 'owner' ? ownerOf(a) : labelOf(a);
    const vb = sortKey === 'owner' ? ownerOf(b) : labelOf(b);
    const d = va.localeCompare(vb, undefined, { sensitivity: 'base' });
    return d !== 0 ? dirMul * d : byLabel(a, b);
  };

  // Ignore a stale `mineOnly` if the filter control is no longer shown (e.g.
  // auth dropped), so the list can't get stuck owner-filtered with no way to
  // clear it. Keep the selected queue visible even when it isn't owned, so the
  // filter can't hide the queue the right pane is still showing.
  const effectiveMineOnly = showFilter && mineOnly;
  const visible = (
    effectiveMineOnly ? queues.filter((q) => isQueueOwner(q, reviewer) || q.queue_id === selectedQueueId) : queues
  )
    .slice()
    .sort(compare);

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
      {(canManage || canCreateQueue) && (
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs, flexWrap: 'wrap' }}>
          {/* Editing the experiment's questions is MANAGE; creating a queue (which
              you then own) only needs EDIT. */}
          {canManage && (
            <Button componentId={`${CID}.manage-questions`} icon={<GearIcon />} onClick={onManageQuestions}>
              <FormattedMessage
                defaultMessage="Manage questions"
                description="Review queue sidebar: manage-questions button"
              />
            </Button>
          )}
          {canCreateQueue && (
            <Button componentId={`${CID}.new-queue`} icon={<PlusIcon />} onClick={onNewQueue}>
              <FormattedMessage defaultMessage="New queue" description="Review queue: create-queue button" />
            </Button>
          )}
        </div>
      )}

      {showFilter && queues.length > 0 && (
        <SegmentedControlGroup
          name="review-queue-owner-filter"
          componentId={`${CID}.owner-filter`}
          size="small"
          value={mineOnly ? 'mine' : 'all'}
          onChange={(e) => setMineOnly(e.target.value === 'mine')}
        >
          <SegmentedControlButton value="all">
            <FormattedMessage defaultMessage="All queues" description="Review queue sidebar: show-all-queues filter" />
          </SegmentedControlButton>
          <SegmentedControlButton value="mine">
            <FormattedMessage
              defaultMessage="My queues"
              description="Review queue sidebar: show-only-owned-queues filter"
            />
          </SegmentedControlButton>
        </SegmentedControlGroup>
      )}

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
          <SortHeader
            label={<FormattedMessage defaultMessage="Queue" description="Review queue sidebar: queue-name column" />}
            active={sortKey === 'name'}
            dir={sortDir}
            align="left"
            onClick={() => toggleSort('name')}
          />
          {showOwner && (
            <SortHeader
              label={<FormattedMessage defaultMessage="Owner" description="Review queue sidebar: queue-owner column" />}
              active={sortKey === 'owner'}
              dir={sortDir}
              width={OWNER_COL_WIDTH}
              align="left"
              onClick={() => toggleSort('owner')}
            />
          )}
          <SortHeader
            label={
              <FormattedMessage
                defaultMessage="To do"
                description="Review queue sidebar: still-to-review count column"
              />
            }
            active={sortKey === 'todo'}
            dir={sortDir}
            width={COUNT_COL_WIDTH}
            align="right"
            onClick={() => toggleSort('todo')}
          />
        </div>
      )}

      {visible.length > 0 ? (
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
          {visible.map((queue) => (
            <QueueRow
              key={queue.queue_id}
              label={labelOf(queue)}
              owner={ownerOf(queue)}
              showOwner={showOwner}
              pending={pendingByQueueId.get(queue.queue_id)}
              inspectable={inspectable(queue)}
              selected={queue.queue_id === selectedQueueId}
              onSelect={() => onSelect(queue.queue_id)}
            />
          ))}
        </div>
      ) : (
        effectiveMineOnly &&
        queues.length > 0 && (
          <Typography.Text color="secondary" css={{ paddingLeft: theme.spacing.sm }}>
            <FormattedMessage
              defaultMessage="You don't own any queues yet."
              description="Review queue sidebar: empty state for the My-queues filter"
            />
          </Typography.Text>
        )
      )}
    </div>
  );
};
