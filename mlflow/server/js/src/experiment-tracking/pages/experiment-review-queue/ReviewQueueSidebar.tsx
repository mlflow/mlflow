import { useEffect, useRef, useState } from 'react';

import {
  ArrowDownIcon,
  ArrowUpIcon,
  Button,
  GearIcon,
  PlusIcon,
  SegmentedControlButton,
  SegmentedControlGroup,
  Spinner,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { useQueries } from '@databricks/web-shared/query-client';
import { FormattedMessage, useIntl } from 'react-intl';

import { useIsAuthAvailable } from '../../../account/hooks';
import { useInfiniteScrollFetch } from '../experiment-evaluation-datasets/hooks/useInfiniteScrollFetch';
import { buildReviewQueueItemsQuery } from './hooks/useListReviewQueueItemsQuery';
import { displayUser } from './hooks/useReviewer';
import { canInspectQueue, isQueueOwner } from './queuePermissions';
import type { ReviewQueueItem, ReviewQueue } from './types';

const CID = 'mlflow.experiment-review-queue.sidebar';

// Fixed widths so the owner and "To do" columns line up across rows.
const OWNER_COL_WIDTH = 120;
const COUNT_COL_WIDTH = 48;

type SortDir = 'asc' | 'desc';

/** Server-sortable queue columns (must match the backend `order_by` whitelist). */
export type ReviewQueueSortField = 'name' | 'created_by' | 'creation_time_ms';
export type ReviewQueueSort = { field: ReviewQueueSortField; direction: SortDir };

// Default: newest created first, matching the server's default order. The "To do"
// count column is intentionally not sortable — it's derived client-side from the
// per-queue count fetches, so it can't drive a server-side (whole-list) sort.
export const DEFAULT_REVIEW_QUEUE_SORT: ReviewQueueSort = { field: 'creation_time_ms', direction: 'desc' };

// Serialize the sidebar sort into backend `order_by` clauses. Sorting is done
// server-side so it spans the entire queue list, not just the loaded pages
// (client-side sorting a paginated list only orders the loaded window).
export const reviewQueueSortToOrderBy = (sort: ReviewQueueSort): string[] => [
  `${sort.field} ${sort.direction.toUpperCase()}`,
];

const SortHeader = ({
  label,
  active = false,
  dir = 'asc',
  width,
  align,
  onClick,
}: {
  label: React.ReactNode;
  active?: boolean;
  dir?: SortDir;
  /** Fixed pixel width; omit to flex-fill the remaining space. */
  width?: number;
  align: 'left' | 'right';
  /** Omit to render a plain, non-sortable column header (no arrow, not clickable). */
  onClick?: () => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const sortable = Boolean(onClick);
  return (
    <div
      {...(sortable
        ? {
            role: 'button',
            tabIndex: 0,
            onClick,
            onKeyDown: (e: React.KeyboardEvent) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                onClick?.();
              }
            },
          }
        : {})}
      css={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: align === 'right' ? 'flex-end' : 'flex-start',
        gap: theme.spacing.xs,
        cursor: sortable ? 'pointer' : 'default',
        ...(width != null ? { width, flexShrink: 0 } : { flex: 1, minWidth: 0 }),
      }}
    >
      <Typography.Text size="sm" color="secondary" bold ellipsis>
        {label}
      </Typography.Text>
      {sortable && active && (dir === 'asc' ? <ArrowUpIcon /> : <ArrowDownIcon />)}
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
        // Greyed when not inspectable: listed but not openable (server detail-tier gate).
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
        {/* Blank for zero, while loading, and for non-inspectable queues. */}
        {pending ? pending : ''}
      </Typography.Text>
    </div>
  );
};

/**
 * Left panel of the Review tab: a flat, sortable list of the reviewer's visible
 * queues with each queue's owner and to-do count. Visibility is decided
 * server-side; queues that can't be opened are greyed (the detail-tier gate), and
 * the "My queues" filter narrows to owned queues. Per-queue actions live in the
 * right pane, not here.
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
  sort,
  onSortChange,
  onLoadMore,
  hasMore,
  isLoadingMore,
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
  /** Current server-side sort, and a setter for it. Changing it refetches the
   *  list from page 1 in the new order (sorting is done by the backend). */
  sort: ReviewQueueSort;
  onSortChange: (sort: ReviewQueueSort) => void;
  /** Fetch the next page of queues; drives infinite scroll. `queues` only holds
   *  the pages loaded so far, so the filter/sort and per-queue counts reflect
   *  the loaded window. */
  onLoadMore?: () => void;
  hasMore?: boolean;
  isLoadingMore?: boolean;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const authAvailable = useIsAuthAvailable();
  const [mineOnly, setMineOnly] = useState(false);

  // Owner is only meaningful on an auth server; the filter only helps users who
  // can see queues they don't own (i.e. editors).
  const showOwner = authAvailable;
  const showFilter = authAvailable && canEdit;

  const inspectable = (q: ReviewQueue) => canInspectQueue(q, reviewer, canManage, canEdit);

  // One pending-count fetch per inspectable queue (shares the right pane's cache).
  // Non-inspectable queues are skipped (their item list would 403) and show blank.
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

  // Toggle direction when re-clicking the active column, else sort that column
  // ascending. The backend performs the sort, so this only updates the request.
  const requestSort = (field: ReviewQueueSortField) => {
    onSortChange(
      sort.field === field
        ? { field, direction: sort.direction === 'asc' ? 'desc' : 'asc' }
        : { field, direction: 'asc' },
    );
  };

  // Ignore a stale `mineOnly` once the filter is hidden (can't get stuck filtered),
  // and always keep the selected queue visible even when it isn't owned.
  const effectiveMineOnly = showFilter && mineOnly;
  const filtered = effectiveMineOnly
    ? queues.filter((q) => isQueueOwner(q, reviewer) || q.queue_id === selectedQueueId)
    : queues;
  // The server returns queues already in `sort` order; render them as-is. (Client
  // re-sorting would only reorder the loaded window, not the whole list.)
  const visible = filtered;

  // Infinite scroll: pull the next page when the list nears the bottom, and keep
  // pulling while the loaded queues don't fill the scroll area (short first page
  // or a tall pane).
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const fetchMoreOnBottomReached = useInfiniteScrollFetch({
    isFetching: Boolean(isLoadingMore),
    hasNextPage: Boolean(hasMore),
    fetchNextPage: onLoadMore ?? (() => {}),
  });
  useEffect(() => {
    const el = scrollContainerRef.current;
    if (!el || !onLoadMore || !hasMore || isLoadingMore) {
      return;
    }
    if (el.scrollHeight - el.clientHeight < 200) {
      onLoadMore();
    }
  }, [onLoadMore, hasMore, isLoadingMore, visible.length]);

  return (
    <div
      ref={scrollContainerRef}
      onScroll={(e) => fetchMoreOnBottomReached(e.currentTarget)}
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
          {/* Managing questions needs MANAGE; creating a queue only needs EDIT. */}
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
            active={sort.field === 'name'}
            dir={sort.direction}
            align="left"
            onClick={() => requestSort('name')}
          />
          {showOwner && (
            <SortHeader
              label={<FormattedMessage defaultMessage="Owner" description="Review queue sidebar: queue-owner column" />}
              active={sort.field === 'created_by'}
              dir={sort.direction}
              width={OWNER_COL_WIDTH}
              align="left"
              onClick={() => requestSort('created_by')}
            />
          )}
          {/* The "To do" count is derived client-side from the per-queue count
              fetches, so it can't drive a server-side sort — render it as a plain,
              non-sortable column header. */}
          <SortHeader
            label={
              <FormattedMessage
                defaultMessage="To do"
                description="Review queue sidebar: still-to-review count column"
              />
            }
            width={COUNT_COL_WIDTH}
            align="right"
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

      {isLoadingMore && (
        <div css={{ display: 'flex', justifyContent: 'center', padding: theme.spacing.sm }}>
          <Spinner size="small" />
        </div>
      )}
    </div>
  );
};
