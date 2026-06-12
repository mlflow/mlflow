import { useCallback, useMemo, useState } from 'react';

import {
  Button,
  Checkbox,
  DropdownMenu,
  GearIcon,
  NewWindowIcon,
  PlusIcon,
  SearchIcon,
  SegmentedControlButton,
  SegmentedControlGroup,
  Table,
  TableCell,
  TableHeader,
  TableRow,
  Tag,
  TrashIcon,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { useGetTracesById } from '@databricks/web-shared/model-trace-explorer';
import { FormattedMessage, useIntl } from 'react-intl';

import { displayUser } from './hooks/useReviewer';
import { ReviewQueueEmptyState } from './ReviewQueueEmptyState';
import type { ReviewQueueItem, ReviewStatus } from './types';

const CID = 'mlflow.experiment-review-queue.list';

const STATUS_META: Record<ReviewStatus, { color: 'turquoise' | 'lime' | 'charcoal' }> = {
  PENDING: { color: 'turquoise' },
  DECLINED: { color: 'charcoal' },
  COMPLETE: { color: 'lime' },
};

export const StatusTag = ({ status }: { status: ReviewStatus }) => {
  const label: Record<ReviewStatus, React.ReactNode> = {
    PENDING: <FormattedMessage defaultMessage="Needs review" description="Review queue status: pending" />,
    COMPLETE: <FormattedMessage defaultMessage="Complete" description="Review queue status: complete" />,
    DECLINED: <FormattedMessage defaultMessage="Declined" description="Review queue status: declined" />,
  };
  return (
    <Tag componentId={`${CID}.status-tag`} color={STATUS_META[status].color}>
      {label[status]}
    </Tag>
  );
};

const formatAgo = (ms: number, nowMs: number) => {
  const hours = Math.max(1, Math.round((nowMs - ms) / (60 * 60 * 1000)));
  return hours < 24 ? `${hours}h ago` : `${Math.round(hours / 24)}d ago`;
};

type ColumnKey = 'request' | 'response' | 'status' | 'creation_time_ms';
type SortDirection = 'asc' | 'desc' | 'none';
type StatusFilter = 'all' | 'PENDING' | 'completed';

const STATUS_ORDER: Record<ReviewStatus, number> = { PENDING: 0, COMPLETE: 1, DECLINED: 2 };

const COLUMNS: { key: ColumnKey; label: React.ReactNode; flex: number; sortable?: boolean }[] = [
  {
    key: 'request',
    label: <FormattedMessage defaultMessage="Request" description="Review queue table: request column" />,
    flex: 2,
  },
  {
    key: 'response',
    label: <FormattedMessage defaultMessage="Response" description="Review queue table: response column" />,
    flex: 2,
  },
  {
    key: 'status',
    label: <FormattedMessage defaultMessage="Status" description="Review queue table: status column" />,
    flex: 1,
    sortable: true,
  },
  {
    key: 'creation_time_ms',
    label: <FormattedMessage defaultMessage="Date added" description="Review queue table: date-added column" />,
    flex: 1,
    sortable: true,
  },
];

export const ReviewQueueList = ({
  items,
  title,
  questionCount,
  onOpen,
  nowMs,
  latestQuestionCreatedAtMs,
  onRemoveItems,
  isRemovingItems,
  onManageQueue,
  onDeleteQueue,
  onGoToTraces,
}: {
  items: ReviewQueueItem[];
  /** Queue name shown in the header, next to the question count + gear menu. */
  title?: React.ReactNode;
  /** Number of questions the queue asks, shown under the name. */
  questionCount?: number;
  onOpen: (item: ReviewQueueItem) => void;
  nowMs: number;
  /** Newest question's creation time; flags completed traces reviewed before it. */
  latestQuestionCreatedAtMs?: number;
  /** When provided, rows become checkbox-selectable and a delete action appears
   *  so the queue's manager can remove traces from this view. */
  onRemoveItems?: (itemIds: string[]) => void;
  isRemovingItems?: boolean;
  /** When set, the gear menu shows "Manage queue" (and "Delete queue" if `onDeleteQueue` is set). */
  onManageQueue?: () => void;
  onDeleteQueue?: () => void;
  onGoToTraces?: () => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const selectable = Boolean(onRemoveItems);

  const [sortKey, setSortKey] = useState<ColumnKey | null>(null);
  const [sortDir, setSortDir] = useState<SortDirection>('none');
  const [statusFilter, setStatusFilter] = useState<StatusFilter>('all');

  const toggleSort = useCallback(
    (key: ColumnKey) => {
      if (sortKey !== key) {
        setSortKey(key);
        setSortDir('asc');
      } else if (sortDir === 'asc') {
        setSortDir('desc');
      } else {
        setSortKey(null);
        setSortDir('none');
      }
    },
    [sortKey, sortDir],
  );

  const { data: traces } = useGetTracesById(items.map((i) => i.item_id));
  const previewsById = useMemo(() => {
    const map = new Map<string, { input?: string; response?: string }>();
    (traces ?? []).forEach((t) => {
      const id = t?.info?.trace_id;
      if (id) {
        map.set(id, {
          input: t?.info?.request_preview,
          response: t?.info?.response_preview,
        });
      }
    });
    return map;
  }, [traces]);

  const filteredItems = useMemo(() => {
    let result = items;
    if (statusFilter === 'PENDING') {
      result = result.filter((i) => i.status === 'PENDING');
    } else if (statusFilter === 'completed') {
      result = result.filter((i) => i.status !== 'PENDING');
    }
    if (sortKey && sortDir !== 'none') {
      const dir = sortDir === 'asc' ? 1 : -1;
      result = [...result].sort((a, b) => {
        if (sortKey === 'status') {
          return (STATUS_ORDER[a.status] - STATUS_ORDER[b.status]) * dir;
        }
        if (sortKey === 'creation_time_ms') {
          return (a.creation_time_ms - b.creation_time_ms) * dir;
        }
        return 0;
      });
    }
    return result;
  }, [items, statusFilter, sortKey, sortDir]);

  const toDo = useMemo(() => items.filter((i) => i.status === 'PENDING'), [items]);

  const allSelected = filteredItems.length > 0 && filteredItems.every((i) => selected.has(i.item_id));
  const toggleSelect = (itemId: string, checked: boolean) =>
    setSelected((prev) => {
      const next = new Set(prev);
      if (checked) {
        next.add(itemId);
      } else {
        next.delete(itemId);
      }
      return next;
    });
  const toggleAll = (checked: boolean) =>
    setSelected(checked ? new Set(filteredItems.map((i) => i.item_id)) : new Set());
  const handleDelete = () => {
    if (onRemoveItems && selected.size > 0) {
      onRemoveItems([...selected]);
      setSelected(new Set());
    }
  };
  // Changing the filter drops any row selection: a row checked under one filter
  // would otherwise stay in `selected` after being filtered out of view, leaving
  // the delete action targeting rows the user can no longer see.
  const handleStatusFilterChange = (value: StatusFilter) => {
    setStatusFilter(value);
    setSelected(new Set());
  };

  const colFlex = useMemo(() => {
    const map = new Map<ColumnKey, number>();
    COLUMNS.forEach((c) => map.set(c.key, c.flex));
    return map;
  }, []);

  const renderRow = (item: ReviewQueueItem) => {
    const hasNewQuestions =
      item.status === 'COMPLETE' &&
      item.completed_time_ms != null &&
      latestQuestionCreatedAtMs != null &&
      latestQuestionCreatedAtMs > item.completed_time_ms;
    const previews = previewsById.get(item.item_id);
    return (
      <TableRow
        key={item.item_id}
        onClick={() => onOpen(item)}
        css={{ cursor: 'pointer', '&:hover': { backgroundColor: theme.colors.actionDefaultBackgroundHover } }}
      >
        {selectable && (
          <TableCell css={{ flex: '0 0 36px' }} onClick={(e) => e.stopPropagation()}>
            <Checkbox
              componentId={`${CID}.select-row`}
              isChecked={selected.has(item.item_id)}
              onChange={(checked) => toggleSelect(item.item_id, checked)}
            />
          </TableCell>
        )}
        <TableCell css={{ flex: colFlex.get('request') }}>
          <Typography.Text ellipsis color="secondary">
            {previews?.input || '—'}
          </Typography.Text>
        </TableCell>
        <TableCell css={{ flex: colFlex.get('response') }}>
          <Typography.Text ellipsis color="secondary">
            {previews?.response || '—'}
          </Typography.Text>
        </TableCell>
        <TableCell css={{ flex: colFlex.get('status'), display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
          <StatusTag status={item.status} />
          {hasNewQuestions && (
            <Tag componentId={`${CID}.new-questions-tag`} color="lemon">
              <FormattedMessage
                defaultMessage="New questions"
                description="Review queue: badge when a question was added after the trace was completed"
              />
            </Tag>
          )}
        </TableCell>
        <TableCell css={{ flex: colFlex.get('creation_time_ms') }}>{formatAgo(item.creation_time_ms, nowMs)}</TableCell>
      </TableRow>
    );
  };

  return (
    <div css={{ display: 'flex', flexDirection: 'column', height: '100%', minHeight: 0, gap: theme.spacing.sm }}>
      {(title || selectable || onManageQueue || onDeleteQueue) && (
        <div css={{ display: 'flex', alignItems: 'flex-start', gap: theme.spacing.sm }}>
          <div css={{ minWidth: 0 }}>
            <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs / 2, minWidth: 0 }}>
              <Typography.Title level={3} withoutMargins ellipsis css={{ minWidth: 0 }}>
                {title}
              </Typography.Title>
              {(onManageQueue || onDeleteQueue) && (
                <DropdownMenu.Root modal={false}>
                  <DropdownMenu.Trigger asChild>
                    <Button
                      componentId={`${CID}.queue-settings-trigger`}
                      icon={<GearIcon />}
                      aria-label={intl.formatMessage({
                        defaultMessage: 'Queue settings',
                        description: 'Review queue header: queue settings gear button aria label',
                      })}
                    />
                  </DropdownMenu.Trigger>
                  <DropdownMenu.Content align="start">
                    {/* USER queues show only "Delete queue" (no settings); CUSTOM show both. */}
                    {onManageQueue && (
                      <DropdownMenu.Item componentId={`${CID}.manage-queue`} onClick={onManageQueue}>
                        <FormattedMessage
                          defaultMessage="Manage queue"
                          description="Review queue header: manage-queue menu item"
                        />
                      </DropdownMenu.Item>
                    )}
                    {onDeleteQueue && (
                      <DropdownMenu.Item danger componentId={`${CID}.delete-queue`} onClick={onDeleteQueue}>
                        <FormattedMessage
                          defaultMessage="Delete queue"
                          description="Review queue header: delete-queue menu item"
                        />
                      </DropdownMenu.Item>
                    )}
                  </DropdownMenu.Content>
                </DropdownMenu.Root>
              )}
            </div>
            {questionCount != null && (
              <Typography.Text size="sm" color="secondary">
                <FormattedMessage
                  defaultMessage="{count, plural, one {# question} other {# questions}}"
                  description="Review queue header: number of questions the queue asks"
                  values={{ count: questionCount }}
                />
              </Typography.Text>
            )}
          </div>
          <div css={{ flex: 1 }} />
          {toDo.length > 0 && (
            <Button componentId={`${CID}.start-review`} type="primary" onClick={() => onOpen(toDo[0])}>
              <FormattedMessage defaultMessage="Start review" description="Review queue: start-review button" />
            </Button>
          )}
          {selectable && selected.size > 0 && (
            <Button
              componentId={`${CID}.delete-selected`}
              danger
              icon={<TrashIcon />}
              disabled={isRemovingItems}
              loading={isRemovingItems}
              onClick={handleDelete}
            >
              <FormattedMessage
                defaultMessage="Remove {count, plural, one {# trace} other {# traces}}"
                description="Review queue: remove selected traces button"
                values={{ count: selected.size }}
              />
            </Button>
          )}
        </div>
      )}

      {items.length === 0 ? (
        <ReviewQueueEmptyState
          title={
            <FormattedMessage
              defaultMessage="No traces in this queue yet"
              description="Review queue: empty queue title"
            />
          }
          description={
            <FormattedMessage
              defaultMessage="Add traces from the Traces tab to start reviewing them with your team."
              description="Review queue: empty queue description"
            />
          }
          button={
            onGoToTraces && (
              <Button
                componentId={`${CID}.go-to-traces`}
                type="primary"
                icon={<PlusIcon />}
                endIcon={<NewWindowIcon />}
                onClick={onGoToTraces}
              >
                <FormattedMessage
                  defaultMessage="Add traces"
                  description="Review queue: button to navigate to Traces tab"
                />
              </Button>
            )
          }
        />
      ) : (
        <>
          <SegmentedControlGroup
            name={`${CID}.status-filter`}
            componentId={`${CID}.status-filter`}
            value={statusFilter}
            onChange={(event) => handleStatusFilterChange(event.target.value as StatusFilter)}
          >
            <SegmentedControlButton value="all">
              <FormattedMessage
                defaultMessage="All ({count})"
                description="Review queue status filter: all"
                values={{ count: items.length }}
              />
            </SegmentedControlButton>
            <SegmentedControlButton value="PENDING">
              <FormattedMessage
                defaultMessage="Needs review ({count})"
                description="Review queue status filter: needs review"
                values={{ count: toDo.length }}
              />
            </SegmentedControlButton>
            <SegmentedControlButton value="completed">
              <FormattedMessage
                defaultMessage="Completed ({count})"
                description="Review queue status filter: completed"
                values={{ count: items.length - toDo.length }}
              />
            </SegmentedControlButton>
          </SegmentedControlGroup>
          <div css={{ flex: 1, minHeight: 0, overflow: 'auto' }}>
            <Table>
              <TableRow isHeader>
                {selectable && (
                  <TableHeader componentId={`${CID}.select-header`} css={{ flex: '0 0 36px' }}>
                    <Checkbox
                      componentId={`${CID}.select-all`}
                      isChecked={allSelected}
                      onChange={(checked) => toggleAll(checked)}
                    />
                  </TableHeader>
                )}
                {COLUMNS.map((col) => (
                  <TableHeader
                    key={col.key}
                    componentId={`${CID}.header`}
                    css={{ flex: col.flex }}
                    sortable={col.sortable}
                    sortDirection={sortKey === col.key ? sortDir : 'none'}
                    onToggleSort={col.sortable ? () => toggleSort(col.key) : undefined}
                  >
                    {col.label}
                  </TableHeader>
                ))}
              </TableRow>

              {filteredItems.length === 0 ? (
                <TableRow>
                  <TableCell css={{ flex: 1 }}>
                    <Typography.Text color="secondary">
                      <FormattedMessage
                        defaultMessage="No traces match this filter."
                        description="Review queue table: empty state when the active status filter matches no traces"
                      />
                    </Typography.Text>
                  </TableCell>
                </TableRow>
              ) : (
                filteredItems.map(renderRow)
              )}
            </Table>
          </div>
        </>
      )}
    </div>
  );
};
