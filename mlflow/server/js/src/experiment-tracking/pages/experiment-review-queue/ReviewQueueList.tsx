import { useMemo, useState } from 'react';

import {
  Button,
  Checkbox,
  ChecklistIcon,
  ChevronDownIcon,
  ChevronRightIcon,
  DropdownMenu,
  GearIcon,
  PlusIcon,
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
  return hours < 24 ? `${hours}h` : `${Math.round(hours / 24)}d`;
};

type ColumnKey = 'item_id' | 'status' | 'completed_by' | 'creation_time_ms';

const COLUMNS: { key: ColumnKey; label: React.ReactNode; flex: number }[] = [
  {
    key: 'item_id',
    label: <FormattedMessage defaultMessage="Trace" description="Review queue table: trace column" />,
    flex: 2,
  },
  {
    key: 'status',
    label: <FormattedMessage defaultMessage="Status" description="Review queue table: status column" />,
    flex: 1,
  },
  {
    key: 'completed_by',
    label: <FormattedMessage defaultMessage="Completed by" description="Review queue table: completed-by column" />,
    flex: 1.5,
  },
  {
    key: 'creation_time_ms',
    label: <FormattedMessage defaultMessage="Date added" description="Review queue table: date-added column" />,
    flex: 1,
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
  /** When provided (editable non-default custom queues only), a gear menu offers
   *  "Manage queue"; `onDeleteQueue`, when also provided, adds "Delete queue". */
  onManageQueue?: () => void;
  onDeleteQueue?: () => void;
  onGoToTraces?: () => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [completedOpen, setCompletedOpen] = useState(false);
  const selectable = Boolean(onRemoveItems);

  // The trace output preview keeps rows human-readable — the raw trace id isn't.
  const { data: traces } = useGetTracesById(items.map((i) => i.item_id));
  const previewById = useMemo(() => {
    const map = new Map<string, string>();
    (traces ?? []).forEach((t) => {
      const id = t?.info?.trace_id;
      const preview = t?.info?.response_preview || t?.info?.request_preview;
      if (id && preview) {
        map.set(id, preview);
      }
    });
    return map;
  }, [traces]);
  // Split into still-to-review vs. resolved (complete/declined). The page passes
  // items already ordered (completed first, then to-do), so filtering preserves
  // that order — keeping the list in sync with the focused view's prev/next.
  const toDo = useMemo(() => items.filter((i) => i.status === 'PENDING'), [items]);
  const completed = useMemo(() => items.filter((i) => i.status !== 'PENDING'), [items]);

  // Select-all only covers the rows currently visible: "To do" always, plus
  // "Completed" only when that group is expanded.
  const visibleItems = useMemo(() => [...toDo, ...(completedOpen ? completed : [])], [toDo, completed, completedOpen]);
  const allSelected = visibleItems.length > 0 && visibleItems.every((i) => selected.has(i.item_id));
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
    setSelected(checked ? new Set(visibleItems.map((i) => i.item_id)) : new Set());
  const handleDelete = () => {
    if (onRemoveItems && selected.size > 0) {
      onRemoveItems([...selected]);
      setSelected(new Set());
    }
  };

  const groupBand = (label: React.ReactNode, count: number, collapse?: { open: boolean; onToggle: () => void }) => (
    <TableRow
      onClick={collapse?.onToggle}
      css={{
        backgroundColor: theme.colors.backgroundSecondary,
        cursor: collapse ? 'pointer' : 'default',
      }}
    >
      <TableCell css={{ flex: 1, display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
        {collapse && (collapse.open ? <ChevronDownIcon /> : <ChevronRightIcon />)}
        <Typography.Text bold>{label}</Typography.Text>
        <Typography.Text color="secondary">({count})</Typography.Text>
      </TableCell>
    </TableRow>
  );

  const renderRow = (item: ReviewQueueItem) => {
    // A question was added after this trace was completed: it was reviewed
    // without ever seeing that question.
    const hasNewQuestions =
      item.status === 'COMPLETE' &&
      item.completed_time_ms != null &&
      latestQuestionCreatedAtMs != null &&
      latestQuestionCreatedAtMs > item.completed_time_ms;
    const previewText = previewById.get(item.item_id) ?? item.item_id;
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
        <TableCell css={{ flex: COLUMNS[0].flex }}>
          <Typography.Text ellipsis>{previewText}</Typography.Text>
        </TableCell>
        <TableCell css={{ flex: COLUMNS[1].flex, display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
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
        <TableCell css={{ flex: COLUMNS[2].flex }}>
          <Typography.Text color="secondary">
            {item.completed_by ? displayUser(item.completed_by, intl) : '—'}
          </Typography.Text>
        </TableCell>
        <TableCell css={{ flex: COLUMNS[3].flex }}>{formatAgo(item.creation_time_ms, nowMs)}</TableCell>
      </TableRow>
    );
  };

  return (
    <div css={{ display: 'flex', flexDirection: 'column', height: '100%', minHeight: 0, gap: theme.spacing.sm }}>
      {(title || selectable || onManageQueue) && (
        <div css={{ display: 'flex', alignItems: 'flex-start', gap: theme.spacing.sm }}>
          <div css={{ minWidth: 0 }}>
            <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs / 2, minWidth: 0 }}>
              <Typography.Title level={3} withoutMargins ellipsis css={{ minWidth: 0 }}>
                {title}
              </Typography.Title>
              {onManageQueue && (
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
                    <DropdownMenu.Item componentId={`${CID}.manage-queue`} onClick={onManageQueue}>
                      <FormattedMessage
                        defaultMessage="Manage queue"
                        description="Review queue header: manage-queue menu item"
                      />
                    </DropdownMenu.Item>
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
          {selectable && (
            <Button
              componentId={`${CID}.delete-selected`}
              danger
              icon={<TrashIcon />}
              disabled={selected.size === 0 || isRemovingItems}
              loading={isRemovingItems}
              onClick={handleDelete}
            >
              <FormattedMessage
                defaultMessage="Delete {count, plural, one {# trace} other {# traces}}"
                description="Review queue: delete selected traces button"
                values={{ count: selected.size }}
              />
            </Button>
          )}
        </div>
      )}

      {items.length === 0 ? (
        <div
          css={{
            display: 'flex',
            flex: 1,
            flexDirection: 'column',
            justifyContent: 'center',
            alignItems: 'center',
            height: '100%',
            minHeight: 400,
            padding: theme.spacing.md,
          }}
        >
          <ChecklistIcon css={{ fontSize: 48, color: theme.colors.textSecondary, marginBottom: theme.spacing.md }} />
          <Typography.Title level={3} color="secondary">
            <FormattedMessage
              defaultMessage="No traces in this queue yet"
              description="Review queue: empty queue title"
            />
          </Typography.Title>
          <Typography.Paragraph
            color="secondary"
            css={{ maxWidth: 520, textAlign: 'center', marginBottom: theme.spacing.md }}
          >
            <FormattedMessage
              defaultMessage="Add traces from the Traces tab to start reviewing them with your team."
              description="Review queue: empty queue description"
            />
          </Typography.Paragraph>
          {onGoToTraces && (
            <Button
              componentId={`${CID}.go-to-traces`}
              type="primary"
              icon={<PlusIcon />}
              onClick={onGoToTraces}
            >
              <FormattedMessage
                defaultMessage="Add traces"
                description="Review queue: button to navigate to Traces tab"
              />
            </Button>
          )}
        </div>
      ) : (
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
                <TableHeader key={col.key} componentId={`${CID}.header`} css={{ flex: col.flex }}>
                  {col.label}
                </TableHeader>
              ))}
            </TableRow>

            {groupBand(
              <FormattedMessage defaultMessage="To do" description="Review queue table: to-do group label" />,
              toDo.length,
            )}
            {toDo.length === 0 ? (
              <TableRow>
                <TableCell css={{ flex: 1 }}>
                  <Typography.Text color="secondary">
                    <FormattedMessage
                      defaultMessage="Nothing left to review."
                      description="Review queue table: empty to-do group"
                    />
                  </Typography.Text>
                </TableCell>
              </TableRow>
            ) : (
              toDo.map(renderRow)
            )}

            {/* To-do is shown first here; the queue's review order (completed
                first, then to-do) lives in ExperimentReviewQueuePage and drives
                the focused view's Prev/Next, not this grouping. */}
            {completed.length > 0 &&
              groupBand(
                <FormattedMessage defaultMessage="Completed" description="Review queue table: completed group label" />,
                completed.length,
                { open: completedOpen, onToggle: () => setCompletedOpen((open) => !open) },
              )}
            {completedOpen && completed.map(renderRow)}
          </Table>
        </div>
      )}
    </div>
  );
};
