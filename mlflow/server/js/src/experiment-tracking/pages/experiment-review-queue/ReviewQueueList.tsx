import { useMemo, useState } from 'react';

import {
  Button,
  Checkbox,
  ChevronDownIcon,
  ChevronRightIcon,
  Empty,
  SearchIcon,
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

// Sort items newest-added first, breaking ties (a bulk add shares one add-time)
// by the trace's own creation time, newest first.
const byAddedThenTrace = (traceTimeMsById: Map<string, number>) => (a: ReviewQueueItem, b: ReviewQueueItem) =>
  b.creation_time_ms - a.creation_time_ms ||
  (traceTimeMsById.get(b.target_id) ?? 0) - (traceTimeMsById.get(a.target_id) ?? 0);

type ColumnKey = 'target_id' | 'status' | 'completed_by' | 'creation_time_ms';

const COLUMNS: { key: ColumnKey; label: React.ReactNode; flex: number }[] = [
  {
    key: 'target_id',
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
  onOpen,
  nowMs,
  latestQuestionCreatedAtMs,
  onRemoveTraces,
  isRemovingTraces,
}: {
  items: ReviewQueueItem[];
  /** Queue name shown in the header, next to the delete-traces action. */
  title?: React.ReactNode;
  onOpen: (item: ReviewQueueItem) => void;
  nowMs: number;
  /** Newest question's creation time; flags completed traces reviewed before it. */
  latestQuestionCreatedAtMs?: number;
  /** When provided, rows become checkbox-selectable and a delete action appears
   *  so the queue's manager can remove traces from this view. */
  onRemoveTraces?: (targetIds: string[]) => void;
  isRemovingTraces?: boolean;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [completedOpen, setCompletedOpen] = useState(false);
  const selectable = Boolean(onRemoveTraces);

  // The trace output preview keeps rows human-readable — the raw trace id isn't.
  const { data: traces } = useGetTracesById(items.map((i) => i.target_id));
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
  // The trace's own creation time, used as a secondary sort key (a bulk add
  // gives every item the same queue add-time, so that alone leaves ties).
  const traceTimeMsById = useMemo(() => {
    const map = new Map<string, number>();
    (traces ?? []).forEach((t) => {
      const id = t?.info?.trace_id;
      const ms = t?.info?.request_time ? Date.parse(t.info.request_time) : NaN;
      if (id && !Number.isNaN(ms)) {
        map.set(id, ms);
      }
    });
    return map;
  }, [traces]);

  // Split into still-to-review vs. resolved (complete/declined). Newest-added
  // first; ties (e.g. a single bulk add) break by the trace's own creation
  // time, newest first.
  const toDo = useMemo(
    () => items.filter((i) => i.status === 'PENDING').sort(byAddedThenTrace(traceTimeMsById)),
    [items, traceTimeMsById],
  );
  const completed = useMemo(
    () => items.filter((i) => i.status !== 'PENDING').sort(byAddedThenTrace(traceTimeMsById)),
    [items, traceTimeMsById],
  );

  // Select-all only covers the rows currently visible: "To do" always, plus
  // "Completed" only when that group is expanded.
  const visibleItems = useMemo(() => [...toDo, ...(completedOpen ? completed : [])], [toDo, completed, completedOpen]);
  const allSelected = visibleItems.length > 0 && visibleItems.every((i) => selected.has(i.target_id));
  const toggleSelect = (targetId: string, checked: boolean) =>
    setSelected((prev) => {
      const next = new Set(prev);
      if (checked) {
        next.add(targetId);
      } else {
        next.delete(targetId);
      }
      return next;
    });
  const toggleAll = (checked: boolean) =>
    setSelected(checked ? new Set(visibleItems.map((i) => i.target_id)) : new Set());
  const handleDelete = () => {
    if (onRemoveTraces && selected.size > 0) {
      onRemoveTraces([...selected]);
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
    const previewText = previewById.get(item.target_id) ?? item.target_id;
    return (
      <TableRow
        key={item.target_id}
        onClick={() => onOpen(item)}
        css={{ cursor: 'pointer', '&:hover': { backgroundColor: theme.colors.actionDefaultBackgroundHover } }}
      >
        {selectable && (
          <TableCell css={{ flex: '0 0 36px' }} onClick={(e) => e.stopPropagation()}>
            <Checkbox
              componentId={`${CID}.select-row`}
              isChecked={selected.has(item.target_id)}
              onChange={(checked) => toggleSelect(item.target_id, checked)}
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
      {(title || selectable) && (
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
          <Typography.Title level={4} withoutMargins css={{ flex: 1, minWidth: 0 }}>
            {title}
          </Typography.Title>
          {selectable && (
            <Button
              componentId={`${CID}.delete-selected`}
              danger
              icon={<TrashIcon />}
              disabled={selected.size === 0 || isRemovingTraces}
              loading={isRemovingTraces}
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
        <div css={{ display: 'flex', justifyContent: 'center', width: '100%', padding: theme.spacing.lg }}>
          <Empty
            description={
              <FormattedMessage defaultMessage="No traces in this queue yet." description="Review queue empty state" />
            }
            image={<SearchIcon />}
          />
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
