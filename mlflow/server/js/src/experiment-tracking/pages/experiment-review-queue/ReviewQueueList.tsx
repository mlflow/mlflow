import { useMemo, useState } from 'react';

import {
  Empty,
  SearchIcon,
  Table,
  TableCell,
  TableHeader,
  TableRow,
  Tag,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

import { displayUser } from './hooks/useReviewer';
import type { ReviewQueueItem, ReviewStatus } from './types';

const CID = 'mlflow.experiment-review-queue.list';

const STATUS_META: Record<ReviewStatus, { color: 'turquoise' | 'lime' | 'charcoal'; rank: number }> = {
  PENDING: { color: 'turquoise', rank: 0 },
  DECLINED: { color: 'charcoal', rank: 1 },
  COMPLETE: { color: 'lime', rank: 2 },
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

type SortKey = 'target_id' | 'status' | 'completed_by' | 'creation_time_ms';
type SortDirection = 'asc' | 'desc';

const COLUMNS: { key: SortKey; label: React.ReactNode; flex: number }[] = [
  {
    key: 'target_id',
    label: <FormattedMessage defaultMessage="Trace" description="Review queue table: trace id column" />,
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
    label: <FormattedMessage defaultMessage="Attached" description="Review queue table: attached-time column" />,
    flex: 1,
  },
];

const sortValue = (item: ReviewQueueItem, key: SortKey): string | number => {
  if (key === 'status') {
    return STATUS_META[item.status].rank;
  }
  if (key === 'completed_by') {
    return item.completed_by ?? '';
  }
  return item[key];
};

export const ReviewQueueList = ({
  items,
  onOpen,
  nowMs,
}: {
  items: ReviewQueueItem[];
  onOpen: (item: ReviewQueueItem) => void;
  nowMs: number;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [sort, setSort] = useState<{ key: SortKey; direction: SortDirection }>({ key: 'status', direction: 'asc' });

  const toggleSort = (key: SortKey) =>
    setSort((prev) =>
      prev.key === key ? { key, direction: prev.direction === 'asc' ? 'desc' : 'asc' } : { key, direction: 'asc' },
    );

  const sortedItems = useMemo(() => {
    return [...items].sort((a, b) => {
      const av = sortValue(a, sort.key);
      const bv = sortValue(b, sort.key);
      const cmp = av < bv ? -1 : av > bv ? 1 : 0;
      return sort.direction === 'asc' ? cmp : -cmp;
    });
  }, [items, sort]);

  if (items.length === 0) {
    return (
      <div css={{ display: 'flex', justifyContent: 'center', width: '100%', padding: theme.spacing.lg }}>
        <Empty
          description={
            <FormattedMessage defaultMessage="No traces in this queue yet." description="Review queue empty state" />
          }
          image={<SearchIcon />}
        />
      </div>
    );
  }

  return (
    <Table>
      <TableRow isHeader>
        {COLUMNS.map((col) => (
          <TableHeader
            key={col.key}
            componentId={`${CID}.header`}
            css={{ flex: col.flex }}
            sortable
            sortDirection={sort.key === col.key ? sort.direction : 'none'}
            onToggleSort={() => toggleSort(col.key)}
          >
            {col.label}
          </TableHeader>
        ))}
      </TableRow>
      {sortedItems.map((item) => (
        <TableRow
          key={item.target_id}
          onClick={() => onOpen(item)}
          css={{ cursor: 'pointer', '&:hover': { backgroundColor: theme.colors.actionDefaultBackgroundHover } }}
        >
          <TableCell css={{ flex: COLUMNS[0].flex }}>
            <Typography.Text bold>{item.target_id}</Typography.Text>
          </TableCell>
          <TableCell css={{ flex: COLUMNS[1].flex }}>
            <StatusTag status={item.status} />
          </TableCell>
          <TableCell css={{ flex: COLUMNS[2].flex }}>
            <Typography.Text color="secondary">
              {item.completed_by ? displayUser(item.completed_by, intl) : '—'}
            </Typography.Text>
          </TableCell>
          <TableCell css={{ flex: COLUMNS[3].flex }}>{formatAgo(item.creation_time_ms, nowMs)}</TableCell>
        </TableRow>
      ))}
    </Table>
  );
};
