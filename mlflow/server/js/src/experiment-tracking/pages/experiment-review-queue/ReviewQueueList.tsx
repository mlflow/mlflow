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

import type { ReviewItem, ReviewStatus } from './mockData';

const CID = 'mlflow.experiment-review-queue.list';

const STATUS_META: Record<ReviewStatus, { label: string; color: 'turquoise' | 'lime' | 'charcoal'; rank: number }> = {
  PENDING: { label: 'Needs review', color: 'turquoise', rank: 0 },
  SKIPPED: { label: 'Skipped', color: 'charcoal', rank: 1 },
  COMPLETED: { label: 'Completed', color: 'lime', rank: 2 },
};

export const StatusTag = ({ status }: { status: ReviewStatus }) => {
  const meta = STATUS_META[status];
  return (
    <Tag componentId={`${CID}.status-tag`} color={meta.color}>
      {meta.label}
    </Tag>
  );
};

const formatAgo = (assignedAtMs: number, nowMs: number) => {
  const hours = Math.max(1, Math.round((nowMs - assignedAtMs) / (60 * 60 * 1000)));
  return hours < 24 ? `${hours}h ago` : `${Math.round(hours / 24)}d ago`;
};

type SortKey = 'traceId' | 'requestPreview' | 'assigner' | 'assignedAtMs' | 'status';
type SortDirection = 'asc' | 'desc';

const COLUMNS: { key: SortKey; label: string; flex: number }[] = [
  { key: 'traceId', label: 'Trace', flex: 1.2 },
  { key: 'requestPreview', label: 'Request', flex: 2.5 },
  { key: 'assigner', label: 'Assigner', flex: 1.2 },
  { key: 'assignedAtMs', label: 'Assigned', flex: 1 },
  { key: 'status', label: 'Status', flex: 1 },
];

const sortValue = (item: ReviewItem, key: SortKey): string | number =>
  key === 'status' ? STATUS_META[item.status].rank : item[key];

export const ReviewQueueList = ({
  items,
  onOpen,
  nowMs,
}: {
  items: ReviewItem[];
  onOpen: (item: ReviewItem) => void;
  nowMs: number;
}) => {
  const { theme } = useDesignSystemTheme();
  const [sort, setSort] = useState<{ key: SortKey; direction: SortDirection }>({ key: 'status', direction: 'asc' });

  const toggleSort = (key: SortKey) =>
    setSort((prev) =>
      prev.key === key ? { key, direction: prev.direction === 'asc' ? 'desc' : 'asc' } : { key, direction: 'asc' },
    );

  const sortedItems = useMemo(() => {
    const sorted = [...items].sort((a, b) => {
      const av = sortValue(a, sort.key);
      const bv = sortValue(b, sort.key);
      const cmp = av < bv ? -1 : av > bv ? 1 : 0;
      return sort.direction === 'asc' ? cmp : -cmp;
    });
    return sorted;
  }, [items, sort]);

  if (items.length === 0) {
    return (
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          height: '100%',
          minHeight: 400,
          width: '100%',
          '& > div': { height: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'center' },
        }}
      >
        <Empty description="No traces assigned to this reviewer yet." image={<SearchIcon />} />
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
          key={item.assignmentId}
          onClick={() => onOpen(item)}
          css={{ cursor: 'pointer', '&:hover': { backgroundColor: theme.colors.actionDefaultBackgroundHover } }}
        >
          <TableCell css={{ flex: COLUMNS[0].flex }}>
            <Typography.Text bold>{item.traceId}</Typography.Text>
          </TableCell>
          <TableCell css={{ flex: COLUMNS[1].flex }}>
            <Typography.Text color="secondary" ellipsis>
              {item.requestPreview}
            </Typography.Text>
          </TableCell>
          <TableCell css={{ flex: COLUMNS[2].flex }}>{item.assigner}</TableCell>
          <TableCell css={{ flex: COLUMNS[3].flex }}>{formatAgo(item.assignedAtMs, nowMs)}</TableCell>
          <TableCell css={{ flex: COLUMNS[4].flex }}>
            <StatusTag status={item.status} />
          </TableCell>
        </TableRow>
      ))}
    </Table>
  );
};
