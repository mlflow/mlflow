import { Empty, SearchIcon, Tag, Typography, useDesignSystemTheme } from '@databricks/design-system';

import type { ReviewItem, ReviewStatus } from './mockData';

const CID = 'mlflow.experiment-review-queue.list';

const STATUS_META: Record<ReviewStatus, { label: string; color: 'turquoise' | 'lime' | 'charcoal' }> = {
  PENDING: { label: 'Needs review', color: 'turquoise' },
  COMPLETED: { label: 'Completed', color: 'lime' },
  SKIPPED: { label: 'Skipped', color: 'charcoal' },
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

const ItemRow = ({ item, onOpen, nowMs }: { item: ReviewItem; onOpen: () => void; nowMs: number }) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div
      role="button"
      tabIndex={0}
      onClick={onOpen}
      onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && onOpen()}
      css={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: theme.spacing.md,
        padding: theme.spacing.md,
        border: `1px solid ${theme.colors.border}`,
        borderRadius: theme.borders.borderRadiusMd,
        cursor: 'pointer',
        '&:hover': { backgroundColor: theme.colors.actionDefaultBackgroundHover },
      }}
    >
      <div css={{ minWidth: 0 }}>
        <Typography.Text bold>{item.traceId}</Typography.Text>
        <Typography.Text color="secondary" css={{ display: 'block' }} ellipsis>
          {item.requestPreview}
        </Typography.Text>
      </div>
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.md, flexShrink: 0 }}>
        <Typography.Hint>
          assigned by {item.assigner} · {formatAgo(item.assignedAtMs, nowMs)}
        </Typography.Hint>
        <StatusTag status={item.status} />
      </div>
    </div>
  );
};

const Section = ({
  title,
  items,
  onOpen,
  nowMs,
}: {
  title: string;
  items: ReviewItem[];
  onOpen: (item: ReviewItem) => void;
  nowMs: number;
}) => {
  const { theme } = useDesignSystemTheme();
  if (items.length === 0) {
    return null;
  }
  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
      <Typography.Title level={4} withoutMargins>
        {title} ({items.length})
      </Typography.Title>
      {items.map((item) => (
        <ItemRow key={item.assignmentId} item={item} onOpen={() => onOpen(item)} nowMs={nowMs} />
      ))}
    </div>
  );
};

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
  const byStatus = (status: ReviewStatus) => items.filter((i) => i.status === status);

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
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.lg }}>
      <Section title="Needs review" items={byStatus('PENDING')} onOpen={onOpen} nowMs={nowMs} />
      <Section title="Skipped" items={byStatus('SKIPPED')} onOpen={onOpen} nowMs={nowMs} />
      <Section title="Completed" items={byStatus('COMPLETED')} onOpen={onOpen} nowMs={nowMs} />
    </div>
  );
};
