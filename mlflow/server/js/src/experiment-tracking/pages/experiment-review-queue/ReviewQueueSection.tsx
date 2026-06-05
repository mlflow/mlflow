import {
  Button,
  ChevronDownIcon,
  ChevronRightIcon,
  Tag,
  TableSkeleton,
  TrashIcon,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

import { ReviewQueueList } from './ReviewQueueList';
import { useListReviewQueueTracesQuery } from './hooks/useListReviewQueueTracesQuery';
import { displayUser } from './hooks/useReviewer';
import type { ReviewQueue, ReviewQueueItem } from './types';

const CID = 'mlflow.experiment-review-queue.section';

/**
 * One collapsible queue on the Review tab: a header (name, type, pending/total
 * counts, delete) over the queue's own trace table. Each section fetches its
 * own traces, so several queues can be shown at once and collapsed
 * independently to focus on one.
 */
export const ReviewQueueSection = ({
  queue,
  expanded,
  onToggle,
  onOpenTrace,
  onDelete,
  nowMs,
}: {
  queue: ReviewQueue;
  expanded: boolean;
  onToggle: () => void;
  onOpenTrace: (item: ReviewQueueItem) => void;
  onDelete: () => void;
  nowMs: number;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { items, isLoading } = useListReviewQueueTracesQuery({ queueId: queue.queue_id });

  const pending = items.filter((i) => i.status === 'PENDING').length;

  return (
    <div
      css={{
        border: `1px solid ${theme.colors.border}`,
        borderRadius: theme.borders.borderRadiusMd,
        overflow: 'hidden',
      }}
    >
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
          gap: theme.spacing.sm,
          padding: theme.spacing.sm,
          cursor: 'pointer',
          backgroundColor: theme.colors.backgroundSecondary,
        }}
      >
        {expanded ? <ChevronDownIcon /> : <ChevronRightIcon />}
        <Typography.Text bold css={{ flex: 1 }}>
          {queue.queue_type === 'USER' ? displayUser(queue.name, intl) : queue.name}
        </Typography.Text>
        <Tag componentId={`${CID}.type-tag`} color={queue.queue_type === 'USER' ? 'turquoise' : 'lime'}>
          {queue.queue_type === 'USER' ? (
            <FormattedMessage defaultMessage="Personal" description="Review queue: USER queue type tag" />
          ) : (
            <FormattedMessage defaultMessage="Custom" description="Review queue: CUSTOM queue type tag" />
          )}
        </Tag>
        {!isLoading && (
          <Typography.Text color="secondary">
            <FormattedMessage
              defaultMessage="{pending} to review · {total} total"
              description="Review queue: section trace count summary"
              values={{ pending, total: items.length }}
            />
          </Typography.Text>
        )}
        {queue.queue_type === 'CUSTOM' && (
          <Button
            componentId={`${CID}.delete`}
            size="small"
            icon={<TrashIcon />}
            aria-label={intl.formatMessage({
              defaultMessage: 'Delete queue',
              description: 'Review queue: delete-queue button aria label',
            })}
            onClick={(e) => {
              e.stopPropagation();
              onDelete();
            }}
          />
        )}
      </div>

      {expanded && (
        <div css={{ padding: theme.spacing.sm }}>
          {isLoading ? (
            <TableSkeleton lines={3} />
          ) : (
            <ReviewQueueList items={items} onOpen={onOpenTrace} nowMs={nowMs} />
          )}
        </div>
      )}
    </div>
  );
};
