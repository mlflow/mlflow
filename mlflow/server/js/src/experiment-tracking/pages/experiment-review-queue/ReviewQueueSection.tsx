import {
  Button,
  ChevronDownIcon,
  ChevronRightIcon,
  PencilIcon,
  Tag,
  TableSkeleton,
  Tooltip,
  TrashIcon,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { useState } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';

import type { LabelSchema } from '../../components/label-schemas';
import { EditQueueQuestionsModal } from './EditQueueQuestionsModal';
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
  labelSchemas,
  expanded,
  onToggle,
  onOpenTrace,
  onDelete,
  nowMs,
}: {
  queue: ReviewQueue;
  labelSchemas: LabelSchema[];
  expanded: boolean;
  onToggle: () => void;
  onOpenTrace: (item: ReviewQueueItem) => void;
  onDelete: () => void;
  nowMs: number;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { items, isLoading } = useListReviewQueueTracesQuery({ queueId: queue.queue_id });

  const [editOpen, setEditOpen] = useState(false);
  const pending = items.filter((i) => i.status === 'PENDING').length;
  // Questions are editable only while the queue is empty (the server freezes
  // them once traces are assigned).
  const canEditQuestions = !isLoading && items.length === 0;

  // A user queue inherits every experiment schema; a custom queue uses its
  // attached subset (resolved against existing schemas, dropping any dangling).
  const questionNames =
    queue.queue_type === 'USER'
      ? labelSchemas.map((s) => s.name)
      : labelSchemas.filter((s) => (queue.schema_ids ?? []).includes(s.schema_id)).map((s) => s.name);

  return (
    <>
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
          {queue.queue_type === 'CUSTOM' &&
            (canEditQuestions ? (
              <Button
                componentId={`${CID}.edit-questions`}
                size="small"
                icon={<PencilIcon />}
                aria-label={intl.formatMessage({
                  defaultMessage: 'Edit questions',
                  description: 'Review queue: edit-questions button aria label',
                })}
                onClick={(e) => {
                  e.stopPropagation();
                  setEditOpen(true);
                }}
              />
            ) : (
              <Tooltip
                componentId={`${CID}.edit-questions-locked-tooltip`}
                content={intl.formatMessage({
                  defaultMessage: 'Questions lock once traces are assigned to the queue.',
                  description: 'Review queue: edit-questions disabled reason',
                })}
              >
                <span onClick={(e) => e.stopPropagation()}>
                  <Button
                    componentId={`${CID}.edit-questions`}
                    size="small"
                    icon={<PencilIcon />}
                    disabled
                    aria-label={intl.formatMessage({
                      defaultMessage: 'Edit questions',
                      description: 'Review queue: edit-questions button aria label',
                    })}
                  />
                </span>
              </Tooltip>
            ))}
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

        <div
          css={{
            paddingLeft: theme.spacing.lg,
            paddingRight: theme.spacing.sm,
            paddingBottom: theme.spacing.sm,
          }}
        >
          <Typography.Hint css={{ display: 'block' }}>
            {questionNames.length > 0 ? (
              <FormattedMessage
                defaultMessage="Questions: {names}"
                description="Review queue: section questions summary"
                values={{ names: questionNames.join(', ') }}
              />
            ) : (
              <FormattedMessage
                defaultMessage="No questions"
                description="Review queue: section no-questions summary"
              />
            )}
          </Typography.Hint>
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

      {editOpen && <EditQueueQuestionsModal queue={queue} onClose={() => setEditOpen(false)} />}
    </>
  );
};
