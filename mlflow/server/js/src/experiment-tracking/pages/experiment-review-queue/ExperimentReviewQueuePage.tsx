import { useMemo, useState } from 'react';

import {
  Button,
  Empty,
  GearIcon,
  SearchIcon,
  TableSkeleton,
  Tooltip,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

import { useCurrentUserQuery } from '../../../account/hooks';
import { useListLabelSchemasQuery } from '../../components/label-schemas';
import { useParams } from '../../../common/utils/RoutingUtils';
import { FocusedReview } from './FocusedReview';
import { ManageQuestionsModal } from './ManageQuestionsModal';
import { ReviewQueueList } from './ReviewQueueList';
import { useListReviewQueueItemsQuery } from './hooks/useListReviewQueueItemsQuery';
import { useListReviewQueuesQuery } from './hooks/useListReviewQueuesQuery';
import { useSetReviewQueueItemStatusMutation } from './hooks/useSetReviewQueueItemStatusMutation';
import type { ReviewStatus } from './types';

const CID = 'mlflow.experiment-review-queue.page';

/** Fallback reviewer on a bare no-auth server (no authenticated user). */
const DEFAULT_REVIEWER = 'default';

/**
 * Review tab — a reviewer works a queue's traces and answers its questions.
 *
 * Lists the experiment's review queues, shows the selected queue's traces in
 * a sortable table, and opens a focused 3-panel review (queue rail | trace |
 * question widgets). Answering writes Feedback/Expectation assessments; the
 * complete / decline / reopen actions drive the shared-pool status.
 */
const ExperimentReviewQueuePage = () => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { experimentId } = useParams<{ experimentId: string }>();
  // Authenticated deployments (basic-auth / account plugin) stamp the real
  // reviewer; a bare no-auth server has no `/users/current`, so the query
  // misses and we fall back to the reserved default user.
  const { data: currentUser } = useCurrentUserQuery();
  const reviewer = currentUser?.user?.username || DEFAULT_REVIEWER;

  const [selectedQueueId, setSelectedQueueId] = useState<string | null>(null);
  const [openItemId, setOpenItemId] = useState<string | null>(null);
  const [manageOpen, setManageOpen] = useState(false);

  const { reviewQueues, isLoading: queuesLoading } = useListReviewQueuesQuery({
    experimentId: experimentId ?? '',
    // Scope to the current reviewer so the rail shows their queues (and the
    // single default queue on a no-auth server), not everyone's.
    user: reviewer,
  });
  const { labelSchemas } = useListLabelSchemasQuery({ experimentId: experimentId ?? '' });
  const { setReviewQueueItemStatusAsync, isSettingStatus } = useSetReviewQueueItemStatusMutation();

  const selectedQueue = useMemo(
    () => reviewQueues.find((q) => q.queue_id === selectedQueueId) ?? reviewQueues[0] ?? null,
    [reviewQueues, selectedQueueId],
  );

  const { items, isLoading: itemsLoading } = useListReviewQueueItemsQuery({
    queueId: selectedQueue?.queue_id ?? '',
    enabled: Boolean(selectedQueue),
  });

  // A user queue inherits all of the experiment's schemas; a custom queue
  // uses its explicit subset.
  const questionSchemas = useMemo(() => {
    if (!selectedQueue) {
      return [];
    }
    if (selectedQueue.queue_type === 'USER') {
      return labelSchemas;
    }
    const ids = new Set(selectedQueue.schema_ids ?? []);
    return labelSchemas.filter((s) => ids.has(s.schema_id));
  }, [selectedQueue, labelSchemas]);

  const openItem = useMemo(() => items.find((i) => i.item_id === openItemId) ?? null, [items, openItemId]);

  const nowMs = Date.now();

  const setOpenStatus = async (status: ReviewStatus) => {
    if (!selectedQueue || !openItem) {
      return;
    }
    await setReviewQueueItemStatusAsync({
      queue_id: selectedQueue.queue_id,
      item_id: openItem.item_id,
      status,
      // Attribution only applies to the terminal states; reopen clears it.
      completed_by: status === 'PENDING' ? undefined : reviewer,
    });
  };

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.md,
        height: '100%',
        padding: theme.spacing.md,
      }}
    >
      {queuesLoading ? (
        <TableSkeleton lines={5} />
      ) : reviewQueues.length === 0 ? (
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
          <Empty
            description={
              <FormattedMessage
                defaultMessage="No review queues yet. Flag traces for review to create one."
                description="Review queue: empty state when no queues exist"
              />
            }
            image={<SearchIcon />}
          />
        </div>
      ) : (
        <>
          {!openItem && (
            <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
              <Typography.Text color="secondary">
                <FormattedMessage defaultMessage="Queue:" description="Review queue: queue selector label" />
              </Typography.Text>
              {reviewQueues.map((q) => (
                <Button
                  key={q.queue_id}
                  componentId={`${CID}.select-queue`}
                  size="small"
                  type={q.queue_id === selectedQueue?.queue_id ? 'primary' : undefined}
                  onClick={() => {
                    setSelectedQueueId(q.queue_id);
                    setOpenItemId(null);
                  }}
                >
                  {q.name}
                </Button>
              ))}
              <div css={{ flex: 1 }} />
              <Tooltip
                componentId={`${CID}.edit-questions-tooltip`}
                content={intl.formatMessage({
                  defaultMessage: 'Edit review questions for this experiment',
                  description: 'Review queue: edit-questions gear tooltip',
                })}
              >
                <Button
                  componentId={`${CID}.edit-questions`}
                  icon={<GearIcon />}
                  aria-label={intl.formatMessage({
                    defaultMessage: 'Edit review questions',
                    description: 'Review queue: edit-questions gear aria label',
                  })}
                  onClick={() => setManageOpen(true)}
                />
              </Tooltip>
            </div>
          )}

          <div css={{ flex: 1, minHeight: 0, overflow: 'auto' }}>
            {openItem ? (
              <FocusedReview
                // Remount per trace so answer state never bleeds across traces.
                key={openItem.item_id}
                item={openItem}
                items={items}
                schemas={questionSchemas}
                completedBy={reviewer}
                isSettingStatus={isSettingStatus}
                onBack={() => setOpenItemId(null)}
                onSelect={setOpenItemId}
                onSetStatus={setOpenStatus}
              />
            ) : itemsLoading ? (
              <TableSkeleton lines={5} />
            ) : (
              <ReviewQueueList items={items} onOpen={(item) => setOpenItemId(item.item_id)} nowMs={nowMs} />
            )}
          </div>
        </>
      )}

      {manageOpen && experimentId && (
        <ManageQuestionsModal experimentId={experimentId} onClose={() => setManageOpen(false)} />
      )}
    </div>
  );
};

export default ExperimentReviewQueuePage;
