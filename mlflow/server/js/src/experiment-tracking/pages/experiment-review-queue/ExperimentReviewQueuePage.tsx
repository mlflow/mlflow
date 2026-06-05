import { useMemo, useState } from 'react';

import {
  Button,
  Empty,
  GearIcon,
  SearchIcon,
  Spinner,
  Tooltip,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

import { useListLabelSchemasQuery } from '../../components/label-schemas';
import { useParams } from '../../../common/utils/RoutingUtils';
import { FocusedReview } from './FocusedReview';
import { ManageQuestionsModal } from './ManageQuestionsModal';
import { ReviewQueueList } from './ReviewQueueList';
import { useListReviewQueueTracesQuery } from './hooks/useListReviewQueueTracesQuery';
import { useListReviewQueuesQuery } from './hooks/useListReviewQueuesQuery';
import { useSetReviewQueueTraceStatusMutation } from './hooks/useSetReviewQueueTraceStatusMutation';
import type { ReviewStatus } from './types';

const CID = 'mlflow.experiment-review-queue.page';

/** No-auth OSS has no per-user identity; everyone is the reserved default user. */
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

  const [selectedQueueId, setSelectedQueueId] = useState<string | null>(null);
  const [openTargetId, setOpenTargetId] = useState<string | null>(null);
  const [manageOpen, setManageOpen] = useState(false);

  const { reviewQueues, isLoading: queuesLoading } = useListReviewQueuesQuery({
    experimentId: experimentId ?? '',
    // No-auth OSS scopes to the single reserved default user; passing it keeps
    // the rail from listing every reviewer's personal queue once auth exists.
    user: DEFAULT_REVIEWER,
  });
  const { labelSchemas } = useListLabelSchemasQuery({ experimentId: experimentId ?? '' });
  const { setReviewQueueTraceStatusAsync, isSettingStatus } = useSetReviewQueueTraceStatusMutation();

  const selectedQueue = useMemo(
    () => reviewQueues.find((q) => q.queue_id === selectedQueueId) ?? reviewQueues[0] ?? null,
    [reviewQueues, selectedQueueId],
  );

  const { items, isLoading: tracesLoading } = useListReviewQueueTracesQuery({
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

  const openItem = useMemo(() => items.find((i) => i.target_id === openTargetId) ?? null, [items, openTargetId]);

  const nowMs = Date.now();

  const setOpenStatus = async (status: ReviewStatus) => {
    if (!selectedQueue || !openItem) {
      return;
    }
    await setReviewQueueTraceStatusAsync({
      queue_id: selectedQueue.queue_id,
      target_id: openItem.target_id,
      status,
      // Attribution only applies to the terminal states; reopen clears it.
      completed_by: status === 'PENDING' ? undefined : DEFAULT_REVIEWER,
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
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
        <Typography.Title level={2} withoutMargins>
          <FormattedMessage defaultMessage="Review" description="Review queue tab title" />
        </Typography.Title>
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

      {queuesLoading ? (
        <Spinner />
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
                    setOpenTargetId(null);
                  }}
                >
                  {q.name}
                </Button>
              ))}
            </div>
          )}

          <div css={{ flex: 1, minHeight: 0, overflow: 'auto' }}>
            {openItem ? (
              <FocusedReview
                // Remount per trace so answer state never bleeds across traces.
                key={openItem.target_id}
                item={openItem}
                items={items}
                schemas={questionSchemas}
                completedBy={DEFAULT_REVIEWER}
                isSettingStatus={isSettingStatus}
                onBack={() => setOpenTargetId(null)}
                onSelect={setOpenTargetId}
                onSetStatus={setOpenStatus}
              />
            ) : tracesLoading ? (
              <Spinner />
            ) : (
              <ReviewQueueList items={items} onOpen={(item) => setOpenTargetId(item.target_id)} nowMs={nowMs} />
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
