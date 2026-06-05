import { useMemo, useState } from 'react';

import {
  Alert,
  Empty,
  Modal,
  PlusIcon,
  Radio,
  TableSkeleton,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import type { ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';
import { FormattedMessage, useIntl } from 'react-intl';

import { useListLabelSchemasQuery } from '../../components/label-schemas';
import { CreateReviewQueueModal } from './CreateReviewQueueModal';
import { getQueueAssignability } from './queueAssignability';
import { useAddTracesToReviewQueueMutation } from './hooks/useAddTracesToReviewQueueMutation';
import { useListReviewQueuesQuery } from './hooks/useListReviewQueuesQuery';
import { useReviewer } from './hooks/useReviewer';
import type { ReviewQueue } from './types';

const CID = 'mlflow.experiment-review-queue.add-to-queue';

const RadioGroup = Radio.Group;

/**
 * Picker for routing one or more traces into a review queue. Shown both from
 * the Traces table bulk action and the trace-detail "Flag for review" button
 * (injected into the shared trace UI via the same render-prop mechanism as
 * "Add to evaluation dataset"). Queues that wouldn't present any questions
 * are disabled (see `getQueueAssignability`), and a new queue can be created
 * inline.
 */
export const AddToReviewQueueModal = ({
  experimentId,
  visible,
  setVisible,
  selectedTraceInfos,
}: {
  experimentId: string;
  visible: boolean;
  setVisible: (visible: boolean) => void;
  selectedTraceInfos: ModelTraceInfoV3[];
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const reviewer = useReviewer();

  const [selectedQueueId, setSelectedQueueId] = useState<string | null>(null);
  const [createOpen, setCreateOpen] = useState(false);

  const {
    reviewQueues,
    isLoading: queuesLoading,
    error: queuesError,
  } = useListReviewQueuesQuery({ experimentId, enabled: visible });
  const { labelSchemas } = useListLabelSchemasQuery({ experimentId, enabled: visible });
  const { addTracesToReviewQueueAsync, isAddingTraces, error, reset } = useAddTracesToReviewQueueMutation();

  // A reviewer can route traces into any shared CUSTOM queue, but only into
  // their own personal USER queue, never someone else's. (The Review tab
  // scopes its own list to `user: reviewer` for the same reason.)
  const visibleQueues = useMemo(
    () => reviewQueues.filter((q) => q.queue_type === 'CUSTOM' || q.name === reviewer),
    [reviewQueues, reviewer],
  );

  const targetIds = useMemo(
    () => selectedTraceInfos.map((info) => info.trace_id).filter((id): id is string => Boolean(id)),
    [selectedTraceInfos],
  );

  const assignabilityById = useMemo(() => {
    const map = new Map<string, ReturnType<typeof getQueueAssignability>>();
    visibleQueues.forEach((q) => map.set(q.queue_id, getQueueAssignability(q, labelSchemas)));
    return map;
  }, [visibleQueues, labelSchemas]);

  const selectedAssignable = selectedQueueId ? assignabilityById.get(selectedQueueId)?.assignable : false;
  const canAdd = Boolean(selectedQueueId && selectedAssignable && targetIds.length > 0 && !isAddingTraces);

  const handleClose = () => {
    setSelectedQueueId(null);
    setCreateOpen(false);
    reset();
    setVisible(false);
  };

  const handleCreated = (queue: ReviewQueue) => {
    setSelectedQueueId(queue.queue_id);
  };

  const handleAdd = async () => {
    if (!canAdd || !selectedQueueId) {
      return;
    }
    await addTracesToReviewQueueAsync({ queue_id: selectedQueueId, target_ids: targetIds });
    handleClose();
  };

  const reasonText = (reason: ReturnType<typeof getQueueAssignability>['reason']) => {
    switch (reason) {
      case 'no-experiment-schemas':
        return intl.formatMessage({
          defaultMessage: 'No questions defined for this experiment yet.',
          description: 'Add to review queue: user queue disabled reason',
        });
      case 'no-resolvable-schemas':
      default:
        return intl.formatMessage({
          defaultMessage: 'This queue has no questions attached.',
          description: 'Add to review queue: custom queue disabled reason',
        });
    }
  };

  return (
    <>
      <Modal
        componentId={`${CID}.modal`}
        visible={visible}
        title={
          <FormattedMessage
            defaultMessage="Add {count, plural, one {# trace} other {# traces}} to a review queue"
            description="Add to review queue modal title"
            values={{ count: targetIds.length }}
          />
        }
        okText={<FormattedMessage defaultMessage="Add" description="Add to review queue: confirm button" />}
        okButtonProps={{ disabled: !canAdd, loading: isAddingTraces }}
        cancelText={<FormattedMessage defaultMessage="Cancel" description="Add to review queue: cancel button" />}
        onOk={handleAdd}
        onCancel={handleClose}
      >
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
          {queuesError ? (
            <Alert
              componentId={`${CID}.load-error`}
              type="error"
              closable={false}
              message={intl.formatMessage({
                defaultMessage: 'Failed to load review queues.',
                description: 'Add to review queue: queue list load error',
              })}
              description={queuesError.message}
            />
          ) : queuesLoading ? (
            <TableSkeleton lines={4} />
          ) : visibleQueues.length === 0 ? (
            <Empty
              description={
                <FormattedMessage
                  defaultMessage="No review queues yet. Create one to get started."
                  description="Add to review queue: empty state"
                />
              }
            />
          ) : (
            <RadioGroup
              componentId={`${CID}.queue-picker`}
              name="add-to-review-queue"
              value={selectedQueueId ?? undefined}
              onChange={(e) => setSelectedQueueId(e.target.value)}
            >
              <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
                {visibleQueues.map((q) => {
                  const assignability = assignabilityById.get(q.queue_id);
                  const disabled = !assignability?.assignable;
                  return (
                    <div key={q.queue_id}>
                      <Radio value={q.queue_id} disabled={disabled} componentId={`${CID}.queue-option`}>
                        {q.name}
                      </Radio>
                      {disabled && (
                        <Typography.Hint css={{ marginLeft: theme.spacing.lg, display: 'block' }}>
                          {reasonText(assignability?.reason)}
                        </Typography.Hint>
                      )}
                    </div>
                  );
                })}
              </div>
            </RadioGroup>
          )}

          <Typography.Link
            componentId={`${CID}.new-queue`}
            css={{ display: 'inline-flex', alignItems: 'center', gap: theme.spacing.xs }}
            onClick={() => setCreateOpen(true)}
          >
            <PlusIcon />
            <FormattedMessage defaultMessage="New queue" description="Add to review queue: create new queue link" />
          </Typography.Link>

          {error && (
            <Alert
              componentId={`${CID}.error`}
              type="error"
              closable={false}
              message={intl.formatMessage({
                defaultMessage: 'Failed to add traces to the review queue.',
                description: 'Add to review queue: error alert title',
              })}
              description={error.message}
            />
          )}
        </div>
      </Modal>

      {createOpen && (
        <CreateReviewQueueModal
          experimentId={experimentId}
          onClose={() => setCreateOpen(false)}
          onCreated={handleCreated}
        />
      )}
    </>
  );
};
