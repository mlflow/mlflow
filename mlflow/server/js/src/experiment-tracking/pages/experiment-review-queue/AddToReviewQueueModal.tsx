import { useMemo, useState } from 'react';

import {
  Alert,
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
import { useGetOrCreateUserQueueMutation } from './hooks/useGetOrCreateUserQueueMutation';
import { useListReviewQueuesQuery } from './hooks/useListReviewQueuesQuery';
import { DEFAULT_REVIEWER, useReviewer } from './hooks/useReviewer';
import type { ReviewQueue } from './types';

const CID = 'mlflow.experiment-review-queue.add-to-queue';

const RadioGroup = Radio.Group;

// Sentinel for the always-present "my personal queue" option. The queue is
// resolved (get-or-create) only when the user confirms, so it works before
// the queue exists.
const MY_QUEUE = '__my_review_queue__';

/**
 * Picker for routing one or more traces into a review queue. Shown both from
 * the Traces table bulk action and the trace-detail "Flag for review" button
 * (injected into the shared trace UI via the same render-prop mechanism as
 * "Add to evaluation dataset").
 *
 * The reviewer's own personal queue is pinned at the top and pre-selected, so
 * the common case is one click: it's resolved (created on first use) via
 * get-or-create on confirm. Shared CUSTOM queues are listed below; queues that
 * wouldn't present any questions are disabled (see `getQueueAssignability`),
 * and a new queue can be created inline.
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

  // Default to the personal queue so flagging is a single click.
  const [selectedQueueId, setSelectedQueueId] = useState<string>(MY_QUEUE);
  const [createOpen, setCreateOpen] = useState(false);

  const {
    reviewQueues,
    isLoading: queuesLoading,
    error: queuesError,
  } = useListReviewQueuesQuery({ experimentId, enabled: visible });
  const { labelSchemas } = useListLabelSchemasQuery({ experimentId, enabled: visible });
  const {
    addTracesToReviewQueueAsync,
    isAddingTraces,
    error: addError,
    reset: resetAdd,
  } = useAddTracesToReviewQueueMutation();
  const {
    getOrCreateUserQueueAsync,
    isResolvingUserQueue,
    error: resolveError,
    reset: resetResolve,
  } = useGetOrCreateUserQueueMutation();

  const targetIds = useMemo(
    () => selectedTraceInfos.map((info) => info.trace_id).filter((id): id is string => Boolean(id)),
    [selectedTraceInfos],
  );

  // Shared queues anyone can route into. The reviewer's own personal queue is
  // surfaced through the pinned MY_QUEUE option instead of the list, and other
  // users' personal queues are never shown.
  const customQueues = useMemo(() => reviewQueues.filter((q) => q.queue_type === 'CUSTOM'), [reviewQueues]);

  // A USER queue presents every experiment schema, so the personal queue is
  // assignable as soon as the experiment has at least one question.
  const myQueueAssignable = labelSchemas.length > 0;

  const assignabilityById = useMemo(() => {
    const map = new Map<string, ReturnType<typeof getQueueAssignability>>();
    customQueues.forEach((q) => map.set(q.queue_id, getQueueAssignability(q, labelSchemas)));
    return map;
  }, [customQueues, labelSchemas]);

  const selectedAssignable =
    selectedQueueId === MY_QUEUE ? myQueueAssignable : assignabilityById.get(selectedQueueId)?.assignable;
  const actionError = addError ?? resolveError;
  const isWorking = isAddingTraces || isResolvingUserQueue;
  const canAdd = Boolean(selectedQueueId && selectedAssignable && targetIds.length > 0 && !isWorking);

  const handleClose = () => {
    setSelectedQueueId(MY_QUEUE);
    setCreateOpen(false);
    resetAdd();
    resetResolve();
    setVisible(false);
  };

  const handleCreated = (queue: ReviewQueue) => {
    setSelectedQueueId(queue.queue_id);
  };

  const handleAdd = async () => {
    if (!canAdd) {
      return;
    }
    let queueId = selectedQueueId;
    if (selectedQueueId === MY_QUEUE) {
      const { review_queue } = await getOrCreateUserQueueAsync({
        experiment_id: experimentId,
        user: reviewer,
        created_by: reviewer,
      });
      queueId = review_queue.queue_id;
    }
    await addTracesToReviewQueueAsync({ queue_id: queueId, target_ids: targetIds });
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
        okButtonProps={{ disabled: !canAdd, loading: isWorking }}
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
          ) : (
            <RadioGroup
              componentId={`${CID}.queue-picker`}
              name="add-to-review-queue"
              value={selectedQueueId}
              onChange={(e) => setSelectedQueueId(e.target.value)}
            >
              <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
                {/* Pinned, pre-selected personal queue (resolved on confirm). */}
                <div>
                  <Radio value={MY_QUEUE} disabled={!myQueueAssignable} componentId={`${CID}.my-queue-option`}>
                    {reviewer === DEFAULT_REVIEWER ? (
                      <FormattedMessage
                        defaultMessage="Default review queue"
                        description="Add to review queue: personal queue option on a no-auth server"
                      />
                    ) : (
                      <FormattedMessage
                        defaultMessage="My review queue"
                        description="Add to review queue: the reviewer's personal queue option"
                      />
                    )}
                  </Radio>
                  {!myQueueAssignable && (
                    <Typography.Hint css={{ marginLeft: theme.spacing.lg, display: 'block' }}>
                      {reasonText('no-experiment-schemas')}
                    </Typography.Hint>
                  )}
                </div>

                {customQueues.map((q) => {
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

          {actionError && (
            <Alert
              componentId={`${CID}.error`}
              type="error"
              closable={false}
              message={intl.formatMessage({
                defaultMessage: 'Failed to add traces to the review queue.',
                description: 'Add to review queue: error alert title',
              })}
              description={actionError.message}
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
