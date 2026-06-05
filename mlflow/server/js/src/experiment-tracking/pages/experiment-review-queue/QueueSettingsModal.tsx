import { useState } from 'react';

import {
  Alert,
  Button,
  Checkbox,
  Empty,
  Modal,
  TableSkeleton,
  Tabs,
  TrashIcon,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

import type { LabelSchema } from '../../components/label-schemas';
import { StatusTag } from './ReviewQueueList';
import { useListReviewQueueTracesQuery } from './hooks/useListReviewQueueTracesQuery';
import { useRemoveTracesFromReviewQueueMutation } from './hooks/useRemoveTracesFromReviewQueueMutation';
import { useUpdateReviewQueueMutation } from './hooks/useUpdateReviewQueueMutation';
import type { ReviewQueue } from './types';

const CID = 'mlflow.experiment-review-queue.settings';

/**
 * Per-queue settings, opened from the queue's gear. Two tabs:
 *   - Questions — edit which label schemas the queue asks (CUSTOM queues only,
 *     and only while empty; personal queues inherit all of the experiment's).
 *   - Traces — the queue's attached traces, each removable. Removal is kept
 *     here (behind settings) rather than in the browse list so it's deliberate.
 */
export const QueueSettingsModal = ({
  queue,
  labelSchemas,
  onClose,
}: {
  queue: ReviewQueue;
  labelSchemas: LabelSchema[];
  onClose: () => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const { items, isLoading: tracesLoading } = useListReviewQueueTracesQuery({ queueId: queue.queue_id });
  const { updateReviewQueueAsync, isUpdatingQueue, error: updateError } = useUpdateReviewQueueMutation();
  const {
    removeTracesFromReviewQueue,
    isRemovingTraces,
    error: removeError,
  } = useRemoveTracesFromReviewQueueMutation();

  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set(queue.schema_ids ?? []));

  const isUser = queue.queue_type === 'USER';
  const hasTraces = items.length > 0;
  const canEditQuestions = !isUser && !tracesLoading && !hasTraces;
  const attachedQuestions = isUser
    ? labelSchemas
    : labelSchemas.filter((s) => (queue.schema_ids ?? []).includes(s.schema_id));

  const toggle = (schemaId: string, checked: boolean) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (checked) {
        next.add(schemaId);
      } else {
        next.delete(schemaId);
      }
      return next;
    });
  };

  const handleSaveQuestions = async () => {
    if (selectedIds.size === 0 || isUpdatingQueue) {
      return;
    }
    await updateReviewQueueAsync({ queue_id: queue.queue_id, schema_ids: [...selectedIds] });
  };

  return (
    <Modal
      componentId={`${CID}.modal`}
      visible
      title={
        <FormattedMessage
          defaultMessage='Queue settings — "{name}"'
          description="Queue settings modal title"
          values={{ name: queue.name }}
        />
      }
      footer={
        <Button componentId={`${CID}.close`} onClick={onClose}>
          <FormattedMessage defaultMessage="Close" description="Queue settings: close button" />
        </Button>
      }
    >
      <Tabs.Root componentId={`${CID}.tabs`} defaultValue="questions">
        <Tabs.List>
          <Tabs.Trigger value="questions">
            <FormattedMessage defaultMessage="Questions" description="Queue settings: questions tab" />
          </Tabs.Trigger>
          <Tabs.Trigger value="traces">
            <FormattedMessage defaultMessage="Traces" description="Queue settings: traces tab" />
          </Tabs.Trigger>
        </Tabs.List>

        <Tabs.Content value="questions">
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm, paddingTop: theme.spacing.sm }}>
            {isUser ? (
              <>
                <Typography.Hint>
                  <FormattedMessage
                    defaultMessage="Personal queues use all of the experiment's questions; they can't be edited here."
                    description="Queue settings: user-queue questions note"
                  />
                </Typography.Hint>
                {attachedQuestions.map((schema) => (
                  <Typography.Text key={schema.schema_id}>{schema.name}</Typography.Text>
                ))}
              </>
            ) : canEditQuestions ? (
              <>
                <Typography.Hint>
                  <FormattedMessage
                    defaultMessage="Choose which questions reviewers answer for traces in this queue."
                    description="Queue settings: editable questions hint"
                  />
                </Typography.Hint>
                {labelSchemas.length === 0 ? (
                  <Empty
                    description={
                      <FormattedMessage
                        defaultMessage="No questions defined for this experiment yet."
                        description="Queue settings: no schemas to attach"
                      />
                    }
                  />
                ) : (
                  labelSchemas.map((schema) => (
                    <Checkbox
                      key={schema.schema_id}
                      componentId={`${CID}.schema-checkbox`}
                      isChecked={selectedIds.has(schema.schema_id)}
                      onChange={(checked) => toggle(schema.schema_id, checked)}
                    >
                      {schema.name}
                    </Checkbox>
                  ))
                )}
                {updateError && (
                  <Alert
                    componentId={`${CID}.questions-error`}
                    type="error"
                    closable={false}
                    message={intl.formatMessage({
                      defaultMessage: 'Failed to update the queue questions.',
                      description: 'Queue settings: questions update error',
                    })}
                    description={updateError.message}
                  />
                )}
                <div>
                  <Button
                    componentId={`${CID}.save-questions`}
                    type="primary"
                    disabled={selectedIds.size === 0 || isUpdatingQueue}
                    loading={isUpdatingQueue}
                    onClick={handleSaveQuestions}
                  >
                    <FormattedMessage defaultMessage="Save questions" description="Queue settings: save questions" />
                  </Button>
                </div>
              </>
            ) : (
              <>
                <Typography.Hint>
                  <FormattedMessage
                    defaultMessage="Questions are locked because this queue has traces. Remove its traces (Traces tab) to edit them."
                    description="Queue settings: questions-locked note"
                  />
                </Typography.Hint>
                {attachedQuestions.map((schema) => (
                  <Typography.Text key={schema.schema_id}>{schema.name}</Typography.Text>
                ))}
              </>
            )}
          </div>
        </Tabs.Content>

        <Tabs.Content value="traces">
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm, paddingTop: theme.spacing.sm }}>
            {removeError && (
              <Alert
                componentId={`${CID}.remove-error`}
                type="error"
                closable={false}
                message={intl.formatMessage({
                  defaultMessage: 'Failed to remove the trace.',
                  description: 'Queue settings: remove trace error',
                })}
                description={removeError.message}
              />
            )}
            {tracesLoading ? (
              <TableSkeleton lines={4} />
            ) : items.length === 0 ? (
              <Empty
                description={
                  <FormattedMessage
                    defaultMessage="No traces in this queue."
                    description="Queue settings: empty traces"
                  />
                }
              />
            ) : (
              items.map((item) => (
                <div
                  key={item.target_id}
                  css={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: theme.spacing.sm,
                    padding: theme.spacing.sm,
                    border: `1px solid ${theme.colors.border}`,
                    borderRadius: theme.borders.borderRadiusMd,
                  }}
                >
                  <Typography.Text bold css={{ flex: 1 }} ellipsis>
                    {item.target_id}
                  </Typography.Text>
                  <StatusTag status={item.status} />
                  <Button
                    componentId={`${CID}.remove-trace`}
                    size="small"
                    icon={<TrashIcon />}
                    disabled={isRemovingTraces}
                    aria-label={intl.formatMessage({
                      defaultMessage: 'Remove trace from queue',
                      description: 'Queue settings: remove-trace button aria label',
                    })}
                    onClick={() =>
                      removeTracesFromReviewQueue({ queue_id: queue.queue_id, target_ids: [item.target_id] })
                    }
                  />
                </div>
              ))
            )}
          </div>
        </Tabs.Content>
      </Tabs.Root>
    </Modal>
  );
};
