import { useState } from 'react';

import {
  Alert,
  Checkbox,
  Empty,
  Modal,
  TableSkeleton,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

import { useListLabelSchemasQuery } from '../../components/label-schemas';
import { useUpdateReviewQueueMutation } from './hooks/useUpdateReviewQueueMutation';
import type { ReviewQueue } from './types';

const CID = 'mlflow.experiment-review-queue.edit-questions';

/**
 * Edit which questions (label schemas) a CUSTOM queue asks. Only reachable
 * while the queue has no traces — the server freezes questions once traces are
 * assigned, and the caller (`ReviewQueueSection`) disables the entry point
 * accordingly.
 */
export const EditQueueQuestionsModal = ({ queue, onClose }: { queue: ReviewQueue; onClose: () => void }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { labelSchemas, isLoading } = useListLabelSchemasQuery({ experimentId: queue.experiment_id });
  const { updateReviewQueueAsync, isUpdatingQueue, error } = useUpdateReviewQueueMutation();

  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set(queue.schema_ids ?? []));

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

  const canSubmit = selectedIds.size > 0 && !isUpdatingQueue;

  const handleSave = async () => {
    if (!canSubmit) {
      return;
    }
    await updateReviewQueueAsync({ queue_id: queue.queue_id, schema_ids: [...selectedIds] });
    onClose();
  };

  return (
    <Modal
      componentId={`${CID}.modal`}
      visible
      title={<FormattedMessage defaultMessage="Edit questions" description="Edit review queue questions modal title" />}
      okText={<FormattedMessage defaultMessage="Save" description="Edit review queue questions: confirm button" />}
      okButtonProps={{ disabled: !canSubmit, loading: isUpdatingQueue }}
      cancelText={<FormattedMessage defaultMessage="Cancel" description="Edit review queue questions: cancel button" />}
      onOk={handleSave}
      onCancel={onClose}
    >
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
        <Typography.Hint>
          <FormattedMessage
            defaultMessage="Choose which questions reviewers answer for traces in “{name}”."
            description="Edit review queue questions: hint"
            values={{ name: queue.name }}
          />
        </Typography.Hint>

        {isLoading ? (
          <TableSkeleton lines={3} />
        ) : labelSchemas.length === 0 ? (
          <Empty
            description={
              <FormattedMessage
                defaultMessage="No questions defined for this experiment yet. Create label schemas first."
                description="Edit review queue questions: empty state"
              />
            }
          />
        ) : (
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
            {labelSchemas.map((schema) => (
              <Checkbox
                key={schema.schema_id}
                componentId={`${CID}.schema-checkbox`}
                isChecked={selectedIds.has(schema.schema_id)}
                onChange={(checked) => toggle(schema.schema_id, checked)}
              >
                {schema.name}
              </Checkbox>
            ))}
          </div>
        )}

        {selectedIds.size === 0 && (
          <Typography.Hint>
            <FormattedMessage
              defaultMessage="Select at least one question so reviewers have something to answer."
              description="Edit review queue questions: no-questions-selected warning"
            />
          </Typography.Hint>
        )}

        {error && (
          <Alert
            componentId={`${CID}.error`}
            type="error"
            closable={false}
            message={intl.formatMessage({
              defaultMessage: 'Failed to update the queue questions.',
              description: 'Edit review queue questions: error alert title',
            })}
            description={error.message}
          />
        )}
      </div>
    </Modal>
  );
};
