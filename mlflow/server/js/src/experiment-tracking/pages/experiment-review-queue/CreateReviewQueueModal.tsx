import { useMemo, useState } from 'react';

import {
  Alert,
  Checkbox,
  Empty,
  FormUI,
  Input,
  Modal,
  TableSkeleton,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

import { useListLabelSchemasQuery } from '../../components/label-schemas';
import { useCreateReviewQueueMutation } from './hooks/useCreateReviewQueueMutation';
import { useReviewer } from './hooks/useReviewer';
import type { ReviewQueue } from './types';

const CID = 'mlflow.experiment-review-queue.create-queue';

/**
 * B-lite create form for a CUSTOM review queue: a name plus the subset of the
 * experiment's label schemas (questions) the queue should ask. All schemas
 * start checked. Personal USER queues aren't created here; they're resolved
 * on demand via get-or-create. Users assignment is deferred to a later
 * iteration.
 */
export const CreateReviewQueueModal = ({
  experimentId,
  onClose,
  onCreated,
}: {
  experimentId: string;
  onClose: () => void;
  onCreated?: (queue: ReviewQueue) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const createdBy = useReviewer();
  const { labelSchemas, isLoading } = useListLabelSchemasQuery({ experimentId });
  const { createReviewQueueAsync, isCreatingQueue, error } = useCreateReviewQueueMutation();

  const [name, setName] = useState('');
  // `undefined` means "not yet initialized"; once schemas load we default to
  // every schema checked.
  const [selectedIds, setSelectedIds] = useState<Set<string> | undefined>(undefined);

  const checkedIds = useMemo(
    () => selectedIds ?? new Set(labelSchemas.map((s) => s.schema_id)),
    [selectedIds, labelSchemas],
  );

  const toggle = (schemaId: string, checked: boolean) => {
    const next = new Set(checkedIds);
    if (checked) {
      next.add(schemaId);
    } else {
      next.delete(schemaId);
    }
    setSelectedIds(next);
  };

  const trimmedName = name.trim();
  const canSubmit = trimmedName.length > 0 && checkedIds.size > 0 && !isCreatingQueue;

  const handleCreate = async () => {
    if (!canSubmit) {
      return;
    }
    const { review_queue } = await createReviewQueueAsync({
      experiment_id: experimentId,
      name: trimmedName,
      queue_type: 'CUSTOM',
      created_by: createdBy,
      schema_ids: [...checkedIds],
    });
    onCreated?.(review_queue);
    onClose();
  };

  return (
    <Modal
      componentId={`${CID}.modal`}
      visible
      title={<FormattedMessage defaultMessage="New review queue" description="Create review queue modal title" />}
      okText={<FormattedMessage defaultMessage="Create" description="Create review queue: confirm button" />}
      okButtonProps={{ disabled: !canSubmit, loading: isCreatingQueue }}
      cancelText={<FormattedMessage defaultMessage="Cancel" description="Create review queue: cancel button" />}
      onOk={handleCreate}
      onCancel={onClose}
    >
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
        <div>
          <FormUI.Label htmlFor={`${CID}.name-input`}>
            <FormattedMessage defaultMessage="Name" description="Create review queue: name field label" />
          </FormUI.Label>
          <Input
            componentId={`${CID}.name`}
            id={`${CID}.name-input`}
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder={intl.formatMessage({
              defaultMessage: 'e.g. Hallucination review',
              description: 'Create review queue: name field placeholder',
            })}
          />
        </div>

        <div>
          <FormUI.Label>
            <FormattedMessage defaultMessage="Questions" description="Create review queue: questions field label" />
          </FormUI.Label>
          <FormUI.Hint css={{ marginBottom: theme.spacing.sm }}>
            <FormattedMessage
              defaultMessage="Choose which questions reviewers answer for traces in this queue."
              description="Create review queue: questions field hint"
            />
          </FormUI.Hint>
          {isLoading ? (
            <TableSkeleton lines={3} />
          ) : labelSchemas.length === 0 ? (
            <Empty
              description={
                <FormattedMessage
                  defaultMessage="No questions defined for this experiment yet. Create label schemas first."
                  description="Create review queue: empty questions state"
                />
              }
            />
          ) : (
            <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
              {labelSchemas.map((schema) => (
                <Checkbox
                  key={schema.schema_id}
                  componentId={`${CID}.schema-checkbox`}
                  isChecked={checkedIds.has(schema.schema_id)}
                  onChange={(checked) => toggle(schema.schema_id, checked)}
                >
                  {schema.name}
                </Checkbox>
              ))}
            </div>
          )}
        </div>

        {trimmedName.length > 0 && checkedIds.size === 0 && (
          <Typography.Hint>
            <FormattedMessage
              defaultMessage="Select at least one question so reviewers have something to answer."
              description="Create review queue: no-questions-selected warning"
            />
          </Typography.Hint>
        )}

        {error && (
          <Alert
            componentId={`${CID}.error`}
            type="error"
            closable={false}
            message={intl.formatMessage({
              defaultMessage: 'Failed to create the review queue.',
              description: 'Create review queue: error alert title',
            })}
            description={error.message}
          />
        )}
      </div>
    </Modal>
  );
};
