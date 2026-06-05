import { useState } from 'react';

import {
  Empty,
  Modal,
  TableSkeleton,
  Tag,
  TrashIcon,
  Button,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

import { useDeleteLabelSchemaMutation, useListLabelSchemasQuery } from '../../components/label-schemas';
import type { LabelSchema } from '../../components/label-schemas';

const CID = 'mlflow.experiment-review-queue.manage-questions';

/**
 * Lightweight question manager opened from the Review tab's gear icon. The
 * questions are the experiment's label schemas, so this reuses the
 * label-schema hooks: it lists the current questions and supports deletion.
 * Full authoring (input-type config) stays in the Label Schemas tab until
 * that surface is folded in.
 */
export const ManageQuestionsModal = ({ experimentId, onClose }: { experimentId: string; onClose: () => void }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { labelSchemas, isLoading } = useListLabelSchemasQuery({ experimentId });
  const { deleteLabelSchema, isDeleting } = useDeleteLabelSchemaMutation();

  // The schema pending deletion confirmation; deleting a label schema is
  // experiment-wide (it removes the question from every queue), so it goes
  // through an explicit confirm rather than firing on the first click.
  const [pendingDelete, setPendingDelete] = useState<LabelSchema | null>(null);

  return (
    <>
      <Modal
        componentId={`${CID}.modal`}
        visible
        onCancel={onClose}
        title={<FormattedMessage defaultMessage="Review questions" description="Manage review questions modal title" />}
        footer={
          <Button componentId={`${CID}.close`} onClick={onClose}>
            <FormattedMessage defaultMessage="Close" description="Manage review questions: close button" />
          </Button>
        }
      >
        <Typography.Hint css={{ marginBottom: theme.spacing.md }}>
          <FormattedMessage
            defaultMessage="Questions are the experiment's label schemas. Create new ones from the Label Schemas tab."
            description="Manage review questions: explanatory hint"
          />
        </Typography.Hint>

        {isLoading ? (
          <TableSkeleton lines={3} />
        ) : labelSchemas.length === 0 ? (
          <Empty
            description={
              <FormattedMessage
                defaultMessage="No questions defined for this experiment yet."
                description="Manage review questions: empty state"
              />
            }
          />
        ) : (
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
            {labelSchemas.map((schema) => (
              <div
                key={schema.schema_id}
                css={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: theme.spacing.sm,
                  padding: theme.spacing.sm,
                  border: `1px solid ${theme.colors.border}`,
                  borderRadius: theme.borders.borderRadiusMd,
                }}
              >
                <Typography.Text bold css={{ flex: 1 }}>
                  {schema.name}
                </Typography.Text>
                <Tag componentId={`${CID}.type-tag`} color={schema.type === 'EXPECTATION' ? 'turquoise' : 'lime'}>
                  {schema.type === 'EXPECTATION' ? (
                    <FormattedMessage defaultMessage="Expectation" description="Label schema type: expectation" />
                  ) : (
                    <FormattedMessage defaultMessage="Feedback" description="Label schema type: feedback" />
                  )}
                </Tag>
                <Button
                  componentId={`${CID}.delete`}
                  icon={<TrashIcon />}
                  size="small"
                  disabled={isDeleting}
                  aria-label={intl.formatMessage({
                    defaultMessage: 'Delete question',
                    description: 'Manage review questions: delete button aria label',
                  })}
                  onClick={() => setPendingDelete(schema)}
                />
              </div>
            ))}
          </div>
        )}
      </Modal>

      {pendingDelete && (
        <Modal
          componentId={`${CID}.delete-confirm`}
          visible
          title={
            <FormattedMessage
              defaultMessage="Delete question?"
              description="Manage review questions: delete confirmation title"
            />
          }
          okText={<FormattedMessage defaultMessage="Delete" description="Manage review questions: confirm delete" />}
          okButtonProps={{ danger: true, loading: isDeleting }}
          cancelText={<FormattedMessage defaultMessage="Cancel" description="Manage review questions: cancel delete" />}
          onCancel={() => setPendingDelete(null)}
          onOk={() =>
            deleteLabelSchema({ schema_id: pendingDelete.schema_id }, { onSettled: () => setPendingDelete(null) })
          }
        >
          <FormattedMessage
            defaultMessage='Deleting "{name}" removes it from every review queue in this experiment. This cannot be undone.'
            description="Manage review questions: delete confirmation body"
            values={{ name: pendingDelete.name }}
          />
        </Modal>
      )}
    </>
  );
};
