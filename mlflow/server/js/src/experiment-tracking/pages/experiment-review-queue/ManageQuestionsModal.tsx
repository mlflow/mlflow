import { useState } from 'react';

import {
  Empty,
  Modal,
  PlusIcon,
  TableSkeleton,
  Tag,
  TrashIcon,
  Button,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

import {
  LabelSchemaFormDrawer,
  useDeleteLabelSchemaMutation,
  useListLabelSchemasQuery,
} from '../../components/label-schemas';
import type { LabelSchema } from '../../components/label-schemas';

const CID = 'mlflow.experiment-review-queue.manage-questions';

/**
 * Question manager opened from the Review tab's gear icon. The questions are
 * the experiment's label schemas, so this reuses the label-schema hooks to
 * list, delete, and — via the authoring drawer — create and edit them.
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
  // The authoring drawer: open with `null` to create, or a schema to edit.
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [editingSchema, setEditingSchema] = useState<LabelSchema | null>(null);

  const openCreate = () => {
    setEditingSchema(null);
    setDrawerOpen(true);
  };
  const openEdit = (schema: LabelSchema) => {
    setEditingSchema(schema);
    setDrawerOpen(true);
  };

  return (
    <>
      <Modal
        componentId={`${CID}.modal`}
        visible
        onCancel={onClose}
        title={<FormattedMessage defaultMessage="Review questions" description="Manage review questions modal title" />}
        footer={
          <div css={{ display: 'flex', justifyContent: 'space-between' }}>
            <Button componentId={`${CID}.add`} icon={<PlusIcon />} onClick={openCreate}>
              <FormattedMessage defaultMessage="Add question" description="Manage review questions: add button" />
            </Button>
            <Button componentId={`${CID}.close`} onClick={onClose}>
              <FormattedMessage defaultMessage="Close" description="Manage review questions: close button" />
            </Button>
          </div>
        }
      >
        <Typography.Hint css={{ marginBottom: theme.spacing.md }}>
          <FormattedMessage
            defaultMessage="Questions are the experiment's label schemas. Add a new one or click a question to edit it."
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
                role="button"
                tabIndex={0}
                onClick={() => openEdit(schema)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    openEdit(schema);
                  }
                }}
                css={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: theme.spacing.sm,
                  padding: theme.spacing.sm,
                  border: `1px solid ${theme.colors.border}`,
                  borderRadius: theme.borders.borderRadiusMd,
                  cursor: 'pointer',
                  '&:hover': { backgroundColor: theme.colors.actionDefaultBackgroundHover },
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
                  onClick={(e) => {
                    e.stopPropagation();
                    setPendingDelete(schema);
                  }}
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

      <LabelSchemaFormDrawer
        experimentId={experimentId}
        editingSchema={editingSchema}
        visible={drawerOpen}
        onClose={() => setDrawerOpen(false)}
      />
    </>
  );
};
