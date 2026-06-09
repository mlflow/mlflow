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
  LabelSchemaFormModal,
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
  // The authoring modal: open with `null` to create, or a schema to edit.
  const [formOpen, setFormOpen] = useState(false);
  const [editingSchema, setEditingSchema] = useState<LabelSchema | null>(null);

  const openCreate = () => {
    setEditingSchema(null);
    setFormOpen(true);
  };
  const openEdit = (schema: LabelSchema) => {
    setEditingSchema(schema);
    setFormOpen(true);
  };

  return (
    <>
      <Modal
        componentId={`${CID}.modal`}
        // Hide the question list while the add/edit form is open rather than
        // stacking modals; it reappears when the form modal closes.
        visible={!formOpen}
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
          <div
            css={{
              display: 'flex',
              flexDirection: 'column',
              gap: theme.spacing.sm,
              // Cap the list so the modal stops growing with every question; the
              // list scrolls past this height.
              maxHeight: 360,
              overflowY: 'auto',
              // CSS scroll-shadows (Lea Verou technique): a soft shadow shows at
              // the top/bottom edge only when there's more to scroll that way, so
              // the scrollable-ness is noticeable. The cover gradients are the
              // modal's own background; the shadow radials are scroll-attached.
              background: `
                linear-gradient(${theme.colors.backgroundPrimary} 30%, rgba(0, 0, 0, 0)) 0 0,
                linear-gradient(rgba(0, 0, 0, 0), ${theme.colors.backgroundPrimary} 70%) 0 100%,
                radial-gradient(farthest-side at 50% 0, rgba(0, 0, 0, 0.12), rgba(0, 0, 0, 0)) 0 0,
                radial-gradient(farthest-side at 50% 100%, rgba(0, 0, 0, 0.12), rgba(0, 0, 0, 0)) 0 100%
              `,
              backgroundRepeat: 'no-repeat',
              backgroundSize: '100% 28px, 100% 28px, 100% 10px, 100% 10px',
              backgroundAttachment: 'local, local, scroll, scroll',
            }}
          >
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

      <LabelSchemaFormModal
        experimentId={experimentId}
        editingSchema={editingSchema}
        visible={formOpen}
        onClose={() => setFormOpen(false)}
      />
    </>
  );
};
