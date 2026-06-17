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
import Utils from '../../../common/utils/Utils';
import { Link } from '../../../common/utils/RoutingUtils';
import { useListReviewQueuesQuery } from './hooks/useListReviewQueuesQuery';
import { getReviewQueuePageRoute } from './hooks/useReviewQueueSearchParams';

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
  const { deleteLabelSchemaAsync, isDeleting } = useDeleteLabelSchemaMutation();
  // Shares the cache with the Review tab's queue list (same query key), so this
  // is a cache hit; used to show which custom queues a deletion would affect.
  // First page only (no maxResults), matching the Review tab; an experiment with
  // more queues than the default page size would undercount the blast radius.
  const { reviewQueues } = useListReviewQueuesQuery({ experimentId });

  // The schema pending deletion confirmation; deleting a label schema is
  // experiment-wide (it removes the question from every queue), so it goes
  // through an explicit confirm rather than firing on the first click.
  const [pendingDelete, setPendingDelete] = useState<LabelSchema | null>(null);
  // Custom queues that explicitly include the pending question (linked below).
  // User queues inherit every question, so all of them are affected by a delete —
  // we surface their count (not a list) so the manager sees the full blast radius.
  const affectedQueues = pendingDelete
    ? reviewQueues.filter((q) => q.queue_type === 'CUSTOM' && (q.schema_ids ?? []).includes(pendingDelete.schema_id))
    : [];
  const userQueueCount = pendingDelete ? reviewQueues.filter((q) => q.queue_type === 'USER').length : 0;
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
          <div css={{ display: 'flex', justifyContent: 'flex-end' }}>
            {/* No explicit Close button — the modal's top-right "X" (onCancel) closes it. */}
            <Button componentId={`${CID}.add`} icon={<PlusIcon />} onClick={openCreate}>
              <FormattedMessage defaultMessage="Add question" description="Manage review questions: add button" />
            </Button>
          </div>
        }
      >
        <Typography.Hint css={{ marginBottom: theme.spacing.md }}>
          <FormattedMessage
            defaultMessage="Add a new question, or click a question to edit it."
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
            {labelSchemas.map((schema) => {
              // The protected default question can't be edited or deleted, so its
              // row is static (no click-to-edit) and its delete button is disabled.
              const isDefault = Boolean(schema.is_default);
              return (
                <div
                  key={schema.schema_id}
                  role={isDefault ? undefined : 'button'}
                  tabIndex={isDefault ? undefined : 0}
                  onClick={isDefault ? undefined : () => openEdit(schema)}
                  onKeyDown={
                    isDefault
                      ? undefined
                      : (e) => {
                          if (e.key === 'Enter' || e.key === ' ') {
                            e.preventDefault();
                            openEdit(schema);
                          }
                        }
                  }
                  css={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: theme.spacing.sm,
                    padding: theme.spacing.sm,
                    border: `1px solid ${theme.colors.border}`,
                    borderRadius: theme.borders.borderRadiusMd,
                    cursor: isDefault ? 'default' : 'pointer',
                    ...(isDefault ? {} : { '&:hover': { backgroundColor: theme.colors.actionDefaultBackgroundHover } }),
                  }}
                >
                  <Typography.Text bold css={{ flex: 1 }}>
                    {schema.name}
                  </Typography.Text>
                  {isDefault && (
                    <Tag componentId={`${CID}.default-tag`} color="charcoal">
                      <FormattedMessage
                        defaultMessage="Default"
                        description="Manage review questions: tag marking the protected default question"
                      />
                    </Tag>
                  )}
                  <Tag componentId={`${CID}.type-tag`} color={schema.type === 'EXPECTATION' ? 'turquoise' : 'lime'}>
                    {schema.type === 'EXPECTATION' ? (
                      <FormattedMessage defaultMessage="Expectation" description="Review question type: expectation" />
                    ) : (
                      <FormattedMessage defaultMessage="Feedback" description="Review question type: feedback" />
                    )}
                  </Tag>
                  <Button
                    componentId={`${CID}.delete`}
                    icon={<TrashIcon />}
                    size="small"
                    disabled={isDeleting || isDefault}
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
              );
            })}
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
          onOk={async () => {
            try {
              await deleteLabelSchemaAsync({ schema_id: pendingDelete.schema_id });
            } catch (e) {
              // Without this the delete failure was swallowed: the confirm closed and
              // the question silently stayed. Surface it as a toast (matching the other
              // review-queue modals); the question remains in the list to retry.
              Utils.displayGlobalErrorNotification(
                intl.formatMessage(
                  {
                    defaultMessage: 'Failed to delete question: {error}',
                    description: 'Manage review questions: error toast shown when deleting a question fails',
                  },
                  { error: e instanceof Error ? e.message : String(e) },
                ),
              );
            } finally {
              setPendingDelete(null);
            }
          }}
        >
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
            <FormattedMessage
              defaultMessage='Deleting "{name}" removes it from every review queue in this experiment. This cannot be undone.'
              description="Manage review questions: delete confirmation body"
              values={{ name: pendingDelete.name }}
            />
            {affectedQueues.length > 0 ? (
              <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
                <Typography.Text bold>
                  {userQueueCount > 0 ? (
                    // Each count is its own single-plural message, composed into a
                    // plural-free sentence (formatjs disallows two plurals per message).
                    <FormattedMessage
                      defaultMessage="Used by {custom} and {user}:"
                      description="Manage review questions: heading above the affected-queue list when the experiment also has user queues (which inherit every question)"
                      values={{
                        custom: intl.formatMessage(
                          {
                            defaultMessage: '{count, plural, one {# custom queue} other {# custom queues}}',
                            description:
                              'Manage review questions: count of custom queues using the question being deleted',
                          },
                          { count: affectedQueues.length },
                        ),
                        user: intl.formatMessage(
                          {
                            defaultMessage: '{count, plural, one {# user queue} other {# user queues}}',
                            description:
                              'Manage review questions: count of user queues affected by deleting the question',
                          },
                          { count: userQueueCount },
                        ),
                      }}
                    />
                  ) : (
                    <FormattedMessage
                      defaultMessage="Used by {count, plural, one {# queue} other {# queues}}:"
                      description="Manage review questions: heading above the list of queues that use the question being deleted"
                      values={{ count: affectedQueues.length }}
                    />
                  )}
                </Typography.Text>
                {/* Only custom queues are listed (user queues inherit every question);
                    scrollable so a question used by many queues doesn't blow up the modal. */}
                <div
                  css={{
                    maxHeight: 160,
                    overflowY: 'auto',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: theme.spacing.xs,
                    border: `1px solid ${theme.colors.border}`,
                    borderRadius: theme.borders.borderRadiusMd,
                    padding: theme.spacing.sm,
                  }}
                >
                  {affectedQueues.map((q) => (
                    // Opens in a new tab so the manager can inspect a queue without
                    // losing this confirmation. Only managers reach this modal.
                    <Link
                      key={q.queue_id}
                      componentId={`${CID}.delete-confirm.queue-link`}
                      to={getReviewQueuePageRoute(experimentId, q.queue_id)}
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      {q.name}
                    </Link>
                  ))}
                </div>
              </div>
            ) : (
              // No custom queue uses the question, but every user queue inherits it,
              // so surface that count as a plain sentence (no list follows).
              userQueueCount > 0 && (
                <Typography.Text bold>
                  <FormattedMessage
                    defaultMessage="Inherited by {count, plural, one {# user queue} other {# user queues}}."
                    description="Manage review questions: note shown when no custom queue uses the question but user queues (which inherit every question) do"
                    values={{ count: userQueueCount }}
                  />
                </Typography.Text>
              )
            )}
          </div>
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
