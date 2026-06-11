import { useMemo, useState } from 'react';

import {
  Alert,
  ApplyDesignSystemContextOverrides,
  Button,
  ChevronDownIcon,
  ChevronRightIcon,
  Empty,
  FormUI,
  Input,
  Modal,
  PlusIcon,
  TableSkeleton,
  TrashIcon,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

import {
  LabelSchemaInputRenderer,
  LabelSchemaFormModal,
  useListLabelSchemasQuery,
} from '../../components/label-schemas';
import { QuestionChecklistCombobox } from './QuestionChecklistCombobox';
import { useIsAuthAvailable } from '../../../account/hooks';
import { useCreateReviewQueueMutation } from './hooks/useCreateReviewQueueMutation';
import { useReviewer } from './hooks/useReviewer';
import type { ReviewQueue } from './types';

const CID = 'mlflow.experiment-review-queue.create-queue';

/**
 * Create form for a CUSTOM review queue: a name, the subset of the experiment's
 * label schemas (questions) the queue asks — picked from a multi-select
 * dropdown with a live preview of each chosen question — and, on an
 * authenticated server, the reviewers to assign. (A no-auth server has a single
 * `default` user, so reviewer assignment is hidden.) Personal USER queues
 * aren't created here; they're resolved on demand via get-or-create.
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
  const authAvailable = useIsAuthAvailable();
  const { labelSchemas, isLoading } = useListLabelSchemasQuery({ experimentId });
  const { createReviewQueueAsync, isCreatingQueue, error } = useCreateReviewQueueMutation();

  const [name, setName] = useState('');
  // No questions are selected by default; the creator chooses them.
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  // Reviewer identifiers (usernames / emails) to assign; auth servers only.
  const [reviewers, setReviewers] = useState<string[]>(['']);
  // Whether the inline "New question" form is open; hides this modal while open.
  const [createQuestionOpen, setCreateQuestionOpen] = useState(false);
  // Which question previews are expanded (collapsed by default).
  const [expandedPreview, setExpandedPreview] = useState<Set<string>>(new Set());
  const togglePreview = (schemaId: string) =>
    setExpandedPreview((prev) => {
      const next = new Set(prev);
      if (next.has(schemaId)) {
        next.delete(schemaId);
      } else {
        next.add(schemaId);
      }
      return next;
    });

  const checkedIds = selectedIds;
  const selectedSchemas = useMemo(
    () => labelSchemas.filter((s) => checkedIds.has(s.schema_id)),
    [labelSchemas, checkedIds],
  );
  // The dropdown trigger shows a count, not the list of names. DialogCombobox
  // joins `value` entries, so collapse to a single summary string; item checked
  // state comes from each item's explicit `checked` prop, not from `value`.
  const questionsTriggerValue = useMemo(
    () =>
      selectedSchemas.length > 0
        ? [
            intl.formatMessage(
              {
                defaultMessage: '{count, plural, one {# question} other {# questions}} selected',
                description: 'Create review queue: questions dropdown selected-count summary',
              },
              { count: selectedSchemas.length },
            ),
          ]
        : [],
    [selectedSchemas, intl],
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

  const setReviewerAt = (index: number, value: string) =>
    setReviewers((prev) => prev.map((r, i) => (i === index ? value : r)));
  const addReviewer = () => setReviewers((prev) => [...prev, '']);
  const removeReviewerAt = (index: number) => setReviewers((prev) => prev.filter((_, i) => i !== index));

  const trimmedName = name.trim();
  const canSubmit = trimmedName.length > 0 && checkedIds.size > 0 && !isCreatingQueue;

  // Keep the questions dropdown above this modal — otherwise it opens behind the
  // modal when launched over a trace-detail drawer. `zIndexBase` is the drawer's
  // (doubled) base when nested, matching the DialogCombobox-in-modal convention
  // used across the app (e.g. RoleUsersSection, DirectPermissionForm).
  const dropdownZIndex = theme.options.zIndexBase + 100;

  const handleCreate = async () => {
    if (!canSubmit) {
      return;
    }
    // The creator is always a member so the queue shows under "Created by me";
    // assigned reviewers (auth only) get it under "Feedback requested".
    const assigned = authAvailable ? reviewers.map((r) => r.trim()).filter(Boolean) : [];
    const users = Array.from(new Set([createdBy, ...assigned]));
    const { review_queue } = await createReviewQueueAsync({
      experiment_id: experimentId,
      name: trimmedName,
      queue_type: 'CUSTOM',
      created_by: createdBy,
      users,
      schema_ids: [...checkedIds],
    });
    onCreated?.(review_queue);
    onClose();
  };

  return (
    <>
    <Modal
      componentId={`${CID}.modal`}
      visible={!createQuestionOpen}
      destroyOnClose
        // Relative to the (drawer-doubled) base so it clears a trace-detail drawer
        // when launched from the flag-for-review picker, like ExportTracesToDatasetModal.
        zIndex={theme.options.zIndexBase + 10}
        title={<FormattedMessage defaultMessage="New review queue" description="Create review queue modal title" />}
        okText={<FormattedMessage defaultMessage="Create" description="Create review queue: confirm button" />}
        okButtonProps={{ disabled: !canSubmit, loading: isCreatingQueue }}
        cancelText={null}
        onOk={handleCreate}
        onCancel={onClose}
      >
        {/* Portal the questions dropdown into document.body, not the trace-detail
          drawer that can host this modal — the drawer's stacking context would
          otherwise trap it below the modal regardless of z-index. */}
        <ApplyDesignSystemContextOverrides getPopupContainer={() => document.body}>
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
                <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
                  <QuestionChecklistCombobox
                    componentId={`${CID}.questions`}
                    schemas={labelSchemas}
                    checkedIds={checkedIds}
                    onToggle={(schemaId) => toggle(schemaId, !checkedIds.has(schemaId))}
                    onCreateQuestion={() => setCreateQuestionOpen(true)}
                    triggerValue={questionsTriggerValue}
                    dropdownZIndex={dropdownZIndex}
                  />

                  {/* Live preview of the selected questions, as a reviewer will see them.
                   Always mounted so the grid-row transition can animate height. */}
                  <div
                    css={{
                      display: 'grid',
                      gridTemplateRows: selectedSchemas.length > 0 ? '1fr' : '0fr',
                      transition: 'grid-template-rows 200ms ease-out',
                    }}
                  >
                    <div css={{ overflow: 'hidden' }}>
                      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
                        <FormUI.Label>
                          <FormattedMessage
                            defaultMessage="Question preview"
                            description="Create review queue: question preview section label"
                          />
                        </FormUI.Label>
                        <FormUI.Hint css={{ marginBottom: theme.spacing.xs }}>
                          <FormattedMessage
                            defaultMessage="Preview how the questions will appear for the human reviewer."
                            description="Create review queue: question preview section hint"
                          />
                        </FormUI.Hint>
                        <div
                          css={{
                            display: 'flex',
                            flexDirection: 'column',
                            gap: theme.spacing.sm,
                            // Fixed height (~4 collapsed question rows), reserved up front
                            // so the modal never resizes: more questions or an expanded
                            // preview scroll within this section instead of growing it.
                            height: 200,
                            flexShrink: 0,
                            overflowY: 'auto',
                          }}
                        >
                          {selectedSchemas.map((schema) => {
                            const open = expandedPreview.has(schema.schema_id);
                            return (
                              <div
                                key={schema.schema_id}
                                css={{
                                  display: 'flex',
                                  flexDirection: 'column',
                                  gap: theme.spacing.xs,
                                  padding: theme.spacing.sm,
                                  border: `1px solid ${theme.colors.border}`,
                                  borderRadius: theme.borders.borderRadiusMd,
                                }}
                              >
                                <div
                                  role="button"
                                  tabIndex={0}
                                  onClick={() => togglePreview(schema.schema_id)}
                                  onKeyDown={(e) => {
                                    if (e.key === 'Enter' || e.key === ' ') {
                                      e.preventDefault();
                                      togglePreview(schema.schema_id);
                                    }
                                  }}
                                  css={{
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: theme.spacing.xs,
                                    cursor: 'pointer',
                                  }}
                                >
                                  {open ? <ChevronDownIcon /> : <ChevronRightIcon />}
                                  <Typography.Text bold>{schema.name}</Typography.Text>
                                </div>
                                {open && (
                                  <>
                                    {schema.instruction && <Typography.Hint>{schema.instruction}</Typography.Hint>}
                                    <LabelSchemaInputRenderer
                                      input={schema.input}
                                      value={null}
                                      onChange={() => {}}
                                      disabled
                                      componentId={`${CID}.preview`}
                                      label={schema.name}
                                      instruction={schema.instruction}
                                    />
                                  </>
                                )}
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {authAvailable && (
              <div>
                <FormUI.Label>
                  <FormattedMessage
                    defaultMessage="Reviewers"
                    description="Create review queue: reviewers field label"
                  />
                </FormUI.Label>
                <FormUI.Hint css={{ marginBottom: theme.spacing.sm }}>
                  <FormattedMessage
                    defaultMessage="Assign reviewers who should answer this queue's questions. They'll find it under “Feedback requested”."
                    description="Create review queue: reviewers field hint"
                  />
                </FormUI.Hint>
                <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
                  {reviewers.map((reviewer, index) => (
                    // eslint-disable-next-line react/no-array-index-key
                    <div key={index} css={{ display: 'flex', gap: theme.spacing.sm }}>
                      <Input
                        componentId={`${CID}.reviewer`}
                        css={{ flex: 1 }}
                        value={reviewer}
                        onChange={(e) => setReviewerAt(index, e.target.value)}
                        placeholder={intl.formatMessage({
                          defaultMessage: 'username or email',
                          description: 'Create review queue: reviewer input placeholder',
                        })}
                      />
                      <Button
                        componentId={`${CID}.remove-reviewer`}
                        icon={<TrashIcon />}
                        aria-label={intl.formatMessage({
                          defaultMessage: 'Remove reviewer',
                          description: 'Create review queue: remove-reviewer button',
                        })}
                        onClick={() => removeReviewerAt(index)}
                      />
                    </div>
                  ))}
                </div>
                <Button componentId={`${CID}.add-reviewer`} icon={<PlusIcon />} onClick={addReviewer}>
                  <FormattedMessage
                    defaultMessage="Add reviewer"
                    description="Create review queue: add-reviewer button"
                  />
                </Button>
              </div>
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
        </ApplyDesignSystemContextOverrides>
      </Modal>

      <LabelSchemaFormModal
        experimentId={experimentId}
        editingSchema={null}
        visible={createQuestionOpen}
        onClose={() => setCreateQuestionOpen(false)}
        onCreated={(schema) => setSelectedIds((prev) => new Set(prev).add(schema.schema_id))}
      />
    </>
  );
};
