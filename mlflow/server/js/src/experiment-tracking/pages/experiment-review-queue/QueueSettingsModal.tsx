import { useMemo, useState } from 'react';

import {
  ApplyDesignSystemContextOverrides,
  Button,
  ChevronDownIcon,
  ChevronRightIcon,
  FormUI,
  Input,
  Modal,
  TableSkeleton,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

import {
  LabelSchemaInputRenderer,
  LabelSchemaFormModal,
  useListLabelSchemasQuery,
} from '../../components/label-schemas';
import { OwnerCombobox } from './OwnerCombobox';
import { QuestionChecklistCombobox } from './QuestionChecklistCombobox';
import { sameUser } from './queuePermissions';
import { MAX_ASSIGNED_USERS, ReviewerChecklistCombobox } from './ReviewerChecklistCombobox';
import { useIsAuthAvailable } from '../../../account/hooks';
import Utils from '../../../common/utils/Utils';
import { useAssignableUsersQuery } from './hooks/useAssignableUsersQuery';
import { useListReviewQueueItemsQuery } from './hooks/useListReviewQueueItemsQuery';
import { useUpdateReviewQueueMutation } from './hooks/useUpdateReviewQueueMutation';
import type { ReviewQueue } from './types';

const CID = 'mlflow.experiment-review-queue.queue-settings';

/**
 * "Manage queue" modal for a CUSTOM queue (opened from the right-pane gear). The
 * modal only opens for someone who can manage the queue — an experiment manager
 * or the owning EDITor. The name and assigned members are editable by either;
 * the questions (also frozen once the queue has traces) and a new
 * owner-reassignment field can only be changed by an experiment manager
 * (`canManage`). Deletion lives on the gear menu, not here. Personal USER queues
 * aren't managed here.
 */
export const QueueSettingsModal = ({
  queue,
  canManage,
  onClose,
}: {
  queue: ReviewQueue;
  canManage: boolean;
  onClose: () => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const authAvailable = useIsAuthAvailable();
  // Any authenticated user may list users server-side, and the modal only opens
  // for someone who can edit this queue's members, so the roster is fetched
  // whenever auth is on.
  const canListUsers = authAvailable;

  const { labelSchemas, isLoading: schemasLoading } = useListLabelSchemasQuery({ experimentId: queue.experiment_id });
  const { items: traces, isLoading: itemsLoading } = useListReviewQueueItemsQuery({ queueId: queue.queue_id });
  const { users: assignableUsers, isLoading: usersLoading } = useAssignableUsersQuery({ enabled: canListUsers });
  const { updateReviewQueueAsync, isUpdatingQueue } = useUpdateReviewQueueMutation();

  // Questions are an experiment-manager concern (`canManage`) and additionally
  // freeze once the queue has traces (the backend rejects schema changes then),
  // so the picker is read-only in either case. Default to frozen until the count
  // loads, so it doesn't flash editable for a queue with traces.
  const canEditQuestions = canManage && !itemsLoading && traces.length === 0;

  const [name, setName] = useState(queue.name);
  // The current owner is implicitly a member (they own the queue) and can't be
  // unassigned, so keep them out of the selectable roster and the toggleable
  // member set, then re-add on save — consistent with the create flow, where the
  // creator is auto-assigned and hidden from the picker.
  const owner = queue.created_by;
  const withoutOwner = (users: string[]) => users.filter((u) => !owner || !sameUser(u, owner));
  // Owner reassignment is manager-only; a free-text new owner pre-filled with
  // the current one. Distinct from `owner` above, which is the immutable key
  // used to filter the reviewer picker.
  const [newOwner, setNewOwner] = useState(queue.created_by ?? '');

  const [selectedSchemaIds, setSelectedSchemaIds] = useState<Set<string>>(new Set(queue.schema_ids ?? []));
  const [members, setMembers] = useState<Set<string>>(() => new Set(withoutOwner(queue.users ?? [])));
  const [createQuestionOpen, setCreateQuestionOpen] = useState(false);

  const selectedSchemas = useMemo(
    () => labelSchemas.filter((s) => selectedSchemaIds.has(s.schema_id)),
    [labelSchemas, selectedSchemaIds],
  );
  const questionsTriggerValue = useMemo(
    () =>
      selectedSchemas.length > 0
        ? [
            intl.formatMessage(
              {
                defaultMessage: '{count, plural, one {# question} other {# questions}} selected',
                description: 'Queue settings: questions dropdown selected-count summary',
              },
              { count: selectedSchemas.length },
            ),
          ]
        : [],
    [selectedSchemas, intl],
  );
  const toggleSchema = (schemaId: string) =>
    setSelectedSchemaIds((prev) => {
      const next = new Set(prev);
      if (next.has(schemaId)) {
        next.delete(schemaId);
      } else {
        next.add(schemaId);
      }
      return next;
    });
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

  // Reviewers picker: searchable multi-select checkbox list of assignable users
  // (same UX as the create modal and the "Flag for review" picker), so members
  // are chosen from the roster rather than free text.
  const usernames = useMemo(
    () => withoutOwner(assignableUsers.map((u) => u.username).filter((u): u is string => Boolean(u))),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [assignableUsers, owner],
  );
  // Owner picker spans the whole roster (anyone with experiment EDIT can own a
  // queue), unlike the reviewers list which hides the current owner.
  const ownerCandidates = useMemo(
    () => assignableUsers.map((u) => u.username).filter((u): u is string => Boolean(u)),
    [assignableUsers],
  );
  const toggleReviewer = (username: string) =>
    setMembers((prev) => {
      const next = new Set(prev);
      if (next.has(username)) {
        next.delete(username);
      } else {
        next.add(username);
      }
      return next;
    });
  const reviewersTriggerValue = useMemo(
    () =>
      members.size > 0
        ? [
            intl.formatMessage(
              {
                defaultMessage: '{count, plural, one {# reviewer} other {# reviewers}} selected',
                description: 'Queue settings: reviewers dropdown selected-count summary',
              },
              { count: members.size },
            ),
          ]
        : [],
    [members, intl],
  );

  const trimmedName = name.trim();
  const nameChanged = trimmedName !== queue.name;
  const trimmedNewOwner = newOwner.trim();
  // Owner can only change for a manager; ignore a cleared field (the backend
  // requires a non-empty owner) so blanking it is a no-op rather than an error.
  const ownerChanged = canManage && trimmedNewOwner !== '' && trimmedNewOwner !== (queue.created_by ?? '');
  const canSave = trimmedName !== '';
  // Dedupe the stored members before diffing so a duplicate in `queue.users`
  // can't read as a spurious change (and trigger a needless update_users write).
  const originalMembers = useMemo(
    () => new Set(withoutOwner(queue.users ?? [])),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [queue.users, owner],
  );
  const membersChanged = members.size !== originalMembers.size || [...originalMembers].some((m) => !members.has(m));
  const originalSchemaIds = useMemo(() => new Set(queue.schema_ids ?? []), [queue.schema_ids]);
  const questionsChanged =
    selectedSchemaIds.size !== originalSchemaIds.size ||
    [...originalSchemaIds].some((id) => !selectedSchemaIds.has(id));

  const handleSave = async () => {
    try {
      await updateReviewQueueAsync({
        queue_id: queue.queue_id,
        // Send each field only when it actually changed: a repeated `users` write
        // is an `update_users` that would otherwise clobber a concurrent edit, and
        // name / owner skip the backend's rename collision and owner re-validation.
        // The owner is re-added to `users` (they're always a member) since they're
        // hidden from the picker.
        ...(membersChanged ? { users: Array.from(new Set([...(owner ? [owner] : []), ...members])) } : {}),
        ...(nameChanged ? { name: trimmedName } : {}),
        ...(ownerChanged ? { new_owner: trimmedNewOwner } : {}),
        // Only send schema_ids when they're still editable AND actually changed:
        // once the queue has traces (or the user lacks MANAGE) the backend freezes
        // them, and a no-op rewrite would needlessly re-write the whole association
        // set and clobber a concurrent question edit.
        ...(canEditQuestions && questionsChanged ? { schema_ids: [...selectedSchemaIds] } : {}),
      });
      onClose();
    } catch (e) {
      // Surface the failure (e.g. a duplicate queue name) as a toast rather than an
      // inline alert, so the modal body (and the Save button) don't shift. The modal
      // stays open so the user can correct and retry.
      Utils.displayGlobalErrorNotification(
        intl.formatMessage(
          {
            defaultMessage: 'Failed to save the queue settings: {error}',
            description: 'Queue settings: error toast shown when saving fails',
          },
          { error: e instanceof Error ? e.message : String(e) },
        ),
      );
    }
  };

  const dropdownZIndex = theme.options.zIndexBase + 100;

  return (
    <>
      <Modal
        componentId={`${CID}.modal`}
        visible={!createQuestionOpen}
        destroyOnClose
        title={intl.formatMessage(
          { defaultMessage: 'Queue settings — “{name}”', description: 'Queue settings modal title' },
          { name: queue.name },
        )}
        onCancel={onClose}
        footer={
          <div css={{ display: 'flex', justifyContent: 'flex-end' }}>
            <Button
              componentId={`${CID}.save`}
              type="primary"
              loading={isUpdatingQueue}
              disabled={isUpdatingQueue || !canSave}
              onClick={handleSave}
            >
              <FormattedMessage defaultMessage="Save" description="Queue settings: save button" />
            </Button>
          </div>
        }
      >
        <ApplyDesignSystemContextOverrides getPopupContainer={() => document.body}>
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
            <div>
              <FormUI.Label htmlFor={`${CID}.name-input`}>
                <FormattedMessage defaultMessage="Name" description="Queue settings: name field label" />
              </FormUI.Label>
              <Input
                componentId={`${CID}.name`}
                id={`${CID}.name-input`}
                value={name}
                onChange={(e) => setName(e.target.value)}
              />
            </div>

            {/* Owner reassignment is an experiment-manager action; an editor who
                owns the queue manages its name and members but not who owns it. */}
            {canManage && authAvailable && (
              <div>
                <FormUI.Label>
                  <FormattedMessage defaultMessage="Owner" description="Queue settings: owner field label" />
                </FormUI.Label>
                <FormUI.Hint css={{ marginBottom: theme.spacing.sm }}>
                  <FormattedMessage
                    defaultMessage="The owner can manage this queue with experiment EDIT access."
                    description="Queue settings: owner field hint"
                  />
                </FormUI.Hint>
                <OwnerCombobox
                  componentId={`${CID}.owner`}
                  usernames={ownerCandidates}
                  selectedUser={newOwner}
                  onSelect={setNewOwner}
                  dropdownZIndex={dropdownZIndex}
                  isLoading={usersLoading}
                />
              </div>
            )}

            {authAvailable && (
              <div>
                <FormUI.Label>
                  <FormattedMessage defaultMessage="Reviewers" description="Queue settings: members field label" />
                </FormUI.Label>
                <FormUI.Hint css={{ marginBottom: theme.spacing.sm }}>
                  <FormattedMessage
                    defaultMessage="Assign reviewers who should answer this queue's questions. They'll find it under “Feedback requested”."
                    description="Queue settings: members field hint"
                  />
                </FormUI.Hint>
                <ReviewerChecklistCombobox
                  componentId={`${CID}.reviewers`}
                  usernames={usernames}
                  checkedUsers={members}
                  onToggle={toggleReviewer}
                  triggerValue={reviewersTriggerValue}
                  dropdownZIndex={dropdownZIndex}
                  isLoading={usersLoading}
                  maxSelected={owner ? MAX_ASSIGNED_USERS - 1 : MAX_ASSIGNED_USERS}
                />
              </div>
            )}

            <div>
              <FormUI.Label>
                <FormattedMessage defaultMessage="Questions" description="Queue settings: questions field label" />
              </FormUI.Label>
              <FormUI.Hint css={{ marginBottom: theme.spacing.sm }}>
                {canEditQuestions ? (
                  <FormattedMessage
                    defaultMessage="Choose which questions reviewers answer for traces in this queue."
                    description="Queue settings: questions field hint"
                  />
                ) : !canManage ? (
                  <FormattedMessage
                    defaultMessage="Only an experiment manager can change a queue's questions."
                    description="Queue settings: questions manager-only hint"
                  />
                ) : (
                  <FormattedMessage
                    defaultMessage="Questions are locked once the queue has traces attached."
                    description="Queue settings: questions frozen hint"
                  />
                )}
              </FormUI.Hint>
              {schemasLoading ? (
                <TableSkeleton lines={2} />
              ) : (
                <QuestionChecklistCombobox
                  componentId={`${CID}.questions`}
                  schemas={labelSchemas}
                  checkedIds={selectedSchemaIds}
                  onToggle={toggleSchema}
                  onCreateQuestion={canEditQuestions ? () => setCreateQuestionOpen(true) : undefined}
                  triggerValue={questionsTriggerValue}
                  disabled={!canEditQuestions}
                  dropdownZIndex={dropdownZIndex}
                />
              )}

              {/* Live preview of the selected questions, as a reviewer will see them.
               Always mounted so the grid-row transition can animate height. */}
              <div
                css={{
                  display: 'grid',
                  gridTemplateRows: selectedSchemas.length > 0 ? '1fr' : '0fr',
                  transition: 'grid-template-rows 200ms ease-out',
                  marginTop: selectedSchemas.length > 0 ? theme.spacing.sm : 0,
                }}
              >
                <div css={{ overflow: 'hidden' }}>
                  <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
                    <FormUI.Label>
                      <FormattedMessage
                        defaultMessage="Question preview"
                        description="Queue settings: question preview section label"
                      />
                    </FormUI.Label>
                    <FormUI.Hint css={{ marginBottom: theme.spacing.xs }}>
                      <FormattedMessage
                        defaultMessage="Preview how the questions will appear for the human reviewer."
                        description="Queue settings: question preview section hint"
                      />
                    </FormUI.Hint>
                    <div
                      css={{
                        display: 'flex',
                        flexDirection: 'column',
                        gap: theme.spacing.sm,
                        // Fixed height (~4 collapsed rows) so the modal doesn't resize;
                        // more questions or an expanded preview scroll within this section.
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
                              css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs, cursor: 'pointer' }}
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
                                {/* Mirror the create-question form preview: a question
                                    that collects an optional rationale shows the box here too. */}
                                {schema.enable_comment && (
                                  <Input.TextArea
                                    componentId={`${CID}.preview.rationale`}
                                    rows={2}
                                    disabled
                                    placeholder={intl.formatMessage({
                                      defaultMessage: 'Rationale (optional)',
                                      description: 'Review question preview free-form rationale placeholder',
                                    })}
                                  />
                                )}
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
          </div>
        </ApplyDesignSystemContextOverrides>
      </Modal>

      <LabelSchemaFormModal
        experimentId={queue.experiment_id}
        editingSchema={null}
        visible={createQuestionOpen}
        onClose={() => setCreateQuestionOpen(false)}
        onCreated={(schema) => setSelectedSchemaIds((prev) => new Set(prev).add(schema.schema_id))}
      />
    </>
  );
};
