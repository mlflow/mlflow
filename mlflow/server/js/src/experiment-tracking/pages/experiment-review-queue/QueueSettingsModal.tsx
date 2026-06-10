import { useEffect, useMemo, useState } from 'react';

import {
  Alert,
  ApplyDesignSystemContextOverrides,
  ChevronDownIcon,
  ChevronRightIcon,
  FormUI,
  Input,
  Modal,
  Tag,
  TableSkeleton,
  Typography,
  TypeaheadComboboxInput,
  TypeaheadComboboxMenu,
  TypeaheadComboboxMenuItem,
  TypeaheadComboboxRoot,
  useComboboxState,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

import { LabelSchemaInputRenderer, LabelSchemaFormModal, useListLabelSchemasQuery } from '../../components/label-schemas';
import { QuestionChecklistCombobox } from './QuestionChecklistCombobox';
import { useCurrentUserIsAdmin, useCurrentUserIsWorkspaceAdmin, useIsAuthAvailable } from '../../../account/hooks';
import { useAssignableUsersQuery } from './hooks/useAssignableUsersQuery';
import { useListReviewQueueItemsQuery } from './hooks/useListReviewQueueItemsQuery';
import { useUpdateReviewQueueMutation } from './hooks/useUpdateReviewQueueMutation';
import type { ReviewQueue } from './types';

const CID = 'mlflow.experiment-review-queue.queue-settings';

/**
 * "Manage queue" modal for a non-default CUSTOM queue (opened from the right-pane
 * gear). Edits the queue's questions (the experiment's label schemas it asks,
 * frozen once the queue has traces) and its assigned members (free-text input
 * autocompleting assignable users). The name is shown read-only for now — making
 * it editable needs an UpdateReviewQueue proto change. Deletion lives on the gear
 * menu, not here. Personal USER queues and the default queue aren't managed here.
 */
export const QueueSettingsModal = ({ queue, onClose }: { queue: ReviewQueue; onClose: () => void }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const authAvailable = useIsAuthAvailable();
  const isAdmin = useCurrentUserIsAdmin();
  const isWorkspaceAdmin = useCurrentUserIsWorkspaceAdmin();
  // User listing is workspace-admin gated server-side; gate the query so a
  // non-admin doesn't 403. Free-text member entry still works without it.
  const canListUsers = authAvailable && (isAdmin || isWorkspaceAdmin);

  const { labelSchemas, isLoading: schemasLoading } = useListLabelSchemasQuery({ experimentId: queue.experiment_id });
  const { items: traces, isLoading: itemsLoading } = useListReviewQueueItemsQuery({ queueId: queue.queue_id });
  const { users: assignableUsers } = useAssignableUsersQuery({ enabled: canListUsers });
  const { updateReviewQueueAsync, isUpdatingQueue, error: updateError } = useUpdateReviewQueueMutation();

  // Questions freeze once the queue has traces (the backend rejects schema
  // changes then), so the picker is read-only in that case. Default to frozen
  // until the count loads, so it doesn't flash editable for a queue with traces.
  const canEditQuestions = !itemsLoading && traces.length === 0;

  const [selectedSchemaIds, setSelectedSchemaIds] = useState<Set<string>>(new Set(queue.schema_ids ?? []));
  const [members, setMembers] = useState<string[]>(queue.users ?? []);
  const [createQuestionOpen, setCreateQuestionOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [selectedItem, setSelectedItem] = useState<string | null>(null);

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

  // Members typeahead: free text autocompleting assignable users, with the typed
  // value injected so an unlisted user (or a user on a server without listing)
  // can be added too.
  const usernames = useMemo(
    () => assignableUsers.map((u) => u.username).filter((u): u is string => Boolean(u)),
    [assignableUsers],
  );
  const [filteredUsers, setFilteredUsers] = useState<(string | null)[]>([]);
  useEffect(() => {
    setFilteredUsers(usernames);
  }, [usernames]);
  const memberItems = useMemo(() => {
    const base = filteredUsers.filter((u): u is string => typeof u === 'string' && !members.includes(u));
    const typed = query.trim();
    if (typed && !members.includes(typed) && !base.includes(typed)) {
      return [typed, ...base];
    }
    return base;
  }, [filteredUsers, members, query]);
  const addMember = (value: string | null) => {
    const name = value?.trim();
    if (name && !members.includes(name)) {
      setMembers((prev) => [...prev, name]);
    }
    setSelectedItem(null);
    setQuery('');
  };
  const comboboxState = useComboboxState<string | null>({
    componentId: `${CID}.member-typeahead`,
    allItems: usernames,
    items: memberItems,
    setItems: setFilteredUsers,
    multiSelect: false,
    setInputValue: setQuery,
    itemToString: (item) => item ?? '',
    matcher: (item, q) => (item ?? '').toLowerCase().includes(q.toLowerCase()),
    formValue: selectedItem,
    formOnChange: addMember,
    preventUnsetOnBlur: true,
  });
  const removeMember = (name: string) => setMembers((prev) => prev.filter((m) => m !== name));

  const handleSave = async () => {
    await updateReviewQueueAsync({
      queue_id: queue.queue_id,
      users: members,
      // Only send schema_ids when they're still editable; once the queue has
      // traces the backend freezes them.
      ...(canEditQuestions ? { schema_ids: [...selectedSchemaIds] } : {}),
    });
    onClose();
  };

  const dropdownZIndex = theme.options.zIndexBase + 100;

  return (
    <>
    <Modal
      componentId={`${CID}.modal`}
      visible={!createQuestionOpen}
      title={intl.formatMessage(
        { defaultMessage: 'Queue settings — “{name}”', description: 'Queue settings modal title' },
        { name: queue.name },
      )}
      okText={<FormattedMessage defaultMessage="Save" description="Queue settings: save button" />}
      okButtonProps={{ loading: isUpdatingQueue, disabled: isUpdatingQueue }}
      cancelText={<FormattedMessage defaultMessage="Cancel" description="Queue settings: cancel button" />}
      onOk={handleSave}
      onCancel={onClose}
    >
      <ApplyDesignSystemContextOverrides getPopupContainer={() => document.body}>
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
          <div>
            <FormUI.Label htmlFor={`${CID}.name-input`}>
              <FormattedMessage defaultMessage="Name" description="Queue settings: name field label" />
            </FormUI.Label>
            {/* Renaming a queue needs an UpdateReviewQueue proto change; read-only for now. */}
            <Input componentId={`${CID}.name`} id={`${CID}.name-input`} value={queue.name} disabled />
          </div>

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

          {authAvailable && (
            <div>
              <FormUI.Label htmlFor={`${CID}.member-typeahead-input`}>
                <FormattedMessage defaultMessage="Reviewers" description="Queue settings: members field label" />
              </FormUI.Label>
              <FormUI.Hint css={{ marginBottom: theme.spacing.sm }}>
                <FormattedMessage
                  defaultMessage="Assign reviewers by name. They'll find this queue under “Feedback requested”."
                  description="Queue settings: members field hint"
                />
              </FormUI.Hint>
              <TypeaheadComboboxRoot id={`${CID}.member-typeahead`} comboboxState={comboboxState}>
                <TypeaheadComboboxInput
                  id={`${CID}.member-typeahead-input`}
                  placeholder={intl.formatMessage({
                    defaultMessage: 'Add a reviewer by username or email',
                    description: 'Queue settings: member typeahead placeholder',
                  })}
                  comboboxState={comboboxState}
                  formOnChange={addMember}
                  onPressEnter={() => {
                    if (memberItems.length > 0) {
                      addMember(memberItems[0]);
                    }
                  }}
                  allowClear
                />
                <TypeaheadComboboxMenu comboboxState={comboboxState}>
                  {memberItems.map((item, index) => (
                    <TypeaheadComboboxMenuItem key={item ?? ''} item={item} index={index} comboboxState={comboboxState}>
                      {item ?? ''}
                    </TypeaheadComboboxMenuItem>
                  ))}
                </TypeaheadComboboxMenu>
              </TypeaheadComboboxRoot>
              {members.length > 0 ? (
                <div css={{ display: 'flex', flexWrap: 'wrap', gap: theme.spacing.xs, marginTop: theme.spacing.sm }}>
                  {members.map((member) => (
                    <Tag key={member} componentId={`${CID}.member-tag`} closable onClose={() => removeMember(member)}>
                      {member}
                    </Tag>
                  ))}
                </div>
              ) : (
                <Typography.Hint css={{ marginTop: theme.spacing.sm }}>
                  <FormattedMessage
                    defaultMessage="No reviewers assigned yet."
                    description="Queue settings: empty members state"
                  />
                </Typography.Hint>
              )}
            </div>
          )}

          {updateError && (
            <Alert
              componentId={`${CID}.error`}
              type="error"
              closable={false}
              message={intl.formatMessage({
                defaultMessage: 'Failed to save the queue settings.',
                description: 'Queue settings: error alert title',
              })}
              description={updateError.message}
            />
          )}
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
