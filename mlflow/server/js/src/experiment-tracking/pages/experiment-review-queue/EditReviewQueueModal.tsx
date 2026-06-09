import { useEffect, useMemo, useState } from 'react';

import {
  Alert,
  Button,
  FormUI,
  Modal,
  Tag,
  Typography,
  TypeaheadComboboxInput,
  TypeaheadComboboxMenu,
  TypeaheadComboboxMenuItem,
  TypeaheadComboboxRoot,
  useComboboxState,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

import { useCurrentUserIsAdmin, useCurrentUserIsWorkspaceAdmin, useIsAuthAvailable } from '../../../account/hooks';
import { useAssignableUsersQuery } from './hooks/useAssignableUsersQuery';
import { useCanManageReviews } from './hooks/useCanManageReviews';
import { useDeleteReviewQueueMutation } from './hooks/useDeleteReviewQueueMutation';
import { useReviewer } from './hooks/useReviewer';
import { useUpdateReviewQueueMutation } from './hooks/useUpdateReviewQueueMutation';
import { canDeleteQueue } from './queuePermissions';
import type { ReviewQueue } from './types';

const CID = 'mlflow.experiment-review-queue.edit-queue';

/**
 * Edit a CUSTOM queue's assigned members, and delete it. Members are entered as
 * free text that autocompletes against the workspace's assignable users (so a
 * user not yet listed, or a reviewer on a server without user listing, can still
 * be typed in). The default queue's questions are fixed and it can't be deleted,
 * so only its membership is editable here. Opened from the sidebar gear; auth
 * servers only (a no-auth server has a single user, so there's nothing to
 * assign).
 */
export const EditReviewQueueModal = ({
  queue,
  onClose,
  onDeleted,
}: {
  queue: ReviewQueue;
  onClose: () => void;
  onDeleted: () => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const reviewer = useReviewer();
  const authAvailable = useIsAuthAvailable();
  const isAdmin = useCurrentUserIsAdmin();
  const isWorkspaceAdmin = useCurrentUserIsWorkspaceAdmin();
  const canManage = useCanManageReviews(queue.experiment_id);
  const canDelete = canDeleteQueue(queue, reviewer, authAvailable, canManage);

  // User listing is workspace-admin gated server-side; gate the query so a
  // non-admin reviewer doesn't 403. Free-text entry still works without it.
  const canListUsers = authAvailable && (isAdmin || isWorkspaceAdmin);
  const { users: assignableUsers } = useAssignableUsersQuery({ enabled: canListUsers });
  const { updateReviewQueueAsync, isUpdatingQueue, error: updateError } = useUpdateReviewQueueMutation();
  const { deleteReviewQueueAsync, isDeletingQueue, error: deleteError } = useDeleteReviewQueueMutation();

  const [members, setMembers] = useState<string[]>(queue.users ?? []);
  const [query, setQuery] = useState('');
  const [selectedItem, setSelectedItem] = useState<string | null>(null);

  const usernames = useMemo(
    () => assignableUsers.map((u) => u.username).filter((u): u is string => Boolean(u)),
    [assignableUsers],
  );

  // useComboboxState filters `allItems` into `items` via the matcher as the user
  // types; seed/refresh the working list when the assignable users load.
  const [filteredUsers, setFilteredUsers] = useState<(string | null)[]>([]);
  useEffect(() => {
    setFilteredUsers(usernames);
  }, [usernames]);

  // Candidate suggestions: assignable users not already added, plus the typed
  // value itself so a free-text (unlisted) member can always be added.
  const items = useMemo(() => {
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
    items,
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
    await updateReviewQueueAsync({ queue_id: queue.queue_id, users: members });
    onClose();
  };

  const handleDelete = async () => {
    await deleteReviewQueueAsync({ queue_id: queue.queue_id });
    onDeleted();
    onClose();
  };

  const title = queue.is_default
    ? intl.formatMessage({
        defaultMessage: 'Edit default queue',
        description: 'Edit review queue modal: title for the default queue',
      })
    : intl.formatMessage(
        { defaultMessage: 'Edit “{name}”', description: 'Edit review queue modal: title for a custom queue' },
        { name: queue.name },
      );

  return (
    <Modal
      componentId={`${CID}.modal`}
      visible
      title={title}
      okText={<FormattedMessage defaultMessage="Save" description="Edit review queue: save button" />}
      okButtonProps={{ loading: isUpdatingQueue, disabled: isUpdatingQueue || isDeletingQueue }}
      cancelText={<FormattedMessage defaultMessage="Cancel" description="Edit review queue: cancel button" />}
      onOk={handleSave}
      onCancel={onClose}
    >
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
        {queue.is_default && (
          <Typography.Hint>
            <FormattedMessage
              defaultMessage="The default queue asks every question in the experiment and can't be deleted. You can still change who's assigned to it."
              description="Edit review queue: default-queue explanation"
            />
          </Typography.Hint>
        )}

        <div>
          <FormUI.Label htmlFor={`${CID}.member-typeahead-input`}>
            <FormattedMessage defaultMessage="Reviewers" description="Edit review queue: members field label" />
          </FormUI.Label>
          <FormUI.Hint css={{ marginBottom: theme.spacing.sm }}>
            <FormattedMessage
              defaultMessage="Assign reviewers by name. They'll find this queue under “Feedback requested”."
              description="Edit review queue: members field hint"
            />
          </FormUI.Hint>
          <TypeaheadComboboxRoot id={`${CID}.member-typeahead`} comboboxState={comboboxState}>
            <TypeaheadComboboxInput
              id={`${CID}.member-typeahead-input`}
              placeholder={intl.formatMessage({
                defaultMessage: 'Add a reviewer by username or email',
                description: 'Edit review queue: member typeahead placeholder',
              })}
              comboboxState={comboboxState}
              formOnChange={addMember}
              onPressEnter={() => {
                if (items.length > 0) {
                  addMember(items[0]);
                }
              }}
              allowClear
            />
            <TypeaheadComboboxMenu comboboxState={comboboxState}>
              {items.map((item, index) => (
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
                description="Edit review queue: empty members state"
              />
            </Typography.Hint>
          )}
        </div>

        {(updateError || deleteError) && (
          <Alert
            componentId={`${CID}.error`}
            type="error"
            closable={false}
            message={intl.formatMessage({
              defaultMessage: 'Failed to save changes to the review queue.',
              description: 'Edit review queue: error alert title',
            })}
            description={(updateError ?? deleteError)?.message}
          />
        )}

        {canDelete && (
          <div css={{ borderTop: `1px solid ${theme.colors.border}`, paddingTop: theme.spacing.md }}>
            <Button componentId={`${CID}.delete`} danger loading={isDeletingQueue} onClick={handleDelete}>
              <FormattedMessage defaultMessage="Delete queue" description="Edit review queue: delete button" />
            </Button>
          </div>
        )}
      </div>
    </Modal>
  );
};
