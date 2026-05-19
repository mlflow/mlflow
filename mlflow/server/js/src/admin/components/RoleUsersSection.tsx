import { useMemo, useState } from 'react';
import {
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxOptionList,
  DialogComboboxOptionListCheckboxItem,
  DialogComboboxOptionListSearch,
  DialogComboboxTrigger,
  Spinner,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FieldLabel } from './FieldLabel';
import { useUsersQuery } from '../hooks';

export interface RoleUsersSectionProps {
  value: string[];
  onChange: (value: string[]) => void;
  disabled?: boolean;
}

/**
 * Multi-select picker for assigning users to a role. Mirrors
 * ``RoleAssignmentForm`` (the role multi-select used in user creation
 * + access editing) so the experience is consistent across role
 * creation and role editing.
 */
export const RoleUsersSection = ({ value, onChange, disabled }: RoleUsersSectionProps) => {
  const { theme } = useDesignSystemTheme();
  const [search, setSearch] = useState('');
  const { data: usersData, isLoading } = useUsersQuery();
  const users = useMemo(() => usersData?.users ?? [], [usersData]);

  const sortedUsers = useMemo(() => [...users].sort((a, b) => a.username.localeCompare(b.username)), [users]);

  const filteredUsers = useMemo(() => {
    const trimmed = search.trim().toLowerCase();
    if (!trimmed) return sortedUsers;
    return sortedUsers.filter((u) => u.username.toLowerCase().includes(trimmed));
  }, [sortedUsers, search]);

  const selectedSet = useMemo(() => new Set(value), [value]);

  const triggerText = useMemo(() => {
    if (value.length === 0) return '';
    if (value.length === 1) return value[0];
    return `${value.length} users selected`;
  }, [value]);

  const toggleUser = (username: string) => {
    if (selectedSet.has(username)) {
      onChange(value.filter((u) => u !== username));
    } else {
      onChange([...value, username]);
    }
  };

  return (
    <div>
      <FieldLabel>Users</FieldLabel>
      {isLoading ? (
        <div css={{ padding: theme.spacing.sm }}>
          <Spinner size="small" />
        </div>
      ) : users.length === 0 ? (
        <Typography.Text color="secondary">No users available.</Typography.Text>
      ) : (
        <DialogCombobox componentId="admin.role_users.users" label="Users" multiSelect value={value}>
          <DialogComboboxTrigger
            withInlineLabel={false}
            placeholder="Select one or more users"
            renderDisplayedValue={() => triggerText}
            onClear={() => onChange([])}
            width="100%"
            disabled={disabled}
          />
          <DialogComboboxContent style={{ zIndex: theme.options.zIndexBase + 100 }}>
            <DialogComboboxOptionList>
              <DialogComboboxOptionListSearch controlledValue={search} setControlledValue={setSearch}>
                {filteredUsers.length === 0 ? (
                  <DialogComboboxOptionListCheckboxItem value="" checked={false} onChange={() => {}} disabled>
                    {search ? 'No matching users' : 'No users available'}
                  </DialogComboboxOptionListCheckboxItem>
                ) : (
                  filteredUsers.map((user) => (
                    <DialogComboboxOptionListCheckboxItem
                      key={user.username}
                      value={user.username}
                      checked={selectedSet.has(user.username)}
                      onChange={() => toggleUser(user.username)}
                    />
                  ))
                )}
              </DialogComboboxOptionListSearch>
            </DialogComboboxOptionList>
          </DialogComboboxContent>
        </DialogCombobox>
      )}
    </div>
  );
};
