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
import { useCurrentUserIsAdmin, useRolesQuery } from '../hooks';
import { useActiveWorkspace } from '../../workspaces/utils/WorkspaceUtils';
import type { Role } from '../types';

export interface RoleAssignmentValue {
  roleIds: number[];
}

export interface RoleAssignmentFormProps {
  value: RoleAssignmentValue;
  onChange: (value: RoleAssignmentValue) => void;
  disabled?: boolean;
}

export const ROLE_ASSIGNMENT_DEFAULT: RoleAssignmentValue = {
  roleIds: [],
};

const formatRoleLabel = (role: Role): string =>
  role.description ? `${role.workspace}/${role.name} — ${role.description}` : `${role.workspace}/${role.name}`;

/**
 * Multi-select picker for assigning one or more roles to a user. Roles
 * across all workspaces are shown in a single dropdown with workspace
 * prefix, so admins can pick roles from multiple workspaces in one go.
 * Used by EditAccessModal and CreateUserModal.
 */
export const RoleAssignmentForm = ({ value, onChange, disabled }: RoleAssignmentFormProps) => {
  const { theme } = useDesignSystemTheme();
  const [search, setSearch] = useState('');
  // Per-workspace scope: platform admins fetch unscoped; workspace
  // managers pass the active workspace. Suppress when none is active.
  const isAdmin = useCurrentUserIsAdmin();
  const activeWorkspace = useActiveWorkspace();
  const queryWorkspace = isAdmin ? undefined : (activeWorkspace ?? undefined);
  const queryEnabled = isAdmin || Boolean(activeWorkspace);
  const { data: rolesData, isLoading, error } = useRolesQuery(queryWorkspace, { enabled: queryEnabled });
  const roles = useMemo(() => rolesData?.roles ?? [], [rolesData]);

  // Pin "default" workspace's roles first; sort the rest alphabetically
  // by workspace then by role name, so the dropdown order is stable.
  const sortedRoles = useMemo(() => {
    return [...roles].sort((a, b) => {
      if (a.workspace !== b.workspace) {
        if (a.workspace === 'default') return -1;
        if (b.workspace === 'default') return 1;
        return a.workspace.localeCompare(b.workspace);
      }
      return a.name.localeCompare(b.name);
    });
  }, [roles]);

  const filteredRoles = useMemo(() => {
    const trimmed = search.trim().toLowerCase();
    if (!trimmed) return sortedRoles;
    return sortedRoles.filter((r) => formatRoleLabel(r).toLowerCase().includes(trimmed));
  }, [sortedRoles, search]);

  const selectedRoles = useMemo(() => roles.filter((r) => value.roleIds.includes(r.id)), [roles, value.roleIds]);

  const triggerText = useMemo(() => {
    if (selectedRoles.length === 0) return '';
    if (selectedRoles.length === 1) return formatRoleLabel(selectedRoles[0]);
    return `${selectedRoles.length} roles selected`;
  }, [selectedRoles]);

  const toggleRole = (roleId: number) => {
    const next = value.roleIds.includes(roleId)
      ? value.roleIds.filter((id) => id !== roleId)
      : [...value.roleIds, roleId];
    onChange({ roleIds: next });
  };

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      <div>
        <FieldLabel>Roles</FieldLabel>
        {!queryEnabled ? (
          <Typography.Text color="secondary">
            Select a workspace from the workspace selector to choose roles.
          </Typography.Text>
        ) : isLoading ? (
          <div css={{ padding: theme.spacing.sm }}>
            <Spinner size="small" />
          </div>
        ) : error ? (
          <Typography.Text color="error">Failed to load roles for this workspace.</Typography.Text>
        ) : roles.length === 0 ? (
          <Typography.Text color="secondary">No roles available. Create a role first.</Typography.Text>
        ) : (
          <DialogCombobox
            componentId="admin.role_assignment.roles"
            label="Roles"
            multiSelect
            value={selectedRoles.map(formatRoleLabel)}
          >
            <DialogComboboxTrigger
              withInlineLabel={false}
              placeholder="Select one or more roles"
              renderDisplayedValue={() => triggerText}
              onClear={() => onChange({ roleIds: [] })}
              width="100%"
              disabled={disabled}
            />
            <DialogComboboxContent style={{ zIndex: theme.options.zIndexBase + 100 }}>
              <DialogComboboxOptionList>
                <DialogComboboxOptionListSearch controlledValue={search} setControlledValue={setSearch}>
                  {filteredRoles.length === 0 ? (
                    <DialogComboboxOptionListCheckboxItem value="" checked={false} onChange={() => {}} disabled>
                      {search ? 'No matching roles' : 'No roles available'}
                    </DialogComboboxOptionListCheckboxItem>
                  ) : (
                    filteredRoles.map((role) => (
                      <DialogComboboxOptionListCheckboxItem
                        key={role.id}
                        value={formatRoleLabel(role)}
                        checked={value.roleIds.includes(role.id)}
                        onChange={() => toggleRole(role.id)}
                      />
                    ))
                  )}
                </DialogComboboxOptionListSearch>
              </DialogComboboxOptionList>
            </DialogComboboxContent>
          </DialogCombobox>
        )}
      </div>
    </div>
  );
};
