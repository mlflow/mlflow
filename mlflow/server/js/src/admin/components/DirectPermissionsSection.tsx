import { useState } from 'react';
import {
  Button,
  CloseIcon,
  Table,
  TableCell,
  TableHeader,
  TableRow,
  Tag,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FieldLabel } from './FieldLabel';
import {
  DirectPermissionForm,
  DIRECT_PERMISSION_DEFAULT,
  isDirectPermissionSubmittable,
  type DirectGrantResourceType,
  type DirectPermissionValue,
} from './DirectPermissionForm';

/**
 * One staged direct grant. ``scope`` is always ``'specific'`` today because
 * the "All <type>" path requires the post-Phase 2 synthetic-role API.
 */
export interface StagedDirectPermission {
  resourceType: DirectGrantResourceType;
  resourceId: string;
  permission: string;
}

export interface DirectPermissionsSectionProps {
  value: StagedDirectPermission[];
  onChange: (value: StagedDirectPermission[]) => void;
  /** Optional: bridge to the Create Role flow when "All <type>" is picked. */
  onCreateRoleForAllOfType?: (resourceType: DirectGrantResourceType) => void;
  disabled?: boolean;
}

/**
 * Wraps ``DirectPermissionForm`` with a staged-list pattern: each "Add"
 * appends a row to the parent's list; rows can be removed individually;
 * the parent submits the whole list. Mirrors ``RolePermissionsSection``.
 */
export const DirectPermissionsSection = ({
  value,
  onChange,
  onCreateRoleForAllOfType,
  disabled,
}: DirectPermissionsSectionProps) => {
  const { theme } = useDesignSystemTheme();
  const [draft, setDraft] = useState<DirectPermissionValue>(DIRECT_PERMISSION_DEFAULT);

  const handleAdd = () => {
    if (!isDirectPermissionSubmittable(draft)) return;
    // Skip exact duplicates so the staged list doesn't repeat the same
    // (type, id, permission) row.
    const isDuplicate = value.some(
      (p) =>
        p.resourceType === draft.resourceType && p.resourceId === draft.resourceId && p.permission === draft.permission,
    );
    if (isDuplicate) {
      setDraft(DIRECT_PERMISSION_DEFAULT);
      return;
    }
    onChange([
      ...value,
      { resourceType: draft.resourceType, resourceId: draft.resourceId, permission: draft.permission },
    ]);
    setDraft(DIRECT_PERMISSION_DEFAULT);
  };

  const handleRemove = (index: number) => {
    onChange(value.filter((_, i) => i !== index));
  };

  const canAdd = isDirectPermissionSubmittable(draft);

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      {value.length > 0 && (
        <Table scrollable noMinHeight css={{ border: `1px solid ${theme.colors.border}` }}>
          <TableRow isHeader>
            <TableHeader componentId="admin.direct_permissions.staged_type" css={{ flex: 1 }}>
              Resource type
            </TableHeader>
            <TableHeader componentId="admin.direct_permissions.staged_resource" css={{ flex: 2 }}>
              Resource
            </TableHeader>
            <TableHeader componentId="admin.direct_permissions.staged_level" css={{ flex: 1 }}>
              Permission
            </TableHeader>
            <TableHeader
              componentId="admin.direct_permissions.staged_actions"
              css={{ flex: 0, minWidth: 60, maxWidth: 60 }}
            >
              {' '}
            </TableHeader>
          </TableRow>
          {value.map((p, i) => (
            <TableRow key={`${p.resourceType}-${p.resourceId}-${p.permission}-${i}`}>
              <TableCell css={{ flex: 1 }}>
                <Tag componentId="admin.direct_permissions.staged_type_tag">{p.resourceType}</Tag>
              </TableCell>
              <TableCell css={{ flex: 2 }}>
                <code>{p.resourceId}</code>
              </TableCell>
              <TableCell css={{ flex: 1 }}>
                <Tag componentId="admin.direct_permissions.staged_level_tag" color="indigo">
                  {p.permission}
                </Tag>
              </TableCell>
              <TableCell css={{ flex: 0, minWidth: 60, maxWidth: 60 }}>
                <Button
                  componentId="admin.direct_permissions.staged_remove"
                  type="tertiary"
                  size="small"
                  icon={<CloseIcon />}
                  aria-label={`Remove ${p.resourceType} ${p.resourceId}`}
                  onClick={() => handleRemove(i)}
                  disabled={disabled}
                />
              </TableCell>
            </TableRow>
          ))}
        </Table>
      )}
      <div
        css={{
          border: `1px dashed ${theme.colors.border}`,
          borderRadius: theme.general.borderRadiusBase,
          padding: theme.spacing.md,
          display: 'flex',
          flexDirection: 'column',
          gap: theme.spacing.md,
        }}
      >
        <FieldLabel>Add a permission</FieldLabel>
        <DirectPermissionForm
          value={draft}
          onChange={setDraft}
          onCreateRoleForAllOfType={onCreateRoleForAllOfType}
          disabled={disabled}
        />
        <div css={{ display: 'flex', justifyContent: 'flex-end' }}>
          <Button componentId="admin.direct_permissions.add" onClick={handleAdd} disabled={!canAdd || disabled}>
            Add
          </Button>
        </div>
      </div>
    </div>
  );
};
