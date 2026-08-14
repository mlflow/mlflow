import { useEffect, useRef, useState } from 'react';
import {
  Button,
  CloseIcon,
  Table,
  TableCell,
  TableHeader,
  TableRow,
  Tag,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FieldLabel } from './FieldLabel';
import { formatResourcePattern, getResourceTypeLabel } from '../types';
import {
  RolePermissionForm,
  ROLE_PERMISSION_DRAFT_DEFAULT,
  draftToResourcePattern,
  isRolePermissionDraftFillable,
  type RolePermissionDraft,
} from './RolePermissionForm';

// Returns true when the draft differs from the default — i.e. the user has
// touched the picker. Used to gate the ``Clear`` affordance and the parent
// modal's discard-confirm dialog so we only nag about a draft the user
// actually engaged with.
const isDraftDirty = (draft: RolePermissionDraft): boolean =>
  draft.resourceType !== ROLE_PERMISSION_DRAFT_DEFAULT.resourceType ||
  draft.scope !== ROLE_PERMISSION_DRAFT_DEFAULT.scope ||
  draft.resourceId !== ROLE_PERMISSION_DRAFT_DEFAULT.resourceId ||
  draft.permission !== ROLE_PERMISSION_DRAFT_DEFAULT.permission;

/**
 * One staged role permission. ``id`` is set when the row was pre-loaded
 * from the role's current permissions (so EditRoleModal can call
 * ``removePermission(id)`` to revoke it on submit). Newly-staged rows
 * have ``id === undefined``; the parent's diff treats those as adds.
 */
export interface StagedRolePermission {
  id?: number;
  resourceType: string;
  resourcePattern: string;
  permission: string;
}

export interface RolePermissionsSectionProps {
  value: StagedRolePermission[];
  onChange: (value: StagedRolePermission[]) => void;
  /** The role's workspace, displayed read-only when the draft picks ``workspace`` resource type. */
  workspace?: string;
  disabled?: boolean;
  /** Called when the in-progress draft becomes dirty (any field touched
   * away from the default) or returns to clean. Parent modals use this to
   * pop a discard-confirm dialog on submit so the admin can't silently
   * abandon a partially-filled permission. */
  onUnsavedDraftChange?: (hasUnsavedDraft: boolean) => void;
}

/**
 * Wraps ``RolePermissionForm`` with a staged-list pattern: each "Add"
 * appends a row to the parent's list; rows can be removed individually;
 * the parent submits the whole list. The draft (resource type + scope
 * radio + resource picker) mirrors ``DirectPermissionsSection`` so the
 * role-creation experience is consistent with user-creation's direct
 * permissions.
 */
export const RolePermissionsSection = ({
  value,
  onChange,
  workspace,
  disabled,
  onUnsavedDraftChange,
}: RolePermissionsSectionProps) => {
  const { theme } = useDesignSystemTheme();
  const [draft, setDraft] = useState<RolePermissionDraft>(ROLE_PERMISSION_DRAFT_DEFAULT);

  const canAdd = isRolePermissionDraftFillable(draft);
  // ``dirty`` covers any field changed away from the default — parent modal
  // uses it to gate its discard-confirm dialog. ``hasUnsavedInvalidDraft``
  // narrows to the dirty + scope=specific + no-resource shape since that's
  // the only case where the inline "Select a specific X" guidance applies.
  const dirty = isDraftDirty(draft);
  const hasUnsavedInvalidDraft = dirty && !canAdd;

  // Ref-wrap so the reporting effect can depend only on the value it
  // reports — otherwise an inline-arrow callback would re-fire every render.
  const onUnsavedDraftChangeRef = useRef(onUnsavedDraftChange);
  useEffect(() => {
    onUnsavedDraftChangeRef.current = onUnsavedDraftChange;
  }, [onUnsavedDraftChange]);
  useEffect(() => {
    onUnsavedDraftChangeRef.current?.(dirty);
  }, [dirty]);

  const handleAdd = () => {
    if (!canAdd) return;
    const resourcePattern = draftToResourcePattern(draft);
    // Skip exact (resource_type, resource_pattern, permission) duplicates
    // — same trio collapses to one row regardless of whether the original
    // came from the backend or was just typed.
    const isDuplicate = value.some(
      (p) =>
        p.resourceType === draft.resourceType &&
        p.resourcePattern === resourcePattern &&
        p.permission === draft.permission,
    );
    if (isDuplicate) {
      setDraft(ROLE_PERMISSION_DRAFT_DEFAULT);
      return;
    }
    onChange([
      ...value,
      {
        resourceType: draft.resourceType,
        resourcePattern,
        permission: draft.permission,
      },
    ]);
    setDraft(ROLE_PERMISSION_DRAFT_DEFAULT);
  };

  const handleRemove = (index: number) => {
    onChange(value.filter((_, i) => i !== index));
  };

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      {value.length > 0 && (
        <Table scrollable noMinHeight css={{ border: `1px solid ${theme.colors.border}` }}>
          <TableRow isHeader>
            <TableHeader componentId="admin.role_permissions.staged_type" css={{ flex: 1 }}>
              Resource type
            </TableHeader>
            <TableHeader componentId="admin.role_permissions.staged_pattern" css={{ flex: 1 }}>
              Pattern
            </TableHeader>
            <TableHeader componentId="admin.role_permissions.staged_level" css={{ flex: 1 }}>
              Permission
            </TableHeader>
            <TableHeader
              componentId="admin.role_permissions.staged_actions"
              css={{ flex: 0, minWidth: 60, maxWidth: 60 }}
            >
              {' '}
            </TableHeader>
          </TableRow>
          {value.map((p, i) => (
            <TableRow key={`${p.resourceType}-${p.resourcePattern}-${p.permission}-${i}`}>
              <TableCell css={{ flex: 1 }}>
                <Tag componentId="admin.role_permissions.staged_type_tag">{getResourceTypeLabel(p.resourceType)}</Tag>
              </TableCell>
              <TableCell css={{ flex: 1 }}>
                <code>{formatResourcePattern(p.resourcePattern)}</code>
              </TableCell>
              <TableCell css={{ flex: 1 }}>
                <Tag componentId="admin.role_permissions.staged_level_tag" color="indigo">
                  {p.permission}
                </Tag>
              </TableCell>
              <TableCell css={{ flex: 0, minWidth: 60, maxWidth: 60 }}>
                <Button
                  componentId="admin.role_permissions.staged_remove"
                  type="tertiary"
                  size="small"
                  icon={<CloseIcon />}
                  aria-label={`Remove ${p.resourceType} ${formatResourcePattern(p.resourcePattern)}`}
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
        <RolePermissionForm
          value={draft}
          onChange={setDraft}
          workspace={workspace}
          disabled={disabled}
          showResourceRequiredError={hasUnsavedInvalidDraft}
        />
        <div css={{ display: 'flex', justifyContent: 'flex-end', gap: theme.spacing.sm }}>
          {dirty && (
            // Escape hatch: if the admin changed their mind about staging
            // a draft, ``Clear`` resets to the default so the parent's
            // discard-confirm dialog stops asking.
            <Button
              componentId="admin.role_permissions.clear"
              type="tertiary"
              onClick={() => setDraft(ROLE_PERMISSION_DRAFT_DEFAULT)}
              disabled={disabled}
            >
              Clear
            </Button>
          )}
          <Button componentId="admin.role_permissions.add" onClick={handleAdd} disabled={!canAdd || disabled}>
            Add
          </Button>
        </div>
      </div>
    </div>
  );
};
