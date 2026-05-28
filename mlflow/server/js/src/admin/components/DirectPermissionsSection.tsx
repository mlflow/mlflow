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
import {
  DirectPermissionForm,
  DIRECT_PERMISSION_DEFAULT,
  isDirectPermissionSubmittable,
  type DirectGrantResourceType,
  type DirectPermissionValue,
} from './DirectPermissionForm';
import { ALL_RESOURCE_PATTERN, formatResourcePattern, getResourceTypeLabel } from '../types';

// Returns true when the draft differs from the default — i.e. the user has
// touched the picker. Used to gate the "Clear" affordance and the parent
// modal's submit lock so we only nag about a draft the user actually
// engaged with.
const isDraftDirty = (draft: DirectPermissionValue): boolean =>
  draft.resourceType !== DIRECT_PERMISSION_DEFAULT.resourceType ||
  draft.scope !== DIRECT_PERMISSION_DEFAULT.scope ||
  draft.resourceId !== DIRECT_PERMISSION_DEFAULT.resourceId ||
  draft.permission !== DIRECT_PERMISSION_DEFAULT.permission;

/**
 * One staged direct grant. ``resourceId`` carries either a specific resource
 * id or the wildcard ``'*'`` (for ``All <type>`` grants).
 */
export interface StagedDirectPermission {
  resourceType: DirectGrantResourceType;
  resourceId: string;
  permission: string;
}

export interface DirectPermissionsSectionProps {
  value: StagedDirectPermission[];
  onChange: (value: StagedDirectPermission[]) => void;
  /** Forwarded to the picker query so the admin can grant resources in a
   * workspace other than their session-active one. */
  workspace?: string;
  disabled?: boolean;
  /** Called when the in-progress draft transitions to a "dirty but not
   * yet submittable" state (e.g. user changed the resource type but
   * hasn't picked a specific resource yet). Parent modals use this to
   * block their submit button so the admin can't silently abandon a
   * partially-filled permission by clicking Create. */
  onUnsavedInvalidDraftChange?: (hasUnsavedInvalidDraft: boolean) => void;
}

/**
 * Wraps ``DirectPermissionForm`` with a staged-list pattern: each "Add"
 * appends a row to the parent's list; rows can be removed individually;
 * the parent submits the whole list. Mirrors ``RolePermissionsSection``.
 */
export const DirectPermissionsSection = ({
  value,
  onChange,
  workspace,
  disabled,
  onUnsavedInvalidDraftChange,
}: DirectPermissionsSectionProps) => {
  const { theme } = useDesignSystemTheme();
  const [draft, setDraft] = useState<DirectPermissionValue>(DIRECT_PERMISSION_DEFAULT);

  const handleAdd = () => {
    if (!isDirectPermissionSubmittable(draft)) return;
    // Translate scope to the API-bound resource id here (not in the draft)
    // so a resource literally named ``*`` can't shadow the wildcard.
    const resourceId = draft.scope === 'all' ? ALL_RESOURCE_PATTERN : draft.resourceId;
    const isDuplicate = value.some(
      (p) => p.resourceType === draft.resourceType && p.resourceId === resourceId && p.permission === draft.permission,
    );
    if (isDuplicate) {
      setDraft(DIRECT_PERMISSION_DEFAULT);
      return;
    }
    onChange([...value, { resourceType: draft.resourceType, resourceId, permission: draft.permission }]);
    setDraft(DIRECT_PERMISSION_DEFAULT);
  };

  const handleRemove = (index: number) => {
    onChange(value.filter((_, i) => i !== index));
  };

  const canAdd = isDirectPermissionSubmittable(draft);
  // A draft is "unsaved-invalid" once the user has changed something
  // (``isDraftDirty``) but it can't be staged yet (``!isSubmittable``).
  // This is the case the user typically hits when they pick a resource
  // type but forget to pick a specific resource — Kris's bug report.
  const dirty = isDraftDirty(draft);
  const hasUnsavedInvalidDraft = dirty && !canAdd;

  // Ref-wrap the callback so the reporting effect can depend only on the
  // value it reports — otherwise a future caller passing an inline arrow
  // function would re-fire the effect on every render. Current call sites
  // (state setters) are stable, but this is the safer contract.
  const onUnsavedInvalidDraftChangeRef = useRef(onUnsavedInvalidDraftChange);
  useEffect(() => {
    onUnsavedInvalidDraftChangeRef.current = onUnsavedInvalidDraftChange;
  }, [onUnsavedInvalidDraftChange]);
  useEffect(() => {
    onUnsavedInvalidDraftChangeRef.current?.(hasUnsavedInvalidDraft);
  }, [hasUnsavedInvalidDraft]);

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
                <Tag componentId="admin.direct_permissions.staged_type_tag">{getResourceTypeLabel(p.resourceType)}</Tag>
              </TableCell>
              <TableCell css={{ flex: 2 }}>
                <code>{formatResourcePattern(p.resourceId)}</code>
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
          workspace={workspace}
          disabled={disabled}
          // ``hasUnsavedInvalidDraft`` already implies ``scope === 'specific'``
          // because ``isDirectPermissionSubmittable`` is unconditionally true
          // for ``scope === 'all'`` (so the dirty+invalid intersection can't
          // happen at ``scope === 'all'``).
          showResourceRequiredError={hasUnsavedInvalidDraft}
        />
        <div css={{ display: 'flex', justifyContent: 'flex-end', gap: theme.spacing.sm }}>
          {dirty && (
            // Escape hatch: if the user changed their mind about staging a
            // draft, ``Clear`` resets to the default state so the parent
            // modal's submit unlocks. Only shown when ``dirty`` to avoid
            // bait-and-switch clicks on a button that does nothing.
            <Button
              componentId="admin.direct_permissions.clear"
              type="tertiary"
              onClick={() => setDraft(DIRECT_PERMISSION_DEFAULT)}
              disabled={disabled}
            >
              Clear
            </Button>
          )}
          <Button componentId="admin.direct_permissions.add" onClick={handleAdd} disabled={!canAdd || disabled}>
            Add
          </Button>
        </div>
      </div>
    </div>
  );
};
