import { useMemo, useState } from 'react';
import {
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxOptionList,
  DialogComboboxOptionListSearch,
  DialogComboboxOptionListSelectItem,
  DialogComboboxTrigger,
  Radio,
  SimpleSelect,
  SimpleSelectOption,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FieldLabel } from './FieldLabel';
import { useResourceOptionsQuery } from '../hooks';
import { PERMISSIONS, getGrantablePermissions, getResourceTypeLabel } from '../types';

// Resource types eligible for per-user direct grants. ``workspace`` is excluded
// because the backend's ``grant_user_resource_permission`` rejects it (workspace
// grants are role-only by design — see ``_reject_workspace_resource_type``).
export const DIRECT_GRANT_RESOURCE_TYPES = [
  'experiment',
  'registered_model',
  'prompt',
  'scorer',
  'gateway_secret',
  'gateway_endpoint',
] as const;

export type DirectGrantResourceType = (typeof DIRECT_GRANT_RESOURCE_TYPES)[number];

export type DirectPermissionScope = 'specific' | 'all';

export interface DirectPermissionValue {
  resourceType: DirectGrantResourceType;
  scope: DirectPermissionScope;
  resourceId: string;
  permission: string;
}

export interface DirectPermissionFormProps {
  value: DirectPermissionValue;
  onChange: (value: DirectPermissionValue) => void;
  disabled?: boolean;
}

export const DIRECT_PERMISSION_DEFAULT: DirectPermissionValue = {
  resourceType: 'experiment',
  scope: 'specific',
  resourceId: '',
  permission: PERMISSIONS[0],
};

/** Submit when ``scope === 'all'``, or when ``scope === 'specific'`` and the
 * picker has produced a non-empty resource id. */
export const isDirectPermissionSubmittable = (value: DirectPermissionValue): boolean =>
  value.scope === 'all' || (value.scope === 'specific' && Boolean(value.resourceId));

/**
 * Pick a per-user direct permission. ``resourceId`` only holds the user's
 * specific picked id; the wildcard pattern is derived from ``scope`` at
 * staging time so a resource literally named ``*`` can't masquerade as an
 * all-of-type grant.
 */
export const DirectPermissionForm = ({ value, onChange, disabled }: DirectPermissionFormProps) => {
  const { theme } = useDesignSystemTheme();
  const [search, setSearch] = useState('');
  const {
    options: resourceOptions,
    isLoading: resourceOptionsLoading,
    error: resourceOptionsError,
  } = useResourceOptionsQuery(value.resourceType);

  const filteredOptions = useMemo(() => {
    const trimmed = search.trim().toLowerCase();
    if (!trimmed) return resourceOptions;
    return resourceOptions.filter(
      (o) => o.name.toLowerCase().includes(trimmed) || o.id.toLowerCase().includes(trimmed),
    );
  }, [resourceOptions, search]);

  const selectedOption = resourceOptions.find((o) => o.id === value.resourceId);
  const renderOption = (o: { id: string; name: string }) => (o.name === o.id ? o.name : `${o.name} (${o.id})`);
  const typeLabel = getResourceTypeLabel(value.resourceType);

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      <div>
        <FieldLabel>Resource type</FieldLabel>
        <SimpleSelect
          id="admin-direct-permission-resource-type"
          componentId="admin.direct_permission.resource_type"
          value={value.resourceType}
          onChange={({ target }) => {
            const next = target.value as DirectGrantResourceType;
            const nextGrantable = getGrantablePermissions(next);
            const nextPermission = nextGrantable.includes(value.permission) ? value.permission : nextGrantable[0];
            onChange({
              ...value,
              resourceType: next,
              resourceId: '',
              permission: nextPermission,
            });
          }}
          disabled={disabled}
        >
          {DIRECT_GRANT_RESOURCE_TYPES.map((rt) => (
            <SimpleSelectOption key={rt} value={rt}>
              {getResourceTypeLabel(rt)}
            </SimpleSelectOption>
          ))}
        </SimpleSelect>
      </div>
      <div>
        <FieldLabel>Scope</FieldLabel>
        <Radio.Group
          componentId="admin.direct_permission.scope"
          name="admin-direct-permission-scope"
          value={value.scope}
          onChange={(e) =>
            onChange({
              ...value,
              scope: e.target.value as DirectPermissionScope,
              resourceId: '',
            })
          }
          layout="vertical"
        >
          <Radio value="specific">Specific {typeLabel.toLowerCase()}</Radio>
          <Radio value="all">All {typeLabel.toLowerCase()}s</Radio>
        </Radio.Group>
      </div>
      {value.scope === 'specific' && (
        <div>
          <FieldLabel>{typeLabel}</FieldLabel>
          <DialogCombobox
            componentId="admin.direct_permission.resource_id"
            label={typeLabel}
            value={value.resourceId ? [value.resourceId] : []}
          >
            <DialogComboboxTrigger
              withInlineLabel={false}
              placeholder={`Select ${typeLabel.toLowerCase()}`}
              renderDisplayedValue={() => (selectedOption ? renderOption(selectedOption) : value.resourceId)}
              onClear={() => onChange({ ...value, resourceId: '' })}
              width="100%"
              disabled={disabled}
            />
            <DialogComboboxContent style={{ zIndex: theme.options.zIndexBase + 100 }} loading={resourceOptionsLoading}>
              {resourceOptionsError && (
                <div css={{ padding: theme.spacing.sm, color: theme.colors.textValidationDanger }}>
                  Failed to load {typeLabel.toLowerCase()}s
                </div>
              )}
              <DialogComboboxOptionList>
                <DialogComboboxOptionListSearch controlledValue={search} setControlledValue={setSearch}>
                  {filteredOptions.length === 0 && !resourceOptionsLoading ? (
                    <DialogComboboxOptionListSelectItem value="" onChange={() => {}} checked={false} disabled>
                      {search ? 'No matching results' : 'No resources found'}
                    </DialogComboboxOptionListSelectItem>
                  ) : (
                    filteredOptions.map((option) => (
                      <DialogComboboxOptionListSelectItem
                        key={option.id}
                        value={option.id}
                        onChange={(v) => {
                          onChange({ ...value, resourceId: v });
                          setSearch('');
                        }}
                        checked={option.id === value.resourceId}
                      >
                        {renderOption(option)}
                      </DialogComboboxOptionListSelectItem>
                    ))
                  )}
                </DialogComboboxOptionListSearch>
              </DialogComboboxOptionList>
            </DialogComboboxContent>
          </DialogCombobox>
        </div>
      )}
      <div>
        <FieldLabel>Permission</FieldLabel>
        <SimpleSelect
          id="admin-direct-permission-level"
          componentId="admin.direct_permission.permission_level"
          value={value.permission}
          onChange={({ target }) => onChange({ ...value, permission: target.value })}
          disabled={disabled}
        >
          {getGrantablePermissions(value.resourceType).map((p) => (
            <SimpleSelectOption key={p} value={p}>
              {p}
            </SimpleSelectOption>
          ))}
        </SimpleSelect>
      </div>
    </div>
  );
};
