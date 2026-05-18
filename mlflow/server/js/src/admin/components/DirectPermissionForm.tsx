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
  Tooltip,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FieldLabel } from './FieldLabel';
import { useResourceOptionsQuery } from '../hooks';
import { formatResourceType, PERMISSIONS } from '../types';

// Resource types eligible for per-user direct grants. ``workspace`` is excluded
// (it's role-only), and ``scorer`` is excluded because its identifier is
// composite (experiment_id + scorer_name) and the form below assumes a single
// string id. ``gateway_model_definition`` stays in the tuple so existing grants
// from earlier RBAC versions still display correctly, but it's filtered out of
// ``PICKER_RESOURCE_TYPES`` below since it's an internal primitive that ships
// behind ``gateway_endpoint`` and shouldn't be a new-grant option.
export const DIRECT_GRANT_RESOURCE_TYPES = [
  'experiment',
  'registered_model',
  'gateway_secret',
  'gateway_endpoint',
  'gateway_model_definition',
] as const;

export type DirectGrantResourceType = (typeof DIRECT_GRANT_RESOURCE_TYPES)[number];

const PICKER_RESOURCE_TYPES: readonly DirectGrantResourceType[] = DIRECT_GRANT_RESOURCE_TYPES.filter(
  (rt) => rt !== 'gateway_model_definition',
);

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
  /** Optional: open the Create Role flow pre-filled. Wired by the parent. */
  onCreateRoleForAllOfType?: (resourceType: DirectGrantResourceType) => void;
  disabled?: boolean;
}

export const DIRECT_PERMISSION_DEFAULT: DirectPermissionValue = {
  resourceType: 'experiment',
  scope: 'specific',
  resourceId: '',
  permission: PERMISSIONS[0],
};

/** Submit only when a specific resource is picked. ``scope === 'all'`` lands post-Phase 2. */
export const isDirectPermissionSubmittable = (value: DirectPermissionValue): boolean =>
  value.scope === 'specific' && Boolean(value.resourceId);

/**
 * Pick a per-user direct permission. Two-axis form: specific resource vs.
 * "all of type", and which resource. The "all of type" option is rendered
 * disabled with a tooltip — it requires the post-Phase 2 synthetic-role API.
 * The shape of ``DirectPermissionValue`` already accommodates the future
 * enable: drop the ``disabled`` and route the submit through whatever
 * synthetic-role grant API ships.
 */
export const DirectPermissionForm = ({
  value,
  onChange,
  onCreateRoleForAllOfType,
  disabled,
}: DirectPermissionFormProps) => {
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
  const typeLabel = formatResourceType(value.resourceType);

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      <div>
        <FieldLabel>Resource type</FieldLabel>
        <SimpleSelect
          id="admin-direct-permission-resource-type"
          componentId="admin.direct_permission.resource_type"
          value={value.resourceType}
          onChange={({ target }) =>
            onChange({
              ...value,
              resourceType: target.value as DirectGrantResourceType,
              resourceId: '',
            })
          }
          disabled={disabled}
        >
          {PICKER_RESOURCE_TYPES.map((rt) => (
            <SimpleSelectOption key={rt} value={rt}>
              {formatResourceType(rt)}
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
          <Tooltip
            componentId="admin.direct_permission.all_disabled_tooltip"
            content={`Granting on all ${typeLabel.toLowerCase()}s directly is coming soon. For a similar effect today, create a role with this permission and assign it.`}
            side="right"
          >
            <Radio value="all" disabled>
              All {typeLabel.toLowerCase()}s
            </Radio>
          </Tooltip>
        </Radio.Group>
        {value.scope === 'all' && onCreateRoleForAllOfType && (
          <Typography.Text color="secondary" css={{ display: 'block', marginTop: theme.spacing.xs }}>
            <Typography.Link
              componentId="admin.direct_permission.create_role_link"
              onClick={() => onCreateRoleForAllOfType(value.resourceType)}
            >
              Create a role with this permission instead →
            </Typography.Link>
          </Typography.Text>
        )}
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
          {PERMISSIONS.map((p) => (
            <SimpleSelectOption key={p} value={p}>
              {p}
            </SimpleSelectOption>
          ))}
        </SimpleSelect>
      </div>
    </div>
  );
};
