import { useMemo, useState } from 'react';
import {
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxOptionList,
  DialogComboboxOptionListSearch,
  DialogComboboxOptionListSelectItem,
  DialogComboboxTrigger,
  Input,
  Radio,
  SimpleSelect,
  SimpleSelectOption,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FieldLabel } from './FieldLabel';
import { useResourceOptionsQuery } from '../hooks';
import { ALL_RESOURCE_PATTERN_LABEL, PERMISSIONS, RESOURCE_TYPES } from '../types';
import { DIRECT_GRANT_RESOURCE_TYPES, type DirectGrantResourceType } from './DirectPermissionForm';

export type RolePermissionScope = 'specific' | 'all';

/**
 * Internal draft state of a role-permission entry. Distinct from the
 * canonical ``StagedRolePermission`` (which carries ``resourcePattern``)
 * because the form needs to track ``scope`` independently of the picked
 * resource id. ``RolePermissionsSection`` translates this draft into a
 * ``StagedRolePermission`` on Add.
 */
export interface RolePermissionDraft {
  resourceType: string;
  scope: RolePermissionScope;
  /**
   * Holds the chosen resource id when ``scope === 'specific'``, or the
   * free-text pattern when ``resourceType === 'scorer'`` (which has no
   * pre-built resource picker today).
   */
  resourceId: string;
  permission: string;
}

export interface RolePermissionFormProps {
  value: RolePermissionDraft;
  onChange: (value: RolePermissionDraft) => void;
  /** The role's workspace, displayed read-only when ``workspace`` resource type is picked. */
  workspace?: string;
  disabled?: boolean;
}

export const ROLE_PERMISSION_DRAFT_DEFAULT: RolePermissionDraft = {
  resourceType: RESOURCE_TYPES[0],
  scope: 'all',
  resourceId: '',
  permission: PERMISSIONS[0],
};

const RESOURCE_TYPE_LABEL: Record<DirectGrantResourceType, string> = {
  experiment: 'Experiment',
  registered_model: 'Registered model',
  gateway_secret: 'Gateway secret',
  gateway_endpoint: 'Gateway endpoint',
  gateway_model_definition: 'Gateway model definition',
};

const isPickableResourceType = (rt: string): rt is DirectGrantResourceType =>
  (DIRECT_GRANT_RESOURCE_TYPES as readonly string[]).includes(rt);

/**
 * Add a permission to a role. Mirrors ``DirectPermissionForm``'s
 * (resource type + scope radio + resource picker) shape so the
 * role-creation experience reads the same as user-creation's direct
 * permissions.
 *
 * Two role-only special cases:
 * - ``workspace``: the only valid pattern is ``*`` (the role's own
 *   workspace), so the scope radio is hidden and we render a static
 *   "Workspace: <name>" line.
 * - ``scorer``: no pre-built picker exists for composite scorer ids,
 *   so the scope radio is hidden and we fall back to a free-text
 *   pattern input. ``RolePermissionsSection`` reads the typed value
 *   off ``resourceId`` for this branch.
 */
export const RolePermissionForm = ({ value, onChange, workspace, disabled }: RolePermissionFormProps) => {
  const { theme } = useDesignSystemTheme();
  const [resourceSearch, setResourceSearch] = useState('');

  const isPickable = isPickableResourceType(value.resourceType);
  const typeLabel = isPickable
    ? RESOURCE_TYPE_LABEL[value.resourceType as DirectGrantResourceType]
    : value.resourceType;

  const {
    options: resourceOptions,
    isLoading: resourceOptionsLoading,
    error: resourceOptionsError,
  } = useResourceOptionsQuery(value.resourceType);

  const filteredOptions = useMemo(() => {
    const trimmed = resourceSearch.trim().toLowerCase();
    if (!trimmed) return resourceOptions;
    return resourceOptions.filter(
      (o) => o.name.toLowerCase().includes(trimmed) || o.id.toLowerCase().includes(trimmed),
    );
  }, [resourceOptions, resourceSearch]);

  const selectedOption = resourceOptions.find((o) => o.id === value.resourceId);
  const renderOption = (o: { id: string; name: string }) => (o.name === o.id ? o.name : `${o.name} (${o.id})`);

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      <div>
        <FieldLabel>Resource Type</FieldLabel>
        <SimpleSelect
          id="admin-role-permission-form-resource-type"
          componentId="admin.role_permission_form.resource_type"
          value={value.resourceType}
          onChange={({ target }) => {
            const next = target.value;
            // Reset scope/resourceId when switching types so a stale
            // resource id doesn't leak across resource types.
            onChange({
              ...value,
              resourceType: next,
              scope: 'all',
              resourceId: '',
            });
          }}
          disabled={disabled}
        >
          {RESOURCE_TYPES.map((rt) => (
            <SimpleSelectOption key={rt} value={rt}>
              {rt}
            </SimpleSelectOption>
          ))}
        </SimpleSelect>
      </div>

      {value.resourceType === 'workspace' ? (
        <div>
          <FieldLabel>Workspace</FieldLabel>
          <Typography.Text color="secondary">
            {workspace ?? 'default'}{' '}
            <Typography.Text color="secondary" size="sm">
              (this grant applies to the role's workspace)
            </Typography.Text>
          </Typography.Text>
        </div>
      ) : value.resourceType === 'scorer' ? (
        <div>
          <FieldLabel>Resource Pattern</FieldLabel>
          <Input
            componentId="admin.role_permission_form.scorer_pattern"
            value={value.resourceId}
            onChange={(e) => onChange({ ...value, scope: 'specific', resourceId: e.target.value })}
            placeholder='Specific scorer id, or "all" to apply to every scorer'
            disabled={disabled}
          />
          <Typography.Text color="secondary" size="sm" css={{ display: 'block', marginTop: theme.spacing.xs }}>
            Scorer resource ids are composite (experiment_id + scorer_name); the picker isn't available yet.
          </Typography.Text>
        </div>
      ) : (
        <>
          <div>
            <FieldLabel>Scope</FieldLabel>
            <Radio.Group
              componentId="admin.role_permission_form.scope"
              name="admin-role-permission-form-scope"
              value={value.scope}
              onChange={(e) =>
                onChange({
                  ...value,
                  scope: e.target.value as RolePermissionScope,
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
                componentId="admin.role_permission_form.resource_id"
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
                <DialogComboboxContent
                  style={{ zIndex: theme.options.zIndexBase + 100 }}
                  loading={resourceOptionsLoading}
                >
                  {resourceOptionsError && (
                    <div css={{ padding: theme.spacing.sm, color: theme.colors.textValidationDanger }}>
                      Failed to load {typeLabel.toLowerCase()}s
                    </div>
                  )}
                  <DialogComboboxOptionList>
                    <DialogComboboxOptionListSearch
                      controlledValue={resourceSearch}
                      setControlledValue={setResourceSearch}
                    >
                      {filteredOptions.length === 0 && !resourceOptionsLoading ? (
                        <DialogComboboxOptionListSelectItem value="" onChange={() => {}} checked={false} disabled>
                          {resourceSearch ? 'No matching results' : 'No resources found'}
                        </DialogComboboxOptionListSelectItem>
                      ) : (
                        filteredOptions.map((option) => (
                          <DialogComboboxOptionListSelectItem
                            key={option.id}
                            value={option.id}
                            onChange={(v) => {
                              onChange({ ...value, resourceId: v });
                              setResourceSearch('');
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
        </>
      )}

      <div>
        <FieldLabel>Permission</FieldLabel>
        <SimpleSelect
          id="admin-role-permission-form-level"
          componentId="admin.role_permission_form.permission_level"
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

/** True when the draft is ready to be added to the staged list. */
export const isRolePermissionDraftFillable = (draft: RolePermissionDraft): boolean => {
  if (draft.resourceType === 'workspace') return true;
  if (draft.resourceType === 'scorer') return draft.resourceId.trim().length > 0;
  return draft.scope === 'all' || (draft.scope === 'specific' && draft.resourceId.trim().length > 0);
};

/**
 * Translate a draft into the canonical ``resourcePattern`` (the
 * user-facing label string — ``"all"`` or a specific id — that the
 * staged list stores). ``ROLE_PERMISSION_DRAFT_DEFAULT`` returns
 * ``"all"``.
 */
export const draftToResourcePattern = (draft: RolePermissionDraft): string => {
  if (draft.resourceType === 'workspace') return ALL_RESOURCE_PATTERN_LABEL;
  if (draft.resourceType === 'scorer') return draft.resourceId.trim();
  return draft.scope === 'all' ? ALL_RESOURCE_PATTERN_LABEL : draft.resourceId.trim();
};
