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
import {
  CONDITION_RESOURCE_TYPES,
  getConditionParentType,
  getResourceTypeLabel,
  conditionParentHasPicker,
  isConditionEmpty,
} from '../types';

export type MutationConditionScope = 'all' | 'parent';

/**
 * Internal draft state of one mutation condition.
 *
 * Distinct from the persisted ``MutationCondition`` in two ways. ``scope`` is tracked
 * separately from ``parentResourceId`` so switching to "all" does not lose a parent the
 * admin already picked, and the two filters are plain strings here because an empty
 * string is a field the admin has not filled while the server's ``null`` means the
 * filter is absent.
 */
export interface MutationConditionDraft {
  resourceType: string;
  scope: MutationConditionScope;
  /** Chosen parent id when ``scope === 'parent'``; empty otherwise. */
  parentResourceId: string;
  /** Constrains the values a request may set. Empty means no request filter. */
  valueCondition: string;
  /** Constrains which existing resources may be mutated. Empty means no target filter. */
  targetCondition: string;
}

export const MUTATION_CONDITION_DRAFT_DEFAULT: MutationConditionDraft = {
  resourceType: CONDITION_RESOURCE_TYPES[0],
  scope: 'all',
  parentResourceId: '',
  valueCondition: '',
  targetCondition: '',
};

export interface MutationConditionFormProps {
  value: MutationConditionDraft;
  onChange: (value: MutationConditionDraft) => void;
  /** The role's workspace, so the parent picker lists resources from the right one. */
  workspace?: string;
  disabled?: boolean;
  /** Render the inline reminder that at least one filter is required. */
  showFilterRequiredError?: boolean;
  /** Render the inline reminder that a parent must be picked for ``scope === 'parent'``. */
  showParentRequiredError?: boolean;
}

/**
 * Add a mutation condition to a role. Shaped like ``RolePermissionForm`` so the two
 * configuration surfaces read the same way.
 */
export const MutationConditionForm = ({
  value,
  onChange,
  workspace,
  disabled,
  showFilterRequiredError = false,
  showParentRequiredError = false,
}: MutationConditionFormProps) => {
  const { theme } = useDesignSystemTheme();
  const [parentSearch, setParentSearch] = useState('');

  const typeLabel = getResourceTypeLabel(value.resourceType);
  const parentType = getConditionParentType(value.resourceType);
  const parentLabel = parentType ? getResourceTypeLabel(parentType) : undefined;
  const hasPicker = conditionParentHasPicker(parentType);

  // The picker lists the PARENT type, not the conditioned type: a scope confines the
  // condition to one parent's children, and the children themselves are never named.
  const {
    options: parentOptions,
    isLoading: parentOptionsLoading,
    error: parentOptionsError,
  } = useResourceOptionsQuery(hasPicker ? (parentType ?? '') : '', workspace);

  const filteredParents = useMemo(() => {
    const trimmed = parentSearch.trim().toLowerCase();
    if (!trimmed) return parentOptions;
    return parentOptions.filter((o) => o.name.toLowerCase().includes(trimmed) || o.id.toLowerCase().includes(trimmed));
  }, [parentOptions, parentSearch]);

  const selectedParent = parentOptions.find((o) => o.id === value.parentResourceId);
  const renderOption = (o: { id: string; name: string }) => (o.name === o.id ? o.name : `${o.name} (${o.id})`);

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      <div>
        <FieldLabel>Resource type</FieldLabel>
        <SimpleSelect
          id="admin-mutation-condition-form-resource-type"
          componentId="admin.mutation_condition_form.resource_type"
          value={value.resourceType}
          onChange={({ target }) => {
            // Reset the scope and parent when the type changes: the parent type changes
            // with it, so a carried-over id would name a resource of the wrong type and
            // persist a scope that can never match.
            onChange({
              ...value,
              resourceType: target.value,
              scope: 'all',
              parentResourceId: '',
            });
            setParentSearch('');
          }}
          disabled={disabled}
        >
          {CONDITION_RESOURCE_TYPES.map((rt) => (
            <SimpleSelectOption key={rt} value={rt}>
              {getResourceTypeLabel(rt)}
            </SimpleSelectOption>
          ))}
        </SimpleSelect>
      </div>

      {parentType ? (
        <div>
          <FieldLabel>Scope</FieldLabel>
          <Radio.Group
            componentId="admin.mutation_condition_form.scope"
            name="admin-mutation-condition-form-scope"
            value={value.scope}
            onChange={(e) =>
              onChange({
                ...value,
                scope: e.target.value as MutationConditionScope,
                parentResourceId: '',
              })
            }
            layout="vertical"
          >
            <Radio value="all">All {typeLabel.toLowerCase()}s in the workspace</Radio>
            <Radio value="parent">Only within one {parentLabel?.toLowerCase()}</Radio>
          </Radio.Group>
          {value.scope === 'parent' && (
            <div css={{ marginTop: theme.spacing.sm }}>
              <FieldLabel>{parentLabel}</FieldLabel>
              {showParentRequiredError && (
                <Typography.Text
                  color="error"
                  size="sm"
                  css={{ display: 'block', marginBottom: theme.spacing.xs }}
                  data-testid="admin.mutation_condition_form.parent_required_error"
                >
                  Select a {parentLabel?.toLowerCase()} or switch the scope to{' '}
                  <strong>All {typeLabel.toLowerCase()}s in the workspace</strong>.
                </Typography.Text>
              )}
              {!hasPicker ? (
                // No lister for this parent type, so the id is typed. The server still
                // validates the pair, but a typo persists as a scope that matches
                // nothing -- which reads as "the condition is not working" rather than
                // as a bad id -- so say that here.
                <Input
                  componentId="admin.mutation_condition_form.parent_resource_id_text"
                  value={value.parentResourceId}
                  onChange={(e) => onChange({ ...value, parentResourceId: e.target.value })}
                  placeholder={`Enter ${parentLabel?.toLowerCase()} name`}
                  disabled={disabled}
                />
              ) : (
                <DialogCombobox
                  componentId="admin.mutation_condition_form.parent_resource_id"
                  label={parentLabel ?? 'Parent'}
                  value={value.parentResourceId ? [value.parentResourceId] : []}
                >
                  <DialogComboboxTrigger
                    withInlineLabel={false}
                    placeholder={`Select ${parentLabel?.toLowerCase()}`}
                    renderDisplayedValue={() =>
                      selectedParent ? renderOption(selectedParent) : value.parentResourceId
                    }
                    onClear={() => onChange({ ...value, parentResourceId: '' })}
                    width="100%"
                    disabled={disabled}
                  />
                  <DialogComboboxContent
                    style={{ zIndex: theme.options.zIndexBase + 100 }}
                    loading={parentOptionsLoading}
                  >
                    {parentOptionsError && (
                      <div css={{ padding: theme.spacing.sm, color: theme.colors.textValidationDanger }}>
                        Failed to load {parentLabel?.toLowerCase()}s
                      </div>
                    )}
                    <DialogComboboxOptionList>
                      <DialogComboboxOptionListSearch
                        controlledValue={parentSearch}
                        setControlledValue={setParentSearch}
                      >
                        {filteredParents.length === 0 && !parentOptionsLoading ? (
                          <DialogComboboxOptionListSelectItem value="" onChange={() => {}} checked={false} disabled>
                            {parentSearch ? 'No matching results' : 'No resources found'}
                          </DialogComboboxOptionListSelectItem>
                        ) : (
                          filteredParents.map((option) => (
                            <DialogComboboxOptionListSelectItem
                              key={option.id}
                              value={option.id}
                              onChange={(v) => {
                                onChange({ ...value, parentResourceId: v });
                                setParentSearch('');
                              }}
                              checked={option.id === value.parentResourceId}
                            >
                              {renderOption(option)}
                            </DialogComboboxOptionListSelectItem>
                          ))
                        )}
                      </DialogComboboxOptionListSearch>
                    </DialogComboboxOptionList>
                  </DialogComboboxContent>
                </DialogCombobox>
              )}
            </div>
          )}
        </div>
      ) : null}

      <div>
        <FieldLabel>Request filter (optional)</FieldLabel>
        <Input
          componentId="admin.mutation_condition_form.value_condition"
          value={value.valueCondition}
          onChange={(e) => onChange({ ...value, valueCondition: e.target.value })}
          placeholder="tag_value != 'prod'"
          disabled={disabled}
        />
      </div>

      <div>
        <FieldLabel>Resource filter (optional)</FieldLabel>
        <Input
          componentId="admin.mutation_condition_form.target_condition"
          value={value.targetCondition}
          onChange={(e) => onChange({ ...value, targetCondition: e.target.value })}
          placeholder="tags.lifecycle != 'prod'"
          disabled={disabled}
        />
      </div>

      {showFilterRequiredError && (
        <Typography.Text color="error" size="sm" data-testid="admin.mutation_condition_form.filter_required_error">
          Enter at least one filter. A condition with neither would restrict nothing.
        </Typography.Text>
      )}
    </div>
  );
};

/** True when the draft is ready to be submitted. */
export const isMutationConditionDraftFillable = (draft: MutationConditionDraft): boolean => {
  if (isConditionEmpty(draft.valueCondition, draft.targetCondition)) return false;
  if (draft.scope === 'parent') return draft.parentResourceId.trim().length > 0;
  return true;
};

/** True when any field has been touched away from the default. */
export const isMutationConditionDraftDirty = (draft: MutationConditionDraft): boolean =>
  draft.resourceType !== MUTATION_CONDITION_DRAFT_DEFAULT.resourceType ||
  draft.scope !== MUTATION_CONDITION_DRAFT_DEFAULT.scope ||
  draft.parentResourceId !== MUTATION_CONDITION_DRAFT_DEFAULT.parentResourceId ||
  draft.valueCondition !== MUTATION_CONDITION_DRAFT_DEFAULT.valueCondition ||
  draft.targetCondition !== MUTATION_CONDITION_DRAFT_DEFAULT.targetCondition;

/**
 * Translate a draft into the condition a staged list holds.
 *
 * The single place this mapping lives. It used to be duplicated in the staging
 * section, which meant the tested copy and the copy actually used could drift -- and
 * drift here produces a malformed write.
 *
 * Both parent fields travel together or neither does (the server enforces the pair with
 * a CHECK constraint), the parent type is derived from the resource type rather than
 * carried in the draft, and an empty filter becomes ``null`` rather than ``""`` --
 * an empty string reaches the condition parser and is reported as a filter syntax error
 * instead of as an absent filter.
 */
export const draftToStagedCondition = (draft: MutationConditionDraft) => {
  const parentType = getConditionParentType(draft.resourceType);
  const scoped = draft.scope === 'parent' && parentType && draft.parentResourceId.trim();
  return {
    resourceType: draft.resourceType,
    parentResourceType: scoped ? parentType : null,
    parentResourceId: scoped ? draft.parentResourceId.trim() : null,
    valueCondition: draft.valueCondition.trim() || null,
    targetCondition: draft.targetCondition.trim() || null,
  };
};
