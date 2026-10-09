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
  getConditionContainerType,
  getResourceTypeLabel,
  conditionScopeHasPicker,
  conditionNarrowsByPattern,
  CONDITION_WILDCARD_PATTERN,
  CONDITION_CONTAINER_WORKSPACE,
  isConditionEmpty,
  getConditionRequestIdentifiers,
  getConditionResourceIdentifiers,
} from '../types';

/**
 * Help text under a condition input.
 *
 * Exists for one reason: the two fields accept overlapping syntax and disagree on what an
 * ABSENT tag means. `tags.a = 'x'` is valid in both, and passes a request that does not set
 * `a` while refusing a resource that does not have `a`. An admin cannot infer that from the
 * field labels, and a clause copied from one field to the other inverts silently, so each
 * field states its own absence rule.
 */
const FieldHint = ({ children, testId }: { children: React.ReactNode; testId: string }) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div css={{ marginTop: theme.spacing.xs }} data-testid={testId}>
      <Typography.Text size="sm" color="secondary">
        {children}
      </Typography.Text>
    </div>
  );
};

export type MutationConditionScope = 'all' | 'scoped';

/**
 * Internal draft state of one mutation condition.
 *
 * Distinct from the persisted ``MutationCondition`` in two ways. ``scope`` is tracked
 * separately from ``scopePattern`` so switching to "all" does not lose an id the admin
 * already picked, and the two filters are plain strings here because an empty string is a
 * field the admin has not filled while the server's ``null`` means the filter is absent.
 *
 * One ``scopePattern`` covers both server axes because a type only ever narrows on one of
 * them: a top-level type by its own ``resource_pattern``, a sub-resource by its
 * ``container_resource_pattern``. Which axis applies is derived from the type and never
 * carried here, so the draft cannot describe a scope the server would reject.
 */
export interface MutationConditionDraft {
  resourceType: string;
  scope: MutationConditionScope;
  /** Chosen id when ``scope === 'scoped'``; empty otherwise. */
  scopePattern: string;
  /** Constrains the values a request may set. Empty means no value condition. */
  valueCondition: string;
  /** Constrains which existing resources may be mutated. Empty means no target condition. */
  targetCondition: string;
}

export const MUTATION_CONDITION_DRAFT_DEFAULT: MutationConditionDraft = {
  resourceType: CONDITION_RESOURCE_TYPES[0],
  scope: 'all',
  scopePattern: '',
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
  /** Render the inline reminder that an id must be picked for ``scope === 'scoped'``. */
  showScopeRequiredError?: boolean;
  /**
   * Identifiers in each filter that the selected type does not accept, computed by the
   * staging section so the same answer gates its Add button -- a form that only *showed*
   * the problem would still let the condition be staged and then refused by the server.
   */
  unsupportedValueIdentifiers?: string[];
  unsupportedTargetIdentifiers?: string[];
}

/** ``a``, ``b`` and ``c`` as inline code, for a vocabulary list. */
const renderIdentifiers = (identifiers: string[]) =>
  identifiers.map((identifier, i) => (
    <span key={identifier}>
      {i > 0 && (i === identifiers.length - 1 ? ' and ' : ', ')}
      <code>{identifier}</code>
    </span>
  ));

const describeUnsupported = (unsupported: string[], typeLabel: string): string =>
  `${unsupported.map((i) => `'${i}'`).join(', ')} ${unsupported.length === 1 ? 'is' : 'are'} not available on ` +
  `${typeLabel}. The server refuses a condition naming it, so it cannot be saved.`;

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
  showScopeRequiredError = false,
  unsupportedValueIdentifiers = [],
  unsupportedTargetIdentifiers = [],
}: MutationConditionFormProps) => {
  const { theme } = useDesignSystemTheme();
  const [parentSearch, setParentSearch] = useState('');

  const typeLabel = getResourceTypeLabel(value.resourceType);
  // Which axis this type narrows on, and therefore what the picker lists. A top-level type
  // names resources of its OWN type; a sub-resource names its container, because the
  // children are not individually addressable at the grain its grants use.
  const narrowsByPattern = conditionNarrowsByPattern(value.resourceType);
  const containerType = getConditionContainerType(value.resourceType);
  const scopeType = narrowsByPattern ? value.resourceType : containerType;
  const scopeLabel = scopeType ? getResourceTypeLabel(scopeType) : undefined;
  const hasPicker = conditionScopeHasPicker(scopeType);

  // The picker lists the PARENT type, not the conditioned type: a scope confines the
  // condition to one parent's children, and the children themselves are never named.
  const {
    options: parentOptions,
    isLoading: parentOptionsLoading,
    error: parentOptionsError,
  } = useResourceOptionsQuery(hasPicker ? (scopeType ?? '') : '', workspace);

  const filteredParents = useMemo(() => {
    const trimmed = parentSearch.trim().toLowerCase();
    if (!trimmed) return parentOptions;
    return parentOptions.filter((o) => o.name.toLowerCase().includes(trimmed) || o.id.toLowerCase().includes(trimmed));
  }, [parentOptions, parentSearch]);

  const selectedParent = parentOptions.find((o) => o.id === value.scopePattern);
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
              scopePattern: '',
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

      {scopeType ? (
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
                scopePattern: '',
              })
            }
            layout="vertical"
          >
            <Radio value="all">All {typeLabel.toLowerCase()}s in the workspace</Radio>
            <Radio value="scoped">
              {narrowsByPattern
                ? `Only one ${typeLabel.toLowerCase()}`
                : `Only ${typeLabel.toLowerCase()}s within one ${scopeLabel?.toLowerCase()}`}
            </Radio>
          </Radio.Group>
          {value.scope === 'scoped' && (
            <div css={{ marginTop: theme.spacing.sm }}>
              <FieldLabel>{scopeLabel}</FieldLabel>
              {showScopeRequiredError && (
                <Typography.Text
                  color="error"
                  size="sm"
                  css={{ display: 'block', marginBottom: theme.spacing.xs }}
                  data-testid="admin.mutation_condition_form.parent_required_error"
                >
                  {/* "Select a specific X" keeps the "a" article correct regardless of
                      the container label, the same way the permission forms do it --
                      "a experiment" and "a mcp server" are both reachable here, and
                      experiment is the common case since runs, traces and logged models
                      all scope by one. */}
                  Select a specific {scopeLabel?.toLowerCase()} or switch the scope to{' '}
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
                  value={value.scopePattern}
                  onChange={(e) => onChange({ ...value, scopePattern: e.target.value })}
                  placeholder={`Enter ${scopeLabel?.toLowerCase()} name`}
                  disabled={disabled}
                />
              ) : (
                <DialogCombobox
                  componentId="admin.mutation_condition_form.parent_resource_id"
                  label={scopeLabel ?? 'Scope'}
                  value={value.scopePattern ? [value.scopePattern] : []}
                >
                  <DialogComboboxTrigger
                    withInlineLabel={false}
                    placeholder={`Select ${scopeLabel?.toLowerCase()}`}
                    renderDisplayedValue={() => (selectedParent ? renderOption(selectedParent) : value.scopePattern)}
                    onClear={() => onChange({ ...value, scopePattern: '' })}
                    width="100%"
                    disabled={disabled}
                  />
                  <DialogComboboxContent
                    style={{ zIndex: theme.options.zIndexBase + 100 }}
                    loading={parentOptionsLoading}
                  >
                    {parentOptionsError && (
                      <div css={{ padding: theme.spacing.sm, color: theme.colors.textValidationDanger }}>
                        Failed to load {scopeLabel?.toLowerCase()}s
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
                                onChange({ ...value, scopePattern: v });
                                setParentSearch('');
                              }}
                              checked={option.id === value.scopePattern}
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
        <FieldLabel
          componentId="admin.mutation_condition_form.value_condition_help"
          hintIconTitle="More information about value conditions"
          hint={
            <>
              Constrains the values being set. Use <code>tag_key</code> to limit which keys may be written and{' '}
              <code>tag_value</code> to limit their values &mdash; but note each clause applies to <em>every</em> tag in
              the request. To pin one key to its own values and leave other keys free, name the key:{' '}
              <code>tags.a IN (&#39;x&#39;,&#39;y&#39;)</code>.
            </>
          }
        >
          Value condition
        </FieldLabel>
        <Input
          componentId="admin.mutation_condition_form.value_condition"
          value={value.valueCondition}
          onChange={(e) => onChange({ ...value, valueCondition: e.target.value })}
          placeholder="tag_value != 'prod'"
          disabled={disabled}
        />
        <FieldHint testId="admin.mutation_condition_form.value_condition_hint">
          <strong>A request that does not set the key passes.</strong> {typeLabel} accepts:{' '}
          {renderIdentifiers(getConditionRequestIdentifiers(value.resourceType))}.
        </FieldHint>
        {unsupportedValueIdentifiers.length > 0 && (
          <Typography.Text
            color="error"
            size="sm"
            data-testid="admin.mutation_condition_form.value_condition_vocabulary_error"
          >
            {describeUnsupported(unsupportedValueIdentifiers, typeLabel)}
          </Typography.Text>
        )}
      </div>

      <div>
        <FieldLabel
          componentId="admin.mutation_condition_form.target_condition_help"
          hintIconTitle="More information about target conditions"
          hint={
            <>
              Constrains which existing resources may be mutated, by their current state. Applies to updates and
              deletes; a resource being created has no state yet, so this never blocks a create.
            </>
          }
        >
          Target condition
        </FieldLabel>
        <Input
          componentId="admin.mutation_condition_form.target_condition"
          value={value.targetCondition}
          onChange={(e) => onChange({ ...value, targetCondition: e.target.value })}
          placeholder="tags.lifecycle != 'prod'"
          disabled={disabled}
        />
        <FieldHint testId="admin.mutation_condition_form.target_condition_hint">
          <strong>A resource that does not have the tag is refused</strong> &mdash; the opposite of the value condition
          above, so the same clause means different things in the two fields. {typeLabel} accepts:{' '}
          {renderIdentifiers(getConditionResourceIdentifiers(value.resourceType))}.
        </FieldHint>
        {unsupportedTargetIdentifiers.length > 0 && (
          <Typography.Text
            color="error"
            size="sm"
            data-testid="admin.mutation_condition_form.target_condition_vocabulary_error"
          >
            {describeUnsupported(unsupportedTargetIdentifiers, typeLabel)}
          </Typography.Text>
        )}
      </div>

      {showFilterRequiredError && (
        <Typography.Text color="error" size="sm" data-testid="admin.mutation_condition_form.filter_required_error">
          Enter at least one filter.
        </Typography.Text>
      )}
    </div>
  );
};

/** True when the draft is ready to be submitted. */
export const isMutationConditionDraftFillable = (draft: MutationConditionDraft): boolean => {
  if (isConditionEmpty(draft.valueCondition, draft.targetCondition)) return false;
  if (draft.scope === 'scoped') return draft.scopePattern.trim().length > 0;
  return true;
};

/** True when any field has been touched away from the default. */
export const isMutationConditionDraftDirty = (draft: MutationConditionDraft): boolean =>
  draft.resourceType !== MUTATION_CONDITION_DRAFT_DEFAULT.resourceType ||
  draft.scope !== MUTATION_CONDITION_DRAFT_DEFAULT.scope ||
  draft.scopePattern !== MUTATION_CONDITION_DRAFT_DEFAULT.scopePattern ||
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
  const narrowsByPattern = conditionNarrowsByPattern(draft.resourceType);
  const containerType = getConditionContainerType(draft.resourceType);
  const scopeType = narrowsByPattern ? draft.resourceType : containerType;
  const scoped = draft.scope === 'scoped' && scopeType && draft.scopePattern.trim();
  return {
    resourceType: draft.resourceType,
    // Which axis carries the id is derived from the type, never taken from the draft, so
    // the triple the server receives is always one it can accept.
    resourcePattern: scoped && narrowsByPattern ? draft.scopePattern.trim() : CONDITION_WILDCARD_PATTERN,
    containerResourceType: scoped && !narrowsByPattern && scopeType ? scopeType : CONDITION_CONTAINER_WORKSPACE,
    containerResourcePattern: scoped && !narrowsByPattern ? draft.scopePattern.trim() : CONDITION_WILDCARD_PATTERN,
    valueCondition: draft.valueCondition.trim() || null,
    targetCondition: draft.targetCondition.trim() || null,
  };
};
