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
  CONDITION_TYPES,
  CONDITION_TYPE_VALUE,
  getConditionTypeLabel,
} from '../types';
import type { ConditionType } from '../types';

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

export type ConditionScope = 'all' | 'scoped';

/**
 * Internal draft state of one condition.
 *
 * Distinct from the persisted ``Condition`` in two ways. ``scope`` is tracked
 * separately from ``scopePattern`` so switching to "all" does not lose an id the admin
 * already picked, and the two filters are plain strings here because an empty string is a
 * field the admin has not filled while the server's ``null`` means the filter is absent.
 *
 * One ``scopePattern`` covers both server axes because a type only ever narrows on one of
 * them: a top-level type by its own ``resource_pattern``, a sub-resource by its
 * ``container_resource_pattern``. Which axis applies is derived from the type and never
 * carried here, so the draft cannot describe a scope the server would reject.
 */
export interface ConditionDraft {
  resourceType: string;
  scope: ConditionScope;
  /** Chosen id when ``scope === 'scoped'``; empty otherwise. */
  scopePattern: string;
  /** Which question this condition answers. One condition carries exactly one. */
  type: ConditionType;
  /** The filter. Never submitted empty: a condition IS its filter. */
  filter: string;
}

export const CONDITION_DRAFT_DEFAULT: ConditionDraft = {
  resourceType: CONDITION_RESOURCE_TYPES[0],
  scope: 'all',
  scopePattern: '',
  type: CONDITION_TYPE_VALUE,
  filter: '',
};

export interface ConditionFormProps {
  value: ConditionDraft;
  onChange: (value: ConditionDraft) => void;
  /** The role's workspace, so the parent picker lists resources from the right one. */
  workspace?: string;
  disabled?: boolean;
  /** Render the inline reminder that at least one filter is required. */
  showFilterRequiredError?: boolean;
  /** Render the inline reminder that an id must be picked for ``scope === 'scoped'``. */
  showScopeRequiredError?: boolean;
  /**
   * Identifiers in the filter that the selected resource type and condition type do not
   * accept, computed by the staging section so the same answer gates its Add button -- a
   * form that only *showed* the problem would still let the condition be staged and then
   * refused by the server.
   */
  unsupportedIdentifiers?: string[];
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
 * Add a condition to a role. Shaped like ``RolePermissionForm`` so the two
 * configuration surfaces read the same way.
 */
export const ConditionForm = ({
  value,
  onChange,
  workspace,
  disabled,
  showFilterRequiredError = false,
  showScopeRequiredError = false,
  unsupportedIdentifiers = [],
}: ConditionFormProps) => {
  const { theme } = useDesignSystemTheme();
  const [parentSearch, setParentSearch] = useState('');

  const typeLabel = getResourceTypeLabel(value.resourceType);
  const isValueType = value.type === CONDITION_TYPE_VALUE;
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
          id="admin-condition-form-resource-type"
          componentId="admin.condition_form.resource_type"
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
            componentId="admin.condition_form.scope"
            name="admin-condition-form-scope"
            value={value.scope}
            onChange={(e) =>
              onChange({
                ...value,
                scope: e.target.value as ConditionScope,
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
                  data-testid="admin.condition_form.parent_required_error"
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
                  componentId="admin.condition_form.parent_resource_id_text"
                  value={value.scopePattern}
                  onChange={(e) => onChange({ ...value, scopePattern: e.target.value })}
                  placeholder={`Enter ${scopeLabel?.toLowerCase()} name`}
                  disabled={disabled}
                />
              ) : (
                <DialogCombobox
                  componentId="admin.condition_form.parent_resource_id"
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
              {isPartialWildcardPattern(value.scopePattern) ? (
                <Typography.Text
                  color="error"
                  size="sm"
                  data-testid="admin.condition_form.scope_pattern_wildcard_error"
                >
                  A scope is matched exactly, so &lsquo;{value.scopePattern.trim()}&rsquo; would restrict nothing. Enter
                  one {scopeLabel?.toLowerCase()} name, or switch the scope to{' '}
                  <strong>All {typeLabel.toLowerCase()}s in the workspace</strong>.
                </Typography.Text>
              ) : null}
            </div>
          )}
        </div>
      ) : null}

      <div>
        <FieldLabel
          componentId="admin.condition_form.type_help"
          hintIconTitle="More information about condition types"
          hint={
            <>
              A condition answers one of two questions, and the same clause means different things in each.{' '}
              <strong>Values a request may set</strong> constrains what is being written; a request that does not set
              the key <em>passes</em>. <strong>Resources that may be mutated</strong> constrains current state; a
              resource that does not have the tag is <em>refused</em>. To restrict both, add two conditions &mdash; the
              two types apply together.
            </>
          }
        >
          Condition type
        </FieldLabel>
        <SimpleSelect
          componentId="admin.condition_form.type"
          id="admin.condition_form.type"
          aria-label="Condition type"
          value={value.type}
          onChange={(e) => onChange({ ...value, type: e.target.value as ConditionType, filter: '' })}
          disabled={disabled}
        >
          {CONDITION_TYPES.map((conditionType) => (
            <SimpleSelectOption key={conditionType} value={conditionType}>
              {getConditionTypeLabel(conditionType)}
            </SimpleSelectOption>
          ))}
        </SimpleSelect>
      </div>

      <div>
        <FieldLabel
          componentId="admin.condition_form.filter_help"
          hintIconTitle="More information about condition filters"
          hint={
            isValueType ? (
              <>
                Constrains the values being set. Use <code>tag_key</code> to limit which keys may be written and{' '}
                <code>tag_value</code> to limit their values &mdash; but note each clause applies to <em>every</em> tag
                in the request. To pin one key to its own values and leave other keys free, name the key:{' '}
                <code>tags.a IN (&#39;x&#39;,&#39;y&#39;)</code>.
              </>
            ) : (
              <>
                Constrains which existing resources may be mutated, by their current state. Applies to updates and
                deletes; a resource being created has no state yet, so this never blocks a create.
              </>
            )
          }
        >
          Filter
        </FieldLabel>
        <Input
          componentId="admin.condition_form.filter"
          value={value.filter}
          onChange={(e) => onChange({ ...value, filter: e.target.value })}
          placeholder={isValueType ? "tag_value != 'prod'" : "tags.lifecycle != 'prod'"}
          disabled={disabled}
        />
        <FieldHint testId="admin.condition_form.filter_hint">
          {isValueType ? (
            <>
              <strong>A request that does not set the key passes.</strong>
            </>
          ) : (
            <>
              <strong>A resource that does not have the tag is refused</strong> &mdash; the opposite of the value type,
              so the same clause means different things in the two.
            </>
          )}{' '}
          {typeLabel} accepts:{' '}
          {renderIdentifiers(
            isValueType
              ? getConditionRequestIdentifiers(value.resourceType)
              : getConditionResourceIdentifiers(value.resourceType),
          )}
          .
        </FieldHint>
        {unsupportedIdentifiers.length > 0 && (
          <Typography.Text color="error" size="sm" data-testid="admin.condition_form.filter_vocabulary_error">
            {describeUnsupported(unsupportedIdentifiers, typeLabel)}
          </Typography.Text>
        )}
      </div>

      {showFilterRequiredError && (
        <Typography.Text color="error" size="sm" data-testid="admin.condition_form.filter_required_error">
          Enter at least one filter.
        </Typography.Text>
      )}
    </div>
  );
};

/** True when the draft is ready to be submitted. */
/**
 * A scope pattern is matched EXACTLY server-side, so a glob restricts nothing.
 *
 * The server refuses this, but the typed-id branch above is where an admin reaches for one
 * -- an ``mcp_server`` name is ``<namespace>/<slug>``, which makes ``team/*`` the obvious
 * thing to write. Caught here so the reason is visible next to the field rather than
 * arriving as a rejected submit. The bare wildcard is not a partial one: it is what the
 * ``All ...`` scope sends.
 */
export const isPartialWildcardPattern = (pattern: string): boolean => {
  const p = pattern.trim();
  return p.includes(CONDITION_WILDCARD_PATTERN) && p !== CONDITION_WILDCARD_PATTERN;
};

export const isConditionDraftFillable = (draft: ConditionDraft): boolean => {
  if (isConditionEmpty(draft.filter)) return false;
  if (draft.scope === 'scoped') {
    return draft.scopePattern.trim().length > 0 && !isPartialWildcardPattern(draft.scopePattern);
  }
  return true;
};

/** True when any field has been touched away from the default. */
export const isConditionDraftDirty = (draft: ConditionDraft): boolean =>
  draft.resourceType !== CONDITION_DRAFT_DEFAULT.resourceType ||
  draft.scope !== CONDITION_DRAFT_DEFAULT.scope ||
  draft.scopePattern !== CONDITION_DRAFT_DEFAULT.scopePattern ||
  draft.type !== CONDITION_DRAFT_DEFAULT.type ||
  draft.filter !== CONDITION_DRAFT_DEFAULT.filter;

/**
 * Translate a draft into the condition a staged list holds.
 *
 * The single place this mapping lives. It used to be duplicated in the staging
 * section, which meant the tested copy and the copy actually used could drift -- and
 * drift here produces a malformed write.
 *
 * Both parent fields travel together or neither does (the server enforces the pair with
 * a CHECK constraint), and the parent type is derived from the resource type rather than
 * carried in the draft.
 *
 * The filter is sent as typed, never normalised to null: there is no absent-filter state
 * to represent any more, and a blank one is refused by the server rather than stored.
 * ``isConditionDraftFillable`` is what keeps a blank from being submitted at all.
 */
export const draftToStagedCondition = (draft: ConditionDraft) => {
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
    type: draft.type,
    filter: draft.filter.trim(),
  };
};
