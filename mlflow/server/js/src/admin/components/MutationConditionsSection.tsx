import { useEffect, useRef, useState } from 'react';
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
import { CONDITION_CONTAINER_WORKSPACE, CONDITION_WILDCARD_PATTERN, getResourceTypeLabel } from '../types';
import {
  draftToStagedCondition,
  isMutationConditionDraftDirty,
  isMutationConditionDraftFillable,
  MUTATION_CONDITION_DRAFT_DEFAULT,
  MutationConditionForm,
  type MutationConditionDraft,
} from './MutationConditionForm';

/**
 * One staged mutation condition. ``id`` is set when the row was pre-loaded from the
 * role's current conditions, so the parent modal can call ``removeMutationCondition(id)``
 * on submit. Newly-staged rows have ``id === undefined`` and the diff treats those as adds.
 *
 * The three scope fields mirror the wire exactly -- never null, defaulting to the
 * wildcard and the workspace -- so a staged row and a stored one compare field for field.
 */
export interface StagedMutationCondition {
  id?: number;
  resourceType: string;
  resourcePattern: string;
  containerResourceType: string;
  containerResourcePattern: string;
  valueCondition: string | null;
  targetCondition: string | null;
}

export interface MutationConditionsSectionProps {
  value: StagedMutationCondition[];
  onChange: (value: StagedMutationCondition[]) => void;
  /** The role's workspace, so the scope picker lists resources from the right one. */
  workspace?: string;
  disabled?: boolean;
  onUnsavedDraftChange?: (hasUnsavedDraft: boolean) => void;
}

/**
 * Stable identity for dedup: the whole tuple, since any field changing makes a new condition.
 *
 * Exported and shared, because the two edit modals each kept their own copy and both had
 * drifted -- they omitted `resourcePattern`, so two conditions differing only in scope
 * collided and a removal of either was silently dropped from the diff.
 */
export const conditionKey = (c: StagedMutationCondition) =>
  [
    c.resourceType,
    c.resourcePattern,
    c.containerResourceType,
    c.containerResourcePattern,
    c.valueCondition ?? '',
    c.targetCondition ?? '',
  ].join('::');

export const formatStagedScope = (c: StagedMutationCondition): string => {
  // Only one axis is ever narrowed, so whichever is set is the whole scope.
  if (c.resourcePattern && c.resourcePattern !== CONDITION_WILDCARD_PATTERN) {
    return `${getResourceTypeLabel(c.resourceType)} ${c.resourcePattern}`;
  }
  if (c.containerResourceType && c.containerResourceType !== CONDITION_CONTAINER_WORKSPACE) {
    return `${getResourceTypeLabel(c.containerResourceType)} ${c.containerResourcePattern}`;
  }
  return 'All in workspace';
};

/** Human one-liner for the review step. */
export const formatStagedCondition = (c: StagedMutationCondition): string => {
  const filters = [
    c.valueCondition && `value: ${c.valueCondition}`,
    c.targetCondition && `target: ${c.targetCondition}`,
  ]
    .filter(Boolean)
    .join(', ');
  return `${c.resourceType} [${formatStagedScope(c)}] ${filters}`;
};

/**
 * Wraps ``MutationConditionForm`` with the same staged-list pattern the permissions
 * sections use: each Add appends a row, rows are removed individually, and the enclosing
 * modal
 * submits the whole list as a diff.
 */
export const MutationConditionsSection = ({
  value,
  onChange,
  workspace,
  disabled,
  onUnsavedDraftChange,
}: MutationConditionsSectionProps) => {
  const { theme } = useDesignSystemTheme();
  const [draft, setDraft] = useState<MutationConditionDraft>(MUTATION_CONDITION_DRAFT_DEFAULT);

  const canAdd = isMutationConditionDraftFillable(draft);
  const dirty = isMutationConditionDraftDirty(draft);
  // Narrow each reminder to the field actually missing, rather than just refusing.
  const showFilterRequired = dirty && !draft.valueCondition.trim() && !draft.targetCondition.trim();
  const showScopeRequired = dirty && draft.scope === 'scoped' && !draft.scopePattern.trim();

  const onUnsavedDraftChangeRef = useRef(onUnsavedDraftChange);
  useEffect(() => {
    onUnsavedDraftChangeRef.current = onUnsavedDraftChange;
  }, [onUnsavedDraftChange]);
  useEffect(() => {
    onUnsavedDraftChangeRef.current?.(dirty);
  }, [dirty]);

  const handleAdd = () => {
    if (!canAdd) return;
    const staged: StagedMutationCondition = draftToStagedCondition(draft);
    if (value.some((c) => conditionKey(c) === conditionKey(staged))) {
      setDraft(MUTATION_CONDITION_DRAFT_DEFAULT);
      return;
    }
    onChange([...value, staged]);
    setDraft(MUTATION_CONDITION_DRAFT_DEFAULT);
  };

  const handleRemove = (index: number) => {
    onChange(value.filter((_, i) => i !== index));
  };

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      {value.length > 0 && (
        <Table scrollable noMinHeight css={{ border: `1px solid ${theme.colors.border}` }}>
          <TableRow isHeader>
            <TableHeader componentId="admin.role_conditions.staged_type" css={{ flex: 1 }}>
              Resource type
            </TableHeader>
            <TableHeader componentId="admin.role_conditions.staged_scope" css={{ flex: 1 }}>
              Scope
            </TableHeader>
            <TableHeader componentId="admin.role_conditions.staged_request" css={{ flex: 2 }}>
              Value condition
            </TableHeader>
            <TableHeader componentId="admin.role_conditions.staged_resource" css={{ flex: 2 }}>
              Target condition
            </TableHeader>
            <TableHeader
              componentId="admin.role_conditions.staged_actions"
              css={{ flex: 0, minWidth: 60, maxWidth: 60 }}
            >
              {' '}
            </TableHeader>
          </TableRow>
          {value.map((c, i) => (
            <TableRow key={`${conditionKey(c)}-${i}`}>
              <TableCell css={{ flex: 1 }}>
                <Tag componentId="admin.role_conditions.staged_type_tag">{getResourceTypeLabel(c.resourceType)}</Tag>
              </TableCell>
              <TableCell css={{ flex: 1 }}>
                <Typography.Text size="sm">{formatStagedScope(c)}</Typography.Text>
              </TableCell>
              <TableCell css={{ flex: 2 }}>
                {c.valueCondition ? (
                  <code>{c.valueCondition}</code>
                ) : (
                  <Typography.Text color="secondary" size="sm">
                    —
                  </Typography.Text>
                )}
              </TableCell>
              <TableCell css={{ flex: 2 }}>
                {c.targetCondition ? (
                  <code>{c.targetCondition}</code>
                ) : (
                  <Typography.Text color="secondary" size="sm">
                    —
                  </Typography.Text>
                )}
              </TableCell>
              <TableCell css={{ flex: 0, minWidth: 60, maxWidth: 60 }}>
                <Button
                  componentId="admin.role_conditions.staged_remove"
                  type="tertiary"
                  size="small"
                  icon={<CloseIcon />}
                  aria-label={`Remove ${getResourceTypeLabel(c.resourceType)} mutation condition`}
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
        <FieldLabel>Add a mutation condition</FieldLabel>
        <MutationConditionForm
          value={draft}
          onChange={setDraft}
          workspace={workspace}
          disabled={disabled}
          showFilterRequiredError={showFilterRequired}
          showScopeRequiredError={showScopeRequired}
        />
        <div css={{ display: 'flex', justifyContent: 'flex-end', gap: theme.spacing.sm }}>
          {dirty && (
            <Button
              componentId="admin.role_conditions.clear"
              type="tertiary"
              onClick={() => setDraft(MUTATION_CONDITION_DRAFT_DEFAULT)}
              disabled={disabled}
            >
              Clear
            </Button>
          )}
          {/* Named, not a bare "Add": this section sits in the same modal as the
              permissions section, and two buttons with the same accessible name give a
              screen-reader user no way to tell them apart. */}
          <Button componentId="admin.role_conditions.add" onClick={handleAdd} disabled={!canAdd || disabled}>
            Add mutation condition
          </Button>
        </div>
      </div>
    </div>
  );
};
