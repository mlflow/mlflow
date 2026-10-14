import {
  useCallback,
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
  useState,
  type ReactNode,
  type Ref,
} from 'react';

import {
  Button,
  FilterIcon,
  FormUI,
  Input,
  PlusIcon,
  Popover,
  SimpleSelect,
  SimpleSelectOption,
  SimpleSelectOptionGroup,
  Tooltip,
  TypeaheadComboboxInput,
  TypeaheadComboboxMenu,
  TypeaheadComboboxMenuItem,
  TypeaheadComboboxRoot,
  TypeaheadComboboxStateChangeTypes,
  TrashIcon,
  useDesignSystemTheme,
  XCircleFillIcon,
  useComboboxState,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

import type { ExperimentRunsSelectorResult } from '../../components/experiment-page/utils/experimentRuns.selector';
import { EVAL_RUNS_TOOLBAR_ICON_SIZE } from './ExperimentEvaluationRunsTable.constants';
import type { EvalRunsFilterClause, EvalRunsFilterFieldDef } from './evalRunsFilterFields';
import {
  EVAL_RUNS_FILTER_OPERATOR_LABELS,
  EvalRunsFilterOperator,
  buildClausesFilterString,
  getEvalRunsFilterFields,
  getEvalRunsFilterField,
  getEvalRunsFilterFieldValues,
  getOperatorsForFieldType,
} from './evalRunsFilterFields';

const COMPONENT_ID = 'mlflow.eval-runs.filter-button';
// Keep intermediate states such as `-`, `1.`, or `1e+` typeable; final numeric validity is
// enforced by the SearchRuns clause compiler before applying the filter.
const NUMERIC_FILTER_INPUT_PATTERN = /^-?(?:(?:\d+(?:\.\d*)?|\.\d*)(?:[eE][+-]?\d*)?)?$/;

const EMPTY_CLAUSE: EvalRunsFilterClause = { field: '', operator: EvalRunsFilterOperator.EQUALS, value: '' };

export interface EvalRunsFilterButtonHandle {
  openForField: (fieldId: string) => void;
}

export interface EvalRunsFilterButtonProps {
  clauses: EvalRunsFilterClause[];
  setClauses: (clauses: EvalRunsFilterClause[]) => void;
  /** Loaded-runs metadata; the filterable metric/param/tag keys are derived from it. */
  runsData: ExperimentRunsSelectorResult;
  excludeIssueDetectionRuns?: boolean;
  filterButtonRef?: Ref<EvalRunsFilterButtonHandle>;
  renderTrigger?: (openEditor: (filterIndex?: number) => void) => ReactNode;
}

// Builds group list in first-seen order.
const groupFields = (fields: EvalRunsFilterFieldDef[]): { group: string; fields: EvalRunsFilterFieldDef[] }[] => {
  const groups: { group: string; fields: EvalRunsFilterFieldDef[] }[] = [];
  fields.forEach((field) => {
    const existing = groups.find((entry) => entry.group === field.group);
    if (existing) {
      existing.fields.push(field);
    } else {
      groups.push({ group: field.group, fields: [field] });
    }
  });
  return groups;
};

const EvalRunsFilterValueTypeahead = ({
  clause,
  index,
  field,
  fieldValues,
  onChange,
}: {
  clause: EvalRunsFilterClause;
  index: number;
  field: EvalRunsFilterFieldDef;
  fieldValues: string[];
  onChange: (clause: EvalRunsFilterClause) => void;
}) => {
  const intl = useIntl();
  const comboboxStateRef = useRef<{ setInputValue: (value: string) => void } | null>(null);
  const [items, setItems] = useState<string[]>(fieldValues);
  const valueLabel = intl.formatMessage({
    defaultMessage: 'Filter value',
    description: 'Aria label for the value input in the eval-runs filter',
  });

  const comboboxState = useComboboxState<string>({
    componentId: `${COMPONENT_ID}.value.${index}`,
    allItems: fieldValues,
    items,
    setItems,
    itemToString: (item) => item ?? '',
    matcher: (item, query) => item.toLocaleLowerCase().includes(query.toLocaleLowerCase()),
    initialInputValue: clause.value,
    allowNewValue: true,
    formOnChange: (value: string | null) => {
      if (value !== null && value !== undefined) onChange({ ...clause, value });
    },
    onStateChange: (change) => {
      if (change.type !== TypeaheadComboboxStateChangeTypes.InputChange || change.inputValue === undefined) return;

      if (field.type === 'number' && !NUMERIC_FILTER_INPUT_PATTERN.test(change.inputValue)) {
        // The combobox owns its input state. Restore the last accepted value when a paste or
        // another non-keyboard edit contains an invalid numeric value.
        comboboxStateRef.current?.setInputValue(clause.value);
        return;
      }
      onChange({ ...clause, value: change.inputValue });
    },
  });
  comboboxStateRef.current = comboboxState;

  return (
    <div css={{ width: 160 }}>
      <TypeaheadComboboxRoot comboboxState={comboboxState}>
        <TypeaheadComboboxInput
          comboboxState={comboboxState}
          id={`eval-runs-filter-value-${index}`}
          aria-label={valueLabel}
          type="text"
          inputMode={field.type === 'number' ? 'decimal' : undefined}
          showComboboxToggleButton={fieldValues.length > 0}
        />
        {items.length > 0 && (
          <TypeaheadComboboxMenu comboboxState={comboboxState} width={440} minWidth={0} maxWidth="calc(100vw - 32px)">
            {items.map((value, itemIndex) => (
              <TypeaheadComboboxMenuItem
                key={value}
                item={value}
                index={itemIndex}
                comboboxState={comboboxState}
                textOverflowMode="multiline"
              >
                {value}
              </TypeaheadComboboxMenuItem>
            ))}
          </TypeaheadComboboxMenu>
        )}
      </TypeaheadComboboxRoot>
    </div>
  );
};

/**
 * Field / operator / value row. Selecting a new field resets the operator and clears the value,
 * since both depend on field type.
 */
const EvalRunsFilterClauseRow = ({
  clause,
  index,
  fields,
  runsData,
  onChange,
  onRemove,
}: {
  clause: EvalRunsFilterClause;
  index: number;
  fields: EvalRunsFilterFieldDef[];
  runsData: ExperimentRunsSelectorResult;
  onChange: (clause: EvalRunsFilterClause) => void;
  onRemove: () => void;
}) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();

  const fieldGroups = useMemo(() => groupFields(fields), [fields]);
  const selectedField = fields.find((field) => field.id === clause.field);
  const selectedFieldId = selectedField?.id;
  const selectedFieldType = selectedField?.type;
  const operators = selectedField ? getOperatorsForFieldType(selectedField.type) : [];
  const fieldValues = useMemo(
    () =>
      selectedFieldId && selectedFieldType !== 'enum' ? getEvalRunsFilterFieldValues(selectedFieldId, runsData) : [],
    [runsData, selectedFieldId, selectedFieldType],
  );
  // Lift SimpleSelect menus above the Popover so they aren't clipped behind it.
  const selectContentProps = { style: { zIndex: theme.options.zIndexBase + 100 } };

  return (
    <div
      css={{
        display: 'flex',
        alignItems: 'flex-end',
        gap: theme.spacing.sm,
        '& [role="combobox"]:focus:not(:focus-visible)': { outline: 'none !important' },
      }}
    >
      <div css={{ display: 'flex', flexDirection: 'column' }}>
        <FormUI.Label htmlFor={`eval-runs-filter-field-${index}`}>
          <FormattedMessage defaultMessage="Field" description="Label for the field selector in the eval-runs filter" />
        </FormUI.Label>
        <SimpleSelect
          componentId={`${COMPONENT_ID}.field`}
          id={`eval-runs-filter-field-${index}`}
          aria-label={intl.formatMessage({
            defaultMessage: 'Filter field',
            description: 'Aria label for the field selector in the eval-runs filter',
          })}
          placeholder={intl.formatMessage({
            defaultMessage: 'Select field',
            description: 'Placeholder for the field selector in the eval-runs filter',
          })}
          width={180}
          contentProps={selectContentProps}
          value={clause.field}
          onChange={(event) => {
            const nextField = fields.find((field) => field.id === event.target.value);
            const nextOperator = nextField
              ? getOperatorsForFieldType(nextField.type)[0]
              : EvalRunsFilterOperator.EQUALS;
            onChange({ field: event.target.value, operator: nextOperator, value: '' });
          }}
        >
          {fieldGroups.map((group) => (
            <SimpleSelectOptionGroup key={group.group} label={group.group}>
              {group.fields.map((field) => (
                <SimpleSelectOption key={field.id} value={field.id}>
                  {field.label}
                </SimpleSelectOption>
              ))}
            </SimpleSelectOptionGroup>
          ))}
        </SimpleSelect>
      </div>

      <div css={{ display: 'flex', flexDirection: 'column' }}>
        <FormUI.Label htmlFor={`eval-runs-filter-operator-${index}`}>
          <FormattedMessage
            defaultMessage="Operator"
            description="Label for the operator selector in the eval-runs filter"
          />
        </FormUI.Label>
        <SimpleSelect
          componentId={`${COMPONENT_ID}.operator`}
          id={`eval-runs-filter-operator-${index}`}
          aria-label={intl.formatMessage({
            defaultMessage: 'Filter operator',
            description: 'Aria label for the operator selector in the eval-runs filter',
          })}
          placeholder="="
          width={100}
          disabled={!selectedField}
          contentProps={selectContentProps}
          value={clause.operator}
          onChange={(event) => onChange({ ...clause, operator: event.target.value as EvalRunsFilterOperator })}
        >
          {operators.map((operator) => (
            <SimpleSelectOption key={operator} value={operator}>
              {EVAL_RUNS_FILTER_OPERATOR_LABELS[operator]}
            </SimpleSelectOption>
          ))}
        </SimpleSelect>
      </div>

      <div css={{ display: 'flex', flexDirection: 'column' }}>
        <FormUI.Label htmlFor={`eval-runs-filter-value-${index}`}>
          <FormattedMessage defaultMessage="Value" description="Label for the value input in the eval-runs filter" />
        </FormUI.Label>
        {selectedField?.type === 'enum' ? (
          <SimpleSelect
            componentId={`${COMPONENT_ID}.value`}
            id={`eval-runs-filter-value-${index}`}
            aria-label={intl.formatMessage({
              defaultMessage: 'Filter value',
              description: 'Aria label for the value selector in the eval-runs filter',
            })}
            placeholder={intl.formatMessage({
              defaultMessage: 'Select value',
              description: 'Placeholder for the value selector in the eval-runs filter',
            })}
            width={160}
            contentProps={selectContentProps}
            value={clause.value}
            onChange={(event) => onChange({ ...clause, value: event.target.value })}
          >
            {selectedField.options?.map((option) => (
              <SimpleSelectOption key={option.value} value={option.value}>
                {option.label}
              </SimpleSelectOption>
            ))}
          </SimpleSelect>
        ) : selectedField ? (
          <EvalRunsFilterValueTypeahead
            key={selectedField.id}
            clause={clause}
            index={index}
            field={selectedField}
            fieldValues={fieldValues}
            onChange={onChange}
          />
        ) : (
          <Input
            componentId={`${COMPONENT_ID}.value`}
            id={`eval-runs-filter-value-${index}`}
            aria-label={intl.formatMessage({
              defaultMessage: 'Filter value',
              description: 'Aria label for the value input in the eval-runs filter',
            })}
            css={{ width: 160 }}
            type="text"
            disabled={!selectedField}
            value={clause.value}
            onChange={(event) => onChange({ ...clause, value: event.target.value })}
          />
        )}
      </div>

      <Button
        componentId={`${COMPONENT_ID}.remove`}
        type="tertiary"
        icon={<TrashIcon />}
        aria-label={intl.formatMessage({
          defaultMessage: 'Remove filter',
          description: 'Aria label for the remove-filter button in the eval-runs filter',
        })}
        onClick={onRemove}
      />
    </div>
  );
};

/**
 * Filters funnel — multi-clause filter popover. Edits are staged in a draft and committed on Apply.
 * Shows active count + clear affordance when filters are set; icon-only with tooltip when inactive.
 */
export const EvalRunsFilterButton = ({
  clauses,
  setClauses,
  runsData,
  excludeIssueDetectionRuns = false,
  filterButtonRef,
  renderTrigger,
}: EvalRunsFilterButtonProps) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();

  const [open, setOpen] = useState(false);
  const [draftClauses, setDraftClauses] = useState<EvalRunsFilterClause[]>(clauses);
  const fields = useMemo(() => {
    const available = getEvalRunsFilterFields(runsData, excludeIssueDetectionRuns);
    for (const clause of [...clauses, ...draftClauses]) {
      if (!available.some((field) => field.id === clause.field)) {
        const field = getEvalRunsFilterField(clause.field, excludeIssueDetectionRuns);
        if (field) available.push(field);
      }
    }
    return available;
  }, [runsData, clauses, draftClauses, excludeIssueDetectionRuns]);

  const activeCount = clauses.length;
  const isActive = activeCount > 0;
  const filtersLabel = intl.formatMessage({
    defaultMessage: 'Filters',
    description: 'Aria label / tooltip for the eval-runs filter button',
  });

  // Reseed draft from committed clauses on open; cancelling without Apply discards changes.
  // Always show at least one row so the pickers are visible.
  const seedDraft = useCallback(() => {
    setDraftClauses(clauses.length > 0 ? clauses : [EMPTY_CLAUSE]);
  }, [clauses]);

  const openTimerRef = useRef<ReturnType<typeof setTimeout>>();
  const editorContentRef = useRef<HTMLDivElement>(null);
  useEffect(() => () => clearTimeout(openTimerRef.current), []);
  const openForField = useCallback(
    (fieldId: string) => {
      const selectedField =
        fields.find((field) => field.id === fieldId) ?? getEvalRunsFilterField(fieldId, excludeIssueDetectionRuns);
      if (!selectedField) return;
      const preserveDraft = open || openTimerRef.current !== undefined;
      setDraftClauses((current) => [
        ...(preserveDraft ? current : clauses),
        { field: selectedField.id, operator: getOperatorsForFieldType(selectedField.type)[0], value: '' },
      ]);
      clearTimeout(openTimerRef.current);
      // Wait for the sibling header menu's Radix dismissal before opening this popover.
      openTimerRef.current = setTimeout(() => {
        openTimerRef.current = undefined;
        setOpen(true);
      });
    },
    [clauses, excludeIssueDetectionRuns, fields, open],
  );
  useImperativeHandle(filterButtonRef, () => ({ openForField }), [openForField]);

  const handleOpenChange = useCallback(
    (nextOpen: boolean) => {
      if (nextOpen) {
        seedDraft();
      }
      setOpen(nextOpen);
    },
    [seedDraft],
  );

  const hasInvalidValue = draftClauses.some(
    (clause) => clause.field && clause.value.trim() && !buildClausesFilterString([clause]),
  );
  const applyFilters = useCallback(() => {
    if (hasInvalidValue) return;
    // Drop incomplete rows so a half-filled clause never reaches the query builder.
    setClauses(draftClauses.filter((clause) => clause.field && clause.value.trim()));
    setOpen(false);
  }, [draftClauses, hasInvalidValue, setClauses]);

  const clearFilters = useCallback(() => {
    setClauses([]);
  }, [setClauses]);
  const openEditor = useCallback(() => {
    seedDraft();
    clearTimeout(openTimerRef.current);
    // The unified search dropdown is portaled outside this Popover. Wait for its outside-click
    // dismissal to finish before opening the editor, or Radix closes the editor immediately.
    openTimerRef.current = setTimeout(() => {
      openTimerRef.current = undefined;
      setOpen(true);
    });
  }, [seedDraft]);

  return (
    <Popover.Root componentId={`${COMPONENT_ID}.popover`} open={open} onOpenChange={handleOpenChange}>
      {renderTrigger ? (
        <Popover.Anchor asChild>
          <div css={{ width: '100%' }}>{renderTrigger(openEditor)}</div>
        </Popover.Anchor>
      ) : isActive ? (
        <Popover.Trigger asChild>
          <Button
            componentId={`${COMPONENT_ID}.trigger`}
            icon={<FilterIcon css={{ fontSize: EVAL_RUNS_TOOLBAR_ICON_SIZE, color: theme.colors.textPlaceholder }} />}
            endIcon={
              <XCircleFillIcon
                css={{ color: theme.colors.textSecondary }}
                onClick={(event) => {
                  // Clear without opening the popover.
                  event.stopPropagation();
                  clearFilters();
                }}
              />
            }
            // `&&&` + !important beats dbui's importantified padding.
            css={{
              flexShrink: 0,
              '&&&': {
                padding: `${theme.spacing.xs}px ${theme.spacing.sm}px !important`,
              },
            }}
          >
            {intl.formatMessage(
              { defaultMessage: 'Filters ({count})', description: 'Eval-runs filter button with active count' },
              { count: activeCount },
            )}
          </Button>
        </Popover.Trigger>
      ) : (
        <Tooltip componentId={`${COMPONENT_ID}.trigger.tooltip`} content={filtersLabel}>
          <Popover.Trigger asChild>
            {/* Icon as child (not `icon` prop) so Button keeps its grey border — icon-only renders borderless. */}
            <Button
              componentId={`${COMPONENT_ID}.trigger`}
              aria-label={filtersLabel}
              // `&&&` + !important beats dbui's importantified padding.
              css={{
                flexShrink: 0,
                '&&&': {
                  padding: `${theme.spacing.xs}px ${theme.spacing.sm}px !important`,
                },
              }}
            >
              <FilterIcon css={{ fontSize: EVAL_RUNS_TOOLBAR_ICON_SIZE, color: theme.colors.textPlaceholder }} />
            </Button>
          </Popover.Trigger>
        </Tooltip>
      )}
      <Popover.Content
        ref={editorContentRef}
        align="start"
        tabIndex={-1}
        onOpenAutoFocus={(event) => {
          event.preventDefault();
          editorContentRef.current?.querySelector<HTMLElement>('input, [role="combobox"]')?.focus();
        }}
        css={{ minWidth: 360, padding: theme.spacing.md }}
      >
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
            {draftClauses.map((clause, index) => (
              <EvalRunsFilterClauseRow
                key={index}
                clause={clause}
                index={index}
                fields={fields}
                runsData={runsData}
                onChange={(nextClause) =>
                  setDraftClauses((prev) => prev.map((existing, i) => (i === index ? nextClause : existing)))
                }
                onRemove={() =>
                  setDraftClauses((prev) => {
                    const next = prev.filter((_, i) => i !== index);
                    return next.length > 0 ? next : [EMPTY_CLAUSE];
                  })
                }
              />
            ))}
          </div>

          <div css={{ display: 'flex', justifyContent: 'space-between' }}>
            <Button
              componentId={`${COMPONENT_ID}.add`}
              icon={<PlusIcon />}
              onClick={() => setDraftClauses((prev) => [...prev, EMPTY_CLAUSE])}
            >
              <FormattedMessage
                defaultMessage="Add filter"
                description="Button to add a filter row in the eval-runs filter"
              />
            </Button>
            <Button
              componentId={`${COMPONENT_ID}.apply`}
              type="primary"
              onClick={applyFilters}
              disabled={hasInvalidValue}
            >
              <FormattedMessage
                defaultMessage="Apply"
                description="Button to apply the filters in the eval-runs filter"
              />
            </Button>
          </div>
        </div>
      </Popover.Content>
    </Popover.Root>
  );
};
