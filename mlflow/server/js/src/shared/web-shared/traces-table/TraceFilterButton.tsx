import { useEffect, useRef, useState } from 'react';
import {
  Button,
  ChevronDownIcon,
  CloseSmallIcon,
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxOptionList,
  DialogComboboxOptionListSearch,
  DialogComboboxOptionListSelectItem,
  DialogComboboxTrigger,
  FilterIcon,
  FormUI,
  Input,
  PlusIcon,
  Popover,
  SimpleSelect,
  SimpleSelectOption,
  Tabs,
  Tooltip,
  Typography,
  XCircleFillIcon,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import {
  type FilterClause,
  type FilterFieldDef,
  type FilterFieldSelectOption,
  type FilterOp,
  type TraceFilterModel,
  isClauseComplete,
  makeEmptyClause,
} from './filterModel';
import { ToolbarCollapsibleLabel } from './TracesToolbarResponsive';

// Module-local static analytics-id namespace (static componentId lint rule).
const COMPONENT_ID = 'web-shared.traces-table';

export interface TraceFilterButtonProps {
  /** The filterable fields, in dropdown order — drives every row (no field names are hardcoded). */
  fields: FilterFieldDef[];
  filterModel: TraceFilterModel;
  onChange: (next: TraceFilterModel) => void;
  /**
   * Clear *every* active filter. `onChange([])` only resets the popover clauses; a consumer with an
   * additional filter source (e.g. URL tag filters counted in `activeCount`) uses this to reset both.
   */
  onClearAll: () => void;
  activeCount: number;
  /**
   * Replaces the default trigger with a consumer-rendered anchor, such as a search box that shows the
   * applied clauses as pills. `openEditor()` opens the editor with a new blank clause after the applied
   * ones; `openEditor(index)` opens it on that one applied clause.
   */
  renderTrigger?: (openEditor: (filterIndex?: number | 'sql') => void) => React.ReactNode;
  /** Explains the consumer's matching semantics above the clauses. */
  description?: React.ReactNode;
  /** Consumer validation runs before Apply; rejected drafts remain editable. */
  validateFilters?: (model: TraceFilterModel) => string | undefined;
  /** Optional search-predicate editor; the consumer owns the DSL and applied state. */
  sqlFilter?: {
    value: string;
    onChange: (value: string) => void;
    appliedClause?: { value: string; onChange: (value: string) => void };
    replaceClause?: (index: number, value: string) => string;
    formatBasic: (model: TraceFilterModel) => string;
    formatBasicDraft?: (model: TraceFilterModel) => string;
    parseBasic: (value: string) => TraceFilterModel | undefined;
    renderEditor?: (value: string, onChange: (value: string) => void) => React.ReactNode;
    error?: Error;
    isFetching?: boolean;
    preferSql?: boolean;
  };
}

interface FilterClauseRowProps {
  clause: FilterClause;
  index: number;
  fields: FilterFieldDef[];
  onChangeClause: (index: number, next: FilterClause) => void;
  onDelete: (index: number) => void;
}

export interface FilterComboboxProps {
  id: string;
  value: string;
  options: FilterFieldSelectOption[];
  placeholder: string;
  ariaLabel: string;
  onChange: (value: string) => void;
  /** z-index for the portaled option list (lifted above the enclosing Popover.Content). */
  zIndex: number;
  /**
   * Clear the typed search whenever the list opens and show the selected option's label in the
   * trigger. Value comboboxes set this; key comboboxes keep their previous behaviour.
   */
  resetOnOpen?: boolean;
}

/**
 * Freeform-capable filter combobox: suggests `options` and offers a `Use "<typed>"` item so an
 * arbitrary key or value can still be entered (the v3 assessment-filter typeahead pattern).
 * Module-scope (not nested) per the repo nested-component rule; the selected option stays controlled
 * by the parent via `value`/`onChange` while the search is local to this input.
 */
const FilterCombobox = ({
  id,
  value,
  options,
  placeholder,
  ariaLabel,
  onChange,
  zIndex,
  resetOnOpen,
}: FilterComboboxProps) => {
  const intl = useIntl();
  const [searchValue, setSearchValue] = useState('');
  const trimmedSearch = searchValue.trim();
  const showCustomValue = trimmedSearch.length > 0 && !options.some((option) => option.value === searchValue);

  const resetSearchOnOpen = (isOpen: boolean) => {
    if (isOpen) {
      setSearchValue('');
    }
  };

  const renderDisplayedValue = (selectedValue: string) =>
    options.find((option) => option.value === selectedValue)?.label ?? selectedValue;

  return (
    <DialogCombobox
      componentId={`${COMPONENT_ID}.filter.key-combobox`}
      id={id}
      value={value ? [value] : []}
      {...(resetOnOpen ? { onOpenChange: resetSearchOnOpen } : {})}
    >
      <DialogComboboxTrigger
        aria-label={ariaLabel}
        withInlineLabel={false}
        placeholder={placeholder}
        width={160}
        allowClear={false}
        {...(resetOnOpen ? { renderDisplayedValue } : {})}
      />
      <DialogComboboxContent width={160} style={{ zIndex }}>
        <DialogComboboxOptionList>
          <DialogComboboxOptionListSearch onSearch={setSearchValue}>
            {options.map((option) => (
              <DialogComboboxOptionListSelectItem
                key={option.value}
                value={option.value}
                checked={option.value === value}
                onChange={onChange}
              >
                {option.label}
              </DialogComboboxOptionListSelectItem>
            ))}
            {showCustomValue ? (
              <DialogComboboxOptionListSelectItem
                key={searchValue}
                value={searchValue}
                checked={searchValue === value}
                onChange={onChange}
              >
                {intl.formatMessage(
                  { defaultMessage: 'Use "{value}"', description: 'Freeform option in the traces filter key combobox' },
                  { value: searchValue },
                )}
              </DialogComboboxOptionListSelectItem>
            ) : null}
          </DialogComboboxOptionListSearch>
        </DialogComboboxOptionList>
      </DialogComboboxContent>
    </DialogCombobox>
  );
};

/**
 * One Field + Operator + Value row in the filter builder. Module-scope (not nested) per the repo
 * nested-component rule; all state lives in the parent's draft list. Selecting a new field resets
 * the operator to that field's default and clears the value.
 */
const FilterClauseRow = ({ clause, index, fields, onChangeClause, onDelete }: FilterClauseRowProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const field = fields.find((f) => f.id === clause.field);
  const operators = field?.operators ?? [];
  const valueOptions = field?.getValueOptions?.(clause.key ?? '', clause.operator) ?? field?.options ?? [];
  const operatorDisabled = operators.length <= 1;

  const fieldId = `${COMPONENT_ID}-filter-field-${index}`;
  const keyId = `${COMPONENT_ID}-filter-key-${index}`;
  const operatorId = `${COMPONENT_ID}-filter-operator-${index}`;
  const valueId = `${COMPONENT_ID}-filter-value-${index}`;
  // Lift the portaled option list above the enclosing Popover.Content, which otherwise stacks over it.
  const contentProps = { style: { zIndex: theme.options.zIndexBase + 100 } };

  return (
    <div css={{ display: 'flex', flexWrap: 'wrap', gap: theme.spacing.sm, alignItems: 'flex-end' }}>
      <div css={{ display: 'flex', flexDirection: 'column' }}>
        <FormUI.Label htmlFor={fieldId}>
          <FormattedMessage defaultMessage="Field" description="Label for the field selector in the traces filter" />
        </FormUI.Label>
        <SimpleSelect
          id={fieldId}
          componentId={`${COMPONENT_ID}.filter.field`}
          aria-label={intl.formatMessage({
            defaultMessage: 'Filter field',
            description: 'Aria label for the trace filter field selector',
          })}
          width={160}
          contentProps={contentProps}
          value={clause.field}
          onChange={(e) => {
            const nextField = fields.find((f) => f.id === e.target.value);
            onChangeClause(index, {
              field: e.target.value,
              operator: nextField?.operators[0] ?? clause.operator,
              value: '',
              // Seed/clear the key so it exists iff the newly-selected field requires one.
              key: nextField?.requiresKey ? '' : undefined,
            });
          }}
        >
          {fields.map((f) => (
            <SimpleSelectOption key={f.id} value={f.id}>
              {f.label}
            </SimpleSelectOption>
          ))}
        </SimpleSelect>
      </div>

      {field?.requiresKey && (
        <div css={{ display: 'flex', flexDirection: 'column' }}>
          <FormUI.Label htmlFor={keyId}>
            <FormattedMessage defaultMessage="Key" description="Label for the key input in the traces filter" />
          </FormUI.Label>
          {field.keyInput === 'combobox' ? (
            <FilterCombobox
              id={keyId}
              value={clause.key ?? ''}
              options={field.keyOptions ?? []}
              ariaLabel={intl.formatMessage({
                defaultMessage: 'Filter key',
                description: 'Aria label for the trace filter key input',
              })}
              placeholder={
                field.keyPlaceholder ??
                intl.formatMessage({ defaultMessage: 'Key', description: 'Placeholder for a filter key input' })
              }
              onChange={(key) => onChangeClause(index, { ...clause, key })}
              zIndex={theme.options.zIndexBase + 100}
            />
          ) : (
            <Input
              id={keyId}
              componentId={`${COMPONENT_ID}.filter.key`}
              aria-label={intl.formatMessage({
                defaultMessage: 'Filter key',
                description: 'Aria label for the trace filter key input',
              })}
              css={{ width: 160 }}
              placeholder={
                field.keyPlaceholder ??
                intl.formatMessage({
                  defaultMessage: 'Key',
                  description: 'Placeholder for a filter key input',
                })
              }
              value={clause.key ?? ''}
              onChange={(e) => onChangeClause(index, { ...clause, key: e.target.value })}
            />
          )}
        </div>
      )}

      <div css={{ display: 'flex', flexDirection: 'column' }}>
        <FormUI.Label htmlFor={operatorId}>
          <FormattedMessage
            defaultMessage="Operator"
            description="Label for the operator selector in the traces filter"
          />
        </FormUI.Label>
        <SimpleSelect
          id={operatorId}
          componentId={`${COMPONENT_ID}.filter.operator`}
          aria-label={intl.formatMessage({
            defaultMessage: 'Filter operator',
            description: 'Aria label for the trace filter operator selector',
          })}
          width={120}
          contentProps={contentProps}
          value={clause.operator}
          disabled={operatorDisabled}
          onChange={(e) => onChangeClause(index, { ...clause, operator: e.target.value as FilterOp })}
        >
          {operators.map((op) => (
            <SimpleSelectOption key={op} value={op}>
              {op}
            </SimpleSelectOption>
          ))}
        </SimpleSelect>
      </div>

      <div css={{ display: 'flex', flexDirection: 'column' }}>
        <FormUI.Label htmlFor={valueId}>
          <FormattedMessage defaultMessage="Value" description="Label for the value input in the traces filter" />
        </FormUI.Label>
        {field?.valueInput === 'select' ? (
          <SimpleSelect
            id={valueId}
            componentId={`${COMPONENT_ID}.filter.value-select`}
            aria-label={intl.formatMessage({
              defaultMessage: 'Filter value',
              description: 'Aria label for the trace filter value selector',
            })}
            width={160}
            contentProps={contentProps}
            value={clause.value}
            placeholder={intl.formatMessage({
              defaultMessage: 'Select',
              description: 'Placeholder for the trace filter value selector',
            })}
            onChange={(e) => onChangeClause(index, { ...clause, value: e.target.value })}
          >
            {valueOptions.map((option) => (
              <SimpleSelectOption key={option.value} value={option.value}>
                {option.label}
              </SimpleSelectOption>
            ))}
          </SimpleSelect>
        ) : field?.valueInput === 'combobox' ? (
          <FilterCombobox
            key={`${field.id}:${clause.key ?? ''}`}
            id={valueId}
            value={clause.value}
            options={valueOptions}
            ariaLabel={intl.formatMessage({
              defaultMessage: 'Filter value',
              description: 'Aria label for the trace filter value combobox',
            })}
            placeholder={
              field.valuePlaceholder ??
              intl.formatMessage({
                defaultMessage: 'Select or type',
                description: 'Placeholder for a searchable trace filter value combobox',
              })
            }
            onChange={(value) => onChangeClause(index, { ...clause, value })}
            zIndex={theme.options.zIndexBase + 100}
            resetOnOpen
          />
        ) : (
          <Input
            id={valueId}
            componentId={`${COMPONENT_ID}.filter.value`}
            aria-label={intl.formatMessage({
              defaultMessage: 'Filter value',
              description: 'Aria label for the trace filter value input',
            })}
            css={{ width: 160 }}
            type={field?.valueInput === 'number' ? 'number' : 'text'}
            placeholder={
              field?.valuePlaceholder ??
              intl.formatMessage({
                defaultMessage: 'Value',
                description: 'Placeholder for a filter value input',
              })
            }
            value={clause.value}
            onChange={(e) => onChangeClause(index, { ...clause, value: e.target.value })}
          />
        )}
      </div>

      <Button
        componentId={`${COMPONENT_ID}.filter.delete-clause`}
        type="tertiary"
        icon={<CloseSmallIcon />}
        aria-label={intl.formatMessage({
          defaultMessage: 'Remove filter',
          description: 'Aria label for the button that removes a trace filter clause',
        })}
        onClick={() => onDelete(index)}
      />
    </div>
  );
};

/**
 * Generic multi-clause filter popover, fully driven by `fields`. A list of Field/Operator/Value rows
 * with "Add filter" and a primary "Apply filters" button; edits accumulate in a local draft and only
 * commit (via `onChange`) on Apply. The trigger shows an active-clause count badge and a clear-all
 * affordance and styles itself active when any clause is applied. The consumer owns compiling the
 * applied `TraceFilterModel` into a server filter string.
 */
export const TraceFilterButton: React.FC<TraceFilterButtonProps> = ({
  fields,
  filterModel,
  onChange,
  onClearAll,
  activeCount,
  renderTrigger,
  description,
  validateFilters,
  sqlFilter: sqlFilterConfig,
}: TraceFilterButtonProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [open, setOpen] = useState(false);
  const [draft, setDraft] = useState<FilterClause[]>([]);
  // A rejected Apply shows the consumer's error until the editor closes.
  const [attemptedApply, setAttemptedApply] = useState(false);
  useEffect(() => {
    if (!open) {
      setAttemptedApply(false);
    }
  }, [open]);
  // The applied clause being edited on its own; only a custom trigger opens the editor this way.
  const [editIndex, setEditIndex] = useState<number>();
  const hasCustomTrigger = Boolean(renderTrigger);

  const [activeTab, setActiveTab] = useState('basic');
  const [sqlDraft, setSqlDraft] = useState('');
  const [sqlOnlyDraft, setSqlOnlyDraft] = useState(false);
  const [basicReplacesSql, setBasicReplacesSql] = useState(false);
  const [editingSqlClause, setEditingSqlClause] = useState(false);
  const [sqlError, setSqlError] = useState<string>();
  const basicSqlDraftRef = useRef<string>();
  const pendingSqlRef = useRef<{ value: string; applied: string; previousError?: Error; sawFetch: boolean }>();
  const sqlFilter =
    editingSqlClause && sqlFilterConfig?.appliedClause
      ? { ...sqlFilterConfig, ...sqlFilterConfig.appliedClause }
      : sqlFilterConfig;

  const resetDraft = (target?: number | 'sql') => {
    const index = typeof target === 'number' ? target : undefined;
    const isSql = target === 'sql';
    const config =
      isSql && sqlFilterConfig?.appliedClause
        ? { ...sqlFilterConfig, ...sqlFilterConfig.appliedClause }
        : sqlFilterConfig;
    const basic = index === undefined ? filterModel.map((clause) => ({ ...clause })) : [{ ...filterModel[index] }];
    const nextSql = index !== undefined ? (config?.formatBasic(basic) ?? '') : (config?.value ?? '');
    setEditIndex(index);
    setEditingSqlClause(isSql);
    setDraft(isSql ? (config?.parseBasic(nextSql) ?? []) : basic.length ? basic : [makeEmptyClause(fields)]);
    setSqlDraft(nextSql);
    setActiveTab(isSql || (index === undefined && config?.preferSql) ? 'sql' : 'basic');
    setSqlOnlyDraft(
      Boolean((isSql || (index === undefined && config?.preferSql)) && config?.parseBasic(nextSql) === undefined),
    );
    setBasicReplacesSql(false);
    setSqlError(undefined);
    setAttemptedApply(false);
    basicSqlDraftRef.current = undefined;
  };

  const handleOpenChange = (nextOpen: boolean) => {
    if (nextOpen) {
      pendingSqlRef.current = undefined;
      resetDraft();
    }
    setOpen(nextOpen);
  };

  useEffect(() => {
    const pending = pendingSqlRef.current;
    if (!pending) return;
    if (sqlFilter?.isFetching) pending.sawFetch = true;
    if (sqlFilter?.error && sqlFilter.error !== pending.previousError && sqlFilter.value.trim() === pending.applied) {
      setEditIndex(undefined);
      setSqlDraft(pending.value);
      setSqlOnlyDraft(true);
      setActiveTab('sql');
      setSqlError(sqlFilter.error.message);
      setOpen(true);
      pendingSqlRef.current = undefined;
    } else if (pending.sawFetch && !sqlFilter?.isFetching) {
      pendingSqlRef.current = undefined;
    }
  }, [sqlFilter?.error, sqlFilter?.value, sqlFilter?.isFetching]);

  const openTimerRef = useRef<ReturnType<typeof setTimeout>>();
  useEffect(() => () => clearTimeout(openTimerRef.current), []);
  const openEditor = (target?: number | 'sql') => {
    if (typeof target === 'number' && !filterModel[target]) return;
    pendingSqlRef.current = undefined;
    clearTimeout(openTimerRef.current);
    resetDraft(target);
    if (target === undefined) setDraft([...filterModel, makeEmptyClause(fields)]);
    // Let the caller's suggestions popup close and restore focus before this editor opens.
    openTimerRef.current = setTimeout(() => {
      openTimerRef.current = undefined;
      setOpen(true);
    }, 0);
  };

  const hasActiveFilters = activeCount > 0;
  const anchorRef = useRef<HTMLDivElement>(null);

  // The buttons drive the popover via a controlled toggle (Popover.Anchor positions it) so a click
  // while open closes it. But the buttons live outside Radix's Content, so an open-state click first
  // triggers Radix's outside-interaction auto-close, then our toggle reopens it — a close→reopen
  // flicker. `onInteractOutside` below preventDefaults interactions that land on the anchor pill, so
  // our toggle is the single source of truth for those clicks.
  const toggleOpen = () => handleOpenChange(!open);

  const changeClause = (index: number, next: FilterClause) =>
    setDraft((prev) => prev.map((clause, i) => (i === index ? next : clause)));

  const deleteClause = (index: number) => {
    if (editIndex !== undefined && draft.length === 1) {
      onChange(filterModel.filter((_, i) => i !== editIndex));
      setOpen(false);
      return;
    }
    setDraft((prev) => {
      const next = prev.filter((_, i) => i !== index);
      return next.length > 0 ? next : [makeEmptyClause(fields)];
    });
  };

  const addClause = () => setDraft((prev) => [...prev, makeEmptyClause(fields)]);

  // The model Apply commits: the edited clause in place, or the whole draft.
  const getAppliedModel = (): TraceFilterModel => {
    if (editIndex !== undefined) {
      return filterModel.flatMap((clause, index) => (index === editIndex ? draft.filter(isClauseComplete) : [clause]));
    }
    // A custom trigger shows only complete clauses, so it doesn't keep a blank row it can't display.
    return hasCustomTrigger ? draft.filter(isClauseComplete) : draft;
  };
  const validationError = attemptedApply ? validateFilters?.(getAppliedModel()) : undefined;

  const applyFilters = () => {
    if (editIndex !== undefined && !isClauseComplete(draft[0])) return;
    const next = getAppliedModel();
    if (validateFilters?.(next)) {
      setAttemptedApply(true);
      return;
    }
    if (sqlOnlyDraft) return;
    if ((basicReplacesSql || editingSqlClause) && editIndex === undefined && sqlFilter) {
      sqlFilter.onChange(sqlFilter.formatBasic(next.filter(isClauseComplete)));
    } else {
      onChange(next);
    }
    setOpen(false);
  };

  const changeSqlDraft = (value: string) => {
    pendingSqlRef.current = undefined;
    basicSqlDraftRef.current = undefined;
    setSqlDraft(value);
    setSqlOnlyDraft(sqlFilter?.parseBasic(value) === undefined);
    setSqlError(undefined);
  };

  const applySqlFilter = () => {
    if (!sqlFilter) return;
    if (sqlDraft === basicSqlDraftRef.current) {
      const error = validateFilters?.(getAppliedModel());
      if (error || draft.some((clause) => !isClauseComplete(clause))) {
        setSqlError(
          error ??
            intl.formatMessage({
              defaultMessage: 'Complete the condition before applying this filter.',
              description: 'Error for an incomplete Basic condition shown in the SQL editor',
            }),
        );
        return;
      }
    }
    const value = sqlDraft.trim();
    if (editIndex === undefined && value === sqlFilter.value.trim()) {
      setOpen(false);
      return;
    }
    if (editIndex !== undefined && sqlFilter.replaceClause) {
      const applied = sqlFilter.replaceClause(editIndex, value);
      setEditIndex(undefined);
      setEditingSqlClause(false);
      pendingSqlRef.current = { value: applied, applied, previousError: sqlFilter.error, sawFetch: false };
    } else {
      const parsed = sqlFilter.parseBasic(value);
      pendingSqlRef.current = {
        value,
        applied: !editingSqlClause && parsed ? sqlFilter.formatBasic(parsed) : value,
        previousError: sqlFilter.error,
        sawFetch: false,
      };
      sqlFilter.onChange(value);
    }
    setOpen(false);
  };

  // Clear everything and close; the next opening seeds from the applied state.
  const clearAll = () => {
    pendingSqlRef.current = undefined;
    onClearAll();
    setOpen(false);
  };

  const basicEditor = (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.lg }}>
      {sqlOnlyDraft && (
        <FormUI.Message
          type="warning"
          message={intl.formatMessage({
            defaultMessage: 'This SQL expression cannot be represented by Basic filters. Edit it in SQL.',
            description: 'Warning for a predicate that Basic controls cannot represent',
          })}
        />
      )}
      {description && <Typography.Text color="secondary">{description}</Typography.Text>}
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
        {!sqlOnlyDraft &&
          draft.map((clause, index) => (
            <FilterClauseRow
              key={index}
              clause={clause}
              index={index}
              fields={fields}
              onChangeClause={changeClause}
              onDelete={deleteClause}
            />
          ))}
      </div>
      {validationError && <FormUI.Message type="error" message={validationError} />}
      {editIndex === undefined && (
        <div>
          <Button componentId={`${COMPONENT_ID}.filter.add-clause`} icon={<PlusIcon />} onClick={addClause}>
            <FormattedMessage defaultMessage="Add filter" description="Button that adds another trace filter clause" />
          </Button>
        </div>
      )}
      <div css={{ display: 'flex', justifyContent: 'flex-end', gap: theme.spacing.sm }}>
        {editIndex === undefined && hasActiveFilters && (
          <Button componentId={`${COMPONENT_ID}.filter.clear`} onClick={clearAll}>
            <FormattedMessage
              defaultMessage="Clear filters"
              description="Button that clears all applied trace filters from the filter popover"
            />
          </Button>
        )}
        <Button
          componentId={`${COMPONENT_ID}.filter.apply`}
          type="primary"
          onClick={applyFilters}
          disabled={sqlOnlyDraft || (editIndex !== undefined && !isClauseComplete(draft[0]))}
        >
          {editIndex === undefined ? (
            <FormattedMessage
              defaultMessage="Apply filters"
              description="Primary button that applies the trace filter clauses"
            />
          ) : (
            <FormattedMessage
              defaultMessage="Apply filter"
              description="Primary button that applies changes to one trace filter clause"
            />
          )}
        </Button>
      </div>
    </div>
  );

  return (
    <Popover.Root componentId={`${COMPONENT_ID}.filter.popover`} open={open} onOpenChange={handleOpenChange}>
      {/* Two states share one bordered wrapper so the visible border is owned in one place (the child DS
          Buttons are borderless tertiary). Inactive: an icon-only funnel with a tooltip, matching the
          Columns control. Active: a pill reading funnel + "Filters" (N), then the clear (×), then the
          chevron last. Trigger and clear stay siblings — nesting an interactive control inside the
          trigger <button> is invalid a11y and bubbles the clear click back into toggling the popover. */}
      {/* Popover.Anchor (not Trigger) positions the content; the buttons drive `open` themselves via a
          controlled toggle so clicking while open closes it (a second Trigger would be read as an
          outside-click then reopen, causing a flicker). The child DS Buttons keep their default type
          (gray icon, not tertiary-blue) and the wrapper owns the single visible border + active fill. */}
      <Popover.Anchor asChild>
        <div
          ref={anchorRef}
          css={
            renderTrigger
              ? { width: '100%' }
              : {
                  display: 'inline-flex',
                  alignItems: 'center',
                  borderRadius: theme.borders.borderRadiusSm,
                  border: `1px solid ${
                    hasActiveFilters ? theme.colors.actionDefaultBorderFocus : theme.colors.actionDefaultBorderDefault
                  }`,
                  backgroundColor: hasActiveFilters ? theme.colors.actionDefaultBackgroundHover : undefined,
                  '& > button': {
                    border: 'none !important',
                    boxShadow: 'none !important',
                    backgroundColor: 'transparent !important',
                  },
                }
          }
        >
          {renderTrigger ? (
            renderTrigger(openEditor)
          ) : hasActiveFilters ? (
            <>
              <Button
                componentId={`${COMPONENT_ID}.filter.trigger`}
                icon={<FilterIcon css={{ color: theme.colors.textSecondary }} />}
                onClick={toggleOpen}
                // Popover.Anchor (unlike Trigger) adds no ARIA, so announce the popover state ourselves.
                aria-haspopup="dialog"
                aria-expanded={open}
                // Names the button when its label collapses to icon-only.
                aria-label={intl.formatMessage({
                  defaultMessage: 'Filters',
                  description: 'Aria label for the traces table filter button',
                })}
                // The DS label padding `4px 12px` is importantified, so the override needs `!important` (plus
                // `&&&` specificity) to win. Trim the right side below 12px so the × sits closer to (N).
                css={{ '&&&': { paddingRight: '6px !important' } }}
              >
                <span css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
                  <ToolbarCollapsibleLabel>
                    <FormattedMessage defaultMessage="Filters" description="Label for the traces table filter button" />
                  </ToolbarCollapsibleLabel>
                  <Typography.Text color="secondary">({activeCount})</Typography.Text>
                </span>
              </Button>
              <Button
                componentId={`${COMPONENT_ID}.filter.clear-all`}
                icon={<XCircleFillIcon />}
                aria-label={intl.formatMessage({
                  defaultMessage: 'Clear all filters',
                  description: 'Aria label for the clear-all-filters button next to the traces filter button',
                })}
                onClick={clearAll}
                // Icon-only DS Buttons are forced to a fixed square width (`width: heightSm`), which leaves a
                // big empty box around the glyph. Unset that, but keep a modest side padding so the × has a
                // comfortable click target (rather than hugging the glyph exactly).
                css={{ '&&&': { width: 'auto !important', minWidth: '0 !important', paddingLeft: 4, paddingRight: 4 } }}
              />
              <Button
                componentId={`${COMPONENT_ID}.filter.chevron`}
                icon={<ChevronDownIcon />}
                aria-label={intl.formatMessage({
                  defaultMessage: 'Toggle filters',
                  description: 'Aria label for the chevron that toggles the traces filter popover',
                })}
                aria-haspopup="dialog"
                aria-expanded={open}
                onClick={toggleOpen}
                css={{ '&&&': { width: 'auto !important', minWidth: '0 !important', paddingLeft: 2, paddingRight: 2 } }}
              />
            </>
          ) : (
            <Tooltip
              componentId={`${COMPONENT_ID}.filter.trigger.tooltip`}
              content={intl.formatMessage({
                defaultMessage: 'Filters',
                description: 'Tooltip for the traces table filter button',
              })}
            >
              <Button
                componentId={`${COMPONENT_ID}.filter.trigger`}
                icon={<FilterIcon css={{ color: theme.colors.textSecondary }} />}
                aria-label={intl.formatMessage({
                  defaultMessage: 'Filters',
                  description: 'Aria label for the traces table filter button',
                })}
                aria-haspopup="dialog"
                aria-expanded={open}
                onClick={toggleOpen}
              />
            </Tooltip>
          )}
        </div>
      </Popover.Anchor>
      <Popover.Content
        align={renderTrigger ? 'start' : 'end'}
        css={{
          padding: theme.spacing.md,
          width: sqlFilter ? (activeTab === 'sql' ? 560 : 'max-content') : undefined,
          maxWidth: 'calc(100vw - 32px)',
          maxHeight: 'calc(100vh - 32px)',
          overflow: 'auto',
          boxSizing: 'border-box',
        }}
        // Clicks on the anchor pill (the toggle buttons) must not trigger Radix's outside-close, or the
        // close races our onClick toggle and reopens — a flicker. Let our toggle own those clicks. A
        // custom trigger holds a search input, and focusing it should close the editor.
        onInteractOutside={(event) => {
          const target = event.detail.originalEvent.target;
          if (!renderTrigger && target instanceof Node && anchorRef.current?.contains(target)) {
            event.preventDefault();
          }
        }}
      >
        {sqlFilter ? (
          <Tabs.Root
            componentId={`${COMPONENT_ID}.filter.tabs`}
            value={activeTab}
            onValueChange={(tab) => {
              if (tab === 'sql' && sqlFilter && !sqlOnlyDraft) {
                const value =
                  sqlFilter.formatBasicDraft?.(draft) ?? sqlFilter.formatBasic(draft.filter(isClauseComplete));
                basicSqlDraftRef.current = value;
                setSqlDraft(value);
                setSqlError(undefined);
              }
              if (tab === 'basic' && sqlFilter && sqlDraft !== basicSqlDraftRef.current) {
                const parsed = sqlFilter.parseBasic(sqlDraft);
                setSqlOnlyDraft(parsed === undefined);
                if (parsed) {
                  setDraft(parsed.length ? parsed : [makeEmptyClause(fields)]);
                  setBasicReplacesSql(true);
                }
              }
              setActiveTab(tab);
            }}
          >
            {sqlFilter && (editIndex === undefined || sqlFilter.replaceClause) && (
              <Tabs.List>
                <Tabs.Trigger value="basic">
                  <FormattedMessage defaultMessage="Basic" description="Basic trace filters tab" />
                </Tabs.Trigger>
                <Tabs.Trigger value="sql">
                  <FormattedMessage defaultMessage="SQL" description="SQL trace filters tab" />
                </Tabs.Trigger>
              </Tabs.List>
            )}
            <Tabs.Content value="basic" tabIndex={-1}>
              {basicEditor}
            </Tabs.Content>
            {sqlFilter && (
              <Tabs.Content value="sql" tabIndex={-1}>
                <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
                  <Typography.Text>
                    {intl.formatMessage({
                      defaultMessage: 'Trace filter predicate',
                      description: 'Label for the trace SQL predicate editor',
                    })}
                  </Typography.Text>
                  {sqlFilter.renderEditor?.(sqlDraft, changeSqlDraft) ?? (
                    <Input.TextArea
                      componentId={`${COMPONENT_ID}.filter.sql`}
                      aria-label={intl.formatMessage({
                        defaultMessage: 'Trace filter predicate',
                        description: 'Accessible trace SQL editor label',
                      })}
                      value={sqlDraft}
                      onChange={(event) => changeSqlDraft(event.target.value)}
                      autoSize={{ minRows: 4, maxRows: 8 }}
                    />
                  )}
                  {(sqlError ||
                    (sqlDraft.trim() && sqlDraft.trim() === sqlFilter.value.trim() && sqlFilter.error?.message)) && (
                    <FormUI.Message type="error" message={sqlError ?? sqlFilter.error?.message} />
                  )}
                  <Button componentId={`${COMPONENT_ID}.filter.sql-apply`} type="primary" onClick={applySqlFilter}>
                    <FormattedMessage defaultMessage="Apply filter" description="Apply trace SQL predicate button" />
                  </Button>
                </div>
              </Tabs.Content>
            )}
          </Tabs.Root>
        ) : (
          basicEditor
        )}
      </Popover.Content>
    </Popover.Root>
  );
};
