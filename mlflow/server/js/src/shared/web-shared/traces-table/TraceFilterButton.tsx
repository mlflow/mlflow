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
  makeEmptyClause,
} from './filterModel';

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
}

interface FilterClauseRowProps {
  clause: FilterClause;
  index: number;
  fields: FilterFieldDef[];
  onChangeClause: (index: number, next: FilterClause) => void;
  onDelete: (index: number) => void;
}

interface FilterKeyComboboxProps {
  id: string;
  value: string;
  options: FilterFieldSelectOption[];
  placeholder: string;
  ariaLabel: string;
  onChange: (value: string) => void;
  /** z-index for the portaled option list (lifted above the enclosing Popover.Content). */
  zIndex: number;
}

/**
 * Freeform-capable key combobox: suggests `options` and offers a `Use "<typed>"` item so an arbitrary
 * key can still be entered (the v3 assessment-filter typeahead pattern). Module-scope (not nested)
 * per the repo nested-component rule; the freeform-search state lives here, and the selected key is
 * controlled by the parent via `value`/`onChange`.
 */
const FilterKeyCombobox = ({
  id,
  value,
  options,
  placeholder,
  ariaLabel,
  onChange,
  zIndex,
}: FilterKeyComboboxProps) => {
  const intl = useIntl();
  const [searchValue, setSearchValue] = useState('');
  const trimmedSearch = searchValue.trim();
  const showCustomValue = trimmedSearch.length > 0 && !options.some((option) => option.value === searchValue);

  return (
    <DialogCombobox componentId={`${COMPONENT_ID}.filter.key-combobox`} id={id} value={value ? [value] : []}>
      <DialogComboboxTrigger
        aria-label={ariaLabel}
        withInlineLabel={false}
        placeholder={placeholder}
        width={160}
        allowClear={false}
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
  const operatorDisabled = operators.length <= 1;

  const fieldId = `${COMPONENT_ID}-filter-field-${index}`;
  const keyId = `${COMPONENT_ID}-filter-key-${index}`;
  const operatorId = `${COMPONENT_ID}-filter-operator-${index}`;
  const valueId = `${COMPONENT_ID}-filter-value-${index}`;
  // Lift the portaled option list above the enclosing Popover.Content, which otherwise stacks over it.
  const contentProps = { style: { zIndex: theme.options.zIndexBase + 100 } };

  return (
    <div css={{ display: 'flex', flexDirection: 'row', gap: theme.spacing.sm, alignItems: 'flex-end' }}>
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
            <FilterKeyCombobox
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
            {(field.options ?? []).map((option) => (
              <SimpleSelectOption key={option.value} value={option.value}>
                {option.label}
              </SimpleSelectOption>
            ))}
          </SimpleSelect>
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
}: TraceFilterButtonProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [open, setOpen] = useState(false);
  const [draft, setDraft] = useState<FilterClause[]>([]);

  // Seed the draft from the applied model whenever the popover opens, so it reflects the current
  // filters (and a fresh blank row when nothing is applied yet).
  useEffect(() => {
    if (open) {
      setDraft(filterModel.length > 0 ? filterModel.map((clause) => ({ ...clause })) : [makeEmptyClause(fields)]);
    }
  }, [open, filterModel, fields]);

  const hasActiveFilters = activeCount > 0;
  const anchorRef = useRef<HTMLDivElement>(null);

  // The buttons drive the popover via a controlled toggle (Popover.Anchor positions it) so a click
  // while open closes it. But the buttons live outside Radix's Content, so an open-state click first
  // triggers Radix's outside-interaction auto-close, then our toggle reopens it — a close→reopen
  // flicker. `onInteractOutside` below preventDefaults interactions that land on the anchor pill, so
  // our toggle is the single source of truth for those clicks.
  const toggleOpen = () => setOpen((prev) => !prev);

  const changeClause = (index: number, next: FilterClause) =>
    setDraft((prev) => prev.map((clause, i) => (i === index ? next : clause)));

  const deleteClause = (index: number) =>
    setDraft((prev) => {
      const next = prev.filter((_, i) => i !== index);
      return next.length > 0 ? next : [makeEmptyClause(fields)];
    });

  const addClause = () => setDraft((prev) => [...prev, makeEmptyClause(fields)]);

  const applyFilters = () => {
    onChange(draft);
    setOpen(false);
  };

  // Clear everything and close. Draft is intentionally not reset here because the open-effect reseeds it.
  const clearAll = () => {
    onClearAll();
    setOpen(false);
  };

  return (
    <Popover.Root componentId={`${COMPONENT_ID}.filter.popover`} open={open} onOpenChange={setOpen}>
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
          css={{
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
          }}
        >
          {hasActiveFilters ? (
            <>
              <Button
                componentId={`${COMPONENT_ID}.filter.trigger`}
                icon={<FilterIcon />}
                onClick={toggleOpen}
                // Popover.Anchor (unlike Trigger) adds no ARIA, so announce the popover state ourselves.
                aria-haspopup="dialog"
                aria-expanded={open}
                // The DS label padding `4px 12px` is importantified, so the override needs `!important` (plus
                // `&&&` specificity) to win. Trim the right side below 12px so the × sits closer to (N).
                css={{ '&&&': { paddingRight: '6px !important' } }}
              >
                <span css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
                  <FormattedMessage defaultMessage="Filters" description="Label for the traces table filter button" />
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
                icon={<FilterIcon />}
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
        align="end"
        css={{ padding: theme.spacing.md }}
        // Clicks on the anchor pill (the toggle buttons) must not trigger Radix's outside-close, or the
        // close races our onClick toggle and reopens — a flicker. Let our toggle own those clicks.
        onInteractOutside={(event) => {
          const target = event.detail.originalEvent.target;
          if (target instanceof Node && anchorRef.current?.contains(target)) {
            event.preventDefault();
          }
        }}
      >
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.lg }}>
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
            {draft.map((clause, index) => (
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
          <div>
            <Button componentId={`${COMPONENT_ID}.filter.add-clause`} icon={<PlusIcon />} onClick={addClause}>
              <FormattedMessage
                defaultMessage="Add filter"
                description="Button that adds another trace filter clause"
              />
            </Button>
          </div>
          <div css={{ display: 'flex', justifyContent: 'flex-end', gap: theme.spacing.sm }}>
            {hasActiveFilters && (
              <Button componentId={`${COMPONENT_ID}.filter.clear`} onClick={clearAll}>
                <FormattedMessage
                  defaultMessage="Clear filters"
                  description="Button that clears all applied trace filters from the filter popover"
                />
              </Button>
            )}
            <Button componentId={`${COMPONENT_ID}.filter.apply`} type="primary" onClick={applyFilters}>
              <FormattedMessage
                defaultMessage="Apply filters"
                description="Primary button that applies the trace filter clauses"
              />
            </Button>
          </div>
        </div>
      </Popover.Content>
    </Popover.Root>
  );
};
