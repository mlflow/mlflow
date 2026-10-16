import { useLayoutEffect, useMemo, useRef, useState, type ReactNode, type RefObject } from 'react';
import { flushSync } from 'react-dom';
import { useIntl } from 'react-intl';
import { type InputRef, useDesignSystemTheme } from '@databricks/design-system';
import { UnifiedSearch, type UnifiedSearchOption } from '../../../../UnifiedSearch';
import type { ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';
import {
  FilterOp,
  isClauseComplete,
  type FilterClause,
  type FilterFieldDef,
  type TraceFilterModel,
} from '@databricks/web-shared/traces-table';
import { useMlflowTraceFilterFields } from '../utils/filterModel';
import {
  buildTraceSearchFields,
  formatTraceSearchFieldName,
  formatTraceSearchValue,
  getTraceSearchExamples,
} from '../utils/traceSearchSuggestions';
import { getTraceSearchQuerySuggestions, parseTraceSearchQuery } from '../utils/traceSearchQuery';

const MAX_SUGGESTIONS = 5;
const SEARCH_ICON_WIDTH = 24;
const PILL_MAX_WIDTH = 200;
const SEARCH_GAP = 4;
const SEARCH_PADDING = 4;
const COMFORTABLE_INPUT_WIDTH = 300;
// Module-level so the field hook's memoization holds across renders.
const NO_ASSESSMENT_NAMES: string[] = [];

export interface TracesV4SearchProps {
  traces: ModelTraceInfoV3[];
  value: string;
  onChange: (value: string) => void;
  onClear: () => void;
  onSubmit: () => void;
  onApplyFilter: (clause: FilterClause) => void;
  filterModel?: TraceFilterModel;
  onRemoveFilter?: (index: number) => void;
  onOpenFilter?: (index?: number) => void;
  onRunTextSearch?: (text: string) => void;
  /** Desired inline width, independent of the available toolbar space. */
  onMinimumWidthChange?: (width: number) => void;
  inputRef?: RefObject<InputRef>;
  suffix?: React.ReactNode;
}

type RecentSearch = { text: string; clause?: FilterClause };

interface QueryLabelProps {
  field: string;
  operator?: string;
  value?: string;
  suffix?: string;
}

const QueryLabel = ({ field, operator, value, suffix }: QueryLabelProps) => {
  const { theme } = useDesignSystemTheme();
  return (
    <code css={{ fontFamily: 'monospace', whiteSpace: 'nowrap' }}>
      <span css={{ color: theme.colors.blue600 }}>{field}</span>
      {suffix && <span css={{ color: theme.colors.blue600 }}>{suffix}</span>}
      {operator && (
        <>
          {' '}
          <span css={{ color: theme.colors.red600 }}>{operator}</span>
        </>
      )}
      {value && (
        <>
          {' '}
          <span css={{ color: theme.colors.textPrimary }}>{value}</span>
        </>
      )}
    </code>
  );
};

interface FilterQueryLabelProps {
  field: Pick<FilterFieldDef, 'id' | 'valueInput'> | string;
  fieldKey?: string;
  operator?: string;
  value?: string;
  displayValue?: string;
  suffix?: string;
}

const FilterQueryLabel = ({ field, fieldKey, operator, value, displayValue, suffix }: FilterQueryLabelProps) => (
  <QueryLabel
    field={formatTraceSearchFieldName(typeof field === 'string' ? field : field.id, fieldKey)}
    operator={operator}
    value={value ? formatTraceSearchValue(field, value, displayValue) : undefined}
    suffix={suffix}
  />
);

export const TracesV4Search = ({
  traces,
  value,
  onChange,
  onClear,
  onSubmit,
  onApplyFilter,
  inputRef,
  suffix,
  filterModel = [],
  onRemoveFilter,
  onOpenFilter,
  onRunTextSearch,
  onMinimumWidthChange,
}: TracesV4SearchProps) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  const fields = useMlflowTraceFilterFields(NO_ASSESSMENT_NAMES);
  const searchFields = useMemo(() => buildTraceSearchFields(fields, traces), [fields, traces]);
  const [open, setOpen] = useState(false);
  const pillRailRef = useRef<HTMLDivElement>(null);
  const lastMinimumWidth = useRef<number>();
  useLayoutEffect(() => {
    if (!onMinimumWidthChange) return;
    const pills = Array.from(pillRailRef.current?.children ?? []);
    let disposed = false;
    const reportWidth = () => {
      if (disposed) return;
      // Pills never shrink within the scroll rail. Measure each capped pill, not the rail's
      // stretched/overflowing width, so moving between toolbar rows cannot change this result.
      const pillWidth = pills.reduce(
        (sum, pill) => sum + Math.min(PILL_MAX_WIDTH, pill.getBoundingClientRect().width),
        0,
      );
      const width = Math.ceil(
        SEARCH_ICON_WIDTH +
          pillWidth +
          SEARCH_GAP * (pills.length + 1) +
          COMFORTABLE_INPUT_WIDTH +
          SEARCH_PADDING * 2 +
          2,
      );
      if (width !== lastMinimumWidth.current) {
        lastMinimumWidth.current = width;
        onMinimumWidthChange(width);
      }
    };
    reportWidth();
    const observer = new ResizeObserver(reportWidth);
    pills.forEach((pill) => observer.observe(pill));
    window.addEventListener('resize', reportWidth);
    document.fonts?.addEventListener('loadingdone', reportWidth);
    void document.fonts?.ready.then(reportWidth);
    return () => {
      disposed = true;
      observer.disconnect();
      window.removeEventListener('resize', reportWidth);
      document.fonts?.removeEventListener('loadingdone', reportWidth);
    };
  }, [filterModel, onMinimumWidthChange]);
  // The root keys this component by experiment. Never persist query contents to storage.
  const [recent, setRecent] = useState<RecentSearch[]>([]);
  const [inputFocused, setInputFocused] = useState(false);
  const fieldDefinition = (clause: FilterClause) => fields.find((field) => field.id === clause.field) ?? clause.field;
  const clauseText = (clause: FilterClause) =>
    `${formatTraceSearchFieldName(clause.field, clause.key)} ${clause.operator} ${formatTraceSearchValue(
      fieldDefinition(clause),
      clause.value,
    )}`;
  const clauseLabel = (clause: FilterClause) => (
    <FilterQueryLabel
      field={fieldDefinition(clause)}
      fieldKey={clause.key}
      operator={clause.operator}
      value={clause.value}
    />
  );
  const remember = (entry: RecentSearch) => {
    setRecent((previous) => [entry, ...previous.filter((item) => item.text !== entry.text)].slice(0, 5));
  };
  const openFilter = (index?: number) => {
    // Close the portal before the editor moves focus to its trigger/input.
    flushSync(() => setOpen(false));
    onOpenFilter?.(index);
  };
  const query = value.trim().toLocaleLowerCase();
  const parsedQuery = useMemo(() => parseTraceSearchQuery(value, fields, searchFields), [fields, searchFields, value]);
  const querySuggestions = useMemo(
    () => getTraceSearchQuerySuggestions(value, fields, searchFields),
    [fields, searchFields, value],
  );
  const querySuggestionLabel = (suggestion: (typeof querySuggestions)[number]): ReactNode => {
    const parsedSuggestion = parseTraceSearchQuery(suggestion.query, fields, searchFields);
    return parsedSuggestion.field ? (
      <FilterQueryLabel
        field={parsedSuggestion.field}
        fieldKey={parsedSuggestion.key}
        operator={parsedSuggestion.operator}
        value={parsedSuggestion.value}
        suffix={
          suggestion.kind === 'key' && parsedSuggestion.field.requiresKey && parsedSuggestion.key === undefined
            ? '.'
            : undefined
        }
      />
    ) : (
      <code css={{ fontFamily: 'monospace' }}>{suggestion.label}</code>
    );
  };

  const apply = (clause: FilterClause) => {
    onApplyFilter(clause);
    remember({ text: clauseText(clause), clause: { ...clause } });
    setOpen(false);
  };
  const selectQuery = (nextValue: string) => {
    const selectedQuery = parseTraceSearchQuery(nextValue, fields, searchFields);
    if (selectedQuery.status === 'complete' && selectedQuery.clause) {
      apply(selectedQuery.clause);
      return;
    }
    onChange(nextValue);
    setOpen(true);
    inputRef?.current?.focus();
  };
  const submitText = () => {
    onSubmit();
    if (value.trim()) remember({ text: value });
    setOpen(false);
  };
  const selectRecent = (entry: RecentSearch) => {
    if (entry.clause) {
      apply(entry.clause);
    } else if (onRunTextSearch) {
      onRunTextSearch(entry.text);
      remember(entry);
      setOpen(false);
    } else {
      // Without the atomic controller command, only insert text; Enter submits it.
      onChange(entry.text);
      setOpen(false);
    }
  };
  const applyOrSubmit = () => {
    if (parsedQuery.status === 'complete' && parsedQuery.clause) {
      apply(parsedQuery.clause);
    } else if (!parsedQuery.structured || parsedQuery.bareField) {
      submitText();
    } else {
      setOpen(false);
    }
  };
  const options: UnifiedSearchOption[] = [];
  if (!query) {
    const examples = getTraceSearchExamples(searchFields);
    options.push(
      ...examples.map((example) => ({
        id: `example:${example.field.id}`,
        label: (
          <FilterQueryLabel
            field={example.field.field}
            fieldKey={example.clause.key}
            operator={example.clause.operator}
            value={example.clause.value}
          />
        ),
        accessibleLabel: example.label,
        select: () => selectQuery(example.label),
      })),
    );
    if (!examples.length) {
      options.push(
        ...searchFields
          .filter((field) => field.field.id !== 'span_attribute' || field.key !== undefined)
          .map((field) => ({
            id: field.id,
            label: <FilterQueryLabel field={field.field} fieldKey={field.key} suffix="…" />,
            accessibleLabel: `${field.label}…`,
            select: () => selectQuery(`${formatTraceSearchFieldName(field.field.id, field.key)} `),
          })),
      );
    }
  } else if (parsedQuery.structured) {
    options.push(
      ...querySuggestions.map((suggestion) => ({
        id: suggestion.id,
        label: querySuggestionLabel(suggestion),
        accessibleLabel: suggestion.label,
        select: () => selectQuery(suggestion.query),
      })),
    );
  } else {
    const spanAttributeNamespace = searchFields.find(
      (field) => field.field.id === 'span_attribute' && field.key === undefined,
    );
    if (spanAttributeNamespace) {
      const namespace = `${formatTraceSearchFieldName(spanAttributeNamespace.field.id)}.`;
      if (namespace.toLocaleLowerCase().startsWith(query)) {
        options.push({
          id: spanAttributeNamespace.id,
          label: <FilterQueryLabel field={spanAttributeNamespace.field} suffix="." />,
          accessibleLabel: namespace,
          select: () => selectQuery(namespace),
        });
      }
    }
    options.push(
      ...searchFields
        .filter(
          (field) =>
            (field.field.id !== 'span_attribute' || field.key !== undefined) &&
            [field.label, ...field.aliases].some((label) => label.toLocaleLowerCase().includes(query)),
        )
        .slice(0, MAX_SUGGESTIONS - 1 - options.length)
        .map((field) => ({
          id: field.id,
          label: <FilterQueryLabel field={field.field} fieldKey={field.key} suffix="…" />,
          accessibleLabel: `${field.label}…`,
          select: () => selectQuery(`${formatTraceSearchFieldName(field.field.id, field.key)} `),
        })),
    );
    if (options.length < MAX_SUGGESTIONS - 1) {
      for (const field of searchFields) {
        for (const candidate of field.values) {
          if (options.length >= MAX_SUGGESTIONS - 1) break;
          if (
            !candidate.label.toLocaleLowerCase().includes(query) ||
            !field.field.operators.includes(FilterOp.EQUALS)
          ) {
            continue;
          }
          options.push({
            id: `value:${field.id}:${candidate.value}`,
            label: (
              <FilterQueryLabel
                field={field.field}
                fieldKey={field.key}
                operator={FilterOp.EQUALS}
                value={candidate.value}
                displayValue={field.field.id === 'duration' ? candidate.label : undefined}
              />
            ),
            accessibleLabel: `${field.label} = ${formatTraceSearchValue(
              field.field,
              candidate.value,
              field.field.id === 'duration' ? candidate.label : undefined,
            )}`,
            select: () =>
              selectQuery(
                `${formatTraceSearchFieldName(field.field.id, field.key)} = ${formatTraceSearchValue(field.field, candidate.value)}`,
              ),
          });
        }
      }
    }
  }
  if (query && (!parsedQuery.structured || parsedQuery.bareField || !options.length)) {
    const label = intl.formatMessage(
      {
        defaultMessage: 'Search text “{text}”',
        description: 'Run a plain text trace search instead of adding a filter',
      },
      { text: value },
    );
    options.push({ id: 'text-search', label, accessibleLabel: label, textSearch: true, select: submitText });
  }
  const recentOptions: UnifiedSearchOption[] = recent
    .filter((entry) => !query || entry.text.toLocaleLowerCase().includes(query))
    .slice(0, 2)
    .map((entry) => ({
      id: `recent:${entry.text}`,
      label: entry.clause ? clauseLabel(entry.clause) : entry.text,
      accessibleLabel: entry.text,
      textSearch: !entry.clause,
      select: () => selectRecent(entry),
    }));
  const seen = new Set(recentOptions.map((option) => option.accessibleLabel));
  const availableSuggestions = options.filter((option) => {
    const parsed = parseTraceSearchQuery(option.accessibleLabel, fields, searchFields);
    const key = option.textSearch ? value : parsed.clause ? clauseText(parsed.clause) : option.accessibleLabel;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
  const suggestionSlots = MAX_SUGGESTIONS - recentOptions.length;
  const bareFieldTextOption =
    parsedQuery.bareField && availableSuggestions.find((option) => option.id === 'text-search');
  const suggestions = bareFieldTextOption
    ? [
        ...availableSuggestions.filter((option) => option !== bareFieldTextOption).slice(0, suggestionSlots - 1),
        bareFieldTextOption,
      ]
    : availableSuggestions.slice(0, suggestionSlots);
  const textSearchPlaceholder = intl.formatMessage({
    defaultMessage: 'Type something to use full text search',
    description: 'Trace typeahead search placeholder',
  });
  const placeholder = inputFocused
    ? textSearchPlaceholder
    : intl.formatMessage({ defaultMessage: 'Add a filter', description: 'Idle trace search placeholder' });

  const chips = filterModel.flatMap((clause, index) =>
    isClauseComplete(clause)
      ? [
          {
            text: clauseText(clause),
            label: clauseLabel(clause),
            onEdit: () => openFilter(index),
            onRemove: () => onRemoveFilter?.(index),
          },
        ]
      : [],
  );
  return (
    <UnifiedSearch
      componentId="mlflow.traces-v4.search.typeahead"
      value={value}
      open={open}
      setOpen={setOpen}
      isAIMode={false}
      placeholder={placeholder}
      ariaLabel={textSearchPlaceholder}
      submitDisabled={
        !value.trim() || (parsedQuery.structured && parsedQuery.status !== 'complete' && !parsedQuery.bareField)
      }
      onSubmit={applyOrSubmit}
      onChange={onChange}
      onClear={onClear}
      onOpenFilter={() => openFilter()}
      onFocusChange={setInputFocused}
      recentOptions={recentOptions}
      suggestions={suggestions}
      chips={chips}
      suffix={suffix}
      inputRef={inputRef}
      pillRailRef={pillRailRef}
      alignDropdownToContainer
      showEmptySuggestionsHeader
    />
  );
};
