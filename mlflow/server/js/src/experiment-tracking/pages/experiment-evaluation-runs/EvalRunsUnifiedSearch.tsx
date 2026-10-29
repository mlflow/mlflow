import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { flushSync } from 'react-dom';

import { InfoFillIcon, InfoPopover, Tooltip, useDesignSystemTheme } from '@databricks/design-system';
import { UnifiedSearch, type UnifiedSearchOption } from '../../components/UnifiedSearch';
import { RunsSearchTooltipContent } from '../../components/experiment-page/components/runs/RunsSearchTooltipContent';
import { FormattedMessage, useIntl } from 'react-intl';

import type { ErrorWrapper } from '../../../common/utils/ErrorWrapper';
import type { ExperimentRunsSelectorResult } from '../../components/experiment-page/utils/experimentRuns.selector';
import type { RunEntity } from '../../types';
import type { EvalRunsFilterClause, EvalRunsFilterFieldDef } from './evalRunsFilterFields';
import {
  EVAL_RUNS_FILTER_OPERATOR_LABELS,
  EvalRunsFilterOperator,
  getEvalRunsFilterField,
  getEvalRunsFilterFields,
  getOperatorsForFieldType,
  quoteFilterValue,
} from './evalRunsFilterFields';
import { useEvalRunsFilterLabels } from './hooks/useEvalRunsFilterLabels';

const MAX_OPTIONS = 5;
const MAX_RECENT_OPTIONS = 2;

interface EvalRunsUnifiedSearchProps {
  runs: RunEntity[];
  runsData: ExperimentRunsSelectorResult;
  searchFilter: string;
  onSearchFilterChange: (value: string) => void;
  clauses: EvalRunsFilterClause[];
  setClauses: (clauses: EvalRunsFilterClause[]) => void;
  openFilter: (filterIndex?: number) => void;
  requestError?: ErrorWrapper | Error | null;
  suffix?: ReactNode;
  isAIMode?: boolean;
  aiPlaceholder?: string;
  onAISubmit?: (request: string) => void;
}

type RecentSearch = { text: string; clause?: EvalRunsFilterClause };

type ParsedQuery =
  | { status: 'text' }
  | { status: 'field'; field: EvalRunsFilterFieldDef }
  | {
      status: 'operator';
      field: EvalRunsFilterFieldDef;
      operator: EvalRunsFilterOperator;
    }
  | {
      status: 'complete';
      field: EvalRunsFilterFieldDef;
      clause: EvalRunsFilterClause;
    };

const fieldForClause = (fields: EvalRunsFilterFieldDef[], clause: EvalRunsFilterClause) =>
  fields.find(({ id }) => id === clause.field) ?? getEvalRunsFilterField(clause.field);

const formatValue = (field: EvalRunsFilterFieldDef | undefined, value: string) =>
  field?.type === 'number' ? value : quoteFilterValue(value);

const formatClause = (fields: EvalRunsFilterFieldDef[], clause: EvalRunsFilterClause) => {
  const field = fieldForClause(fields, clause);
  return `${field?.id ?? clause.field} ${EVAL_RUNS_FILTER_OPERATOR_LABELS[clause.operator]} ${formatValue(
    field,
    clause.value,
  )}`;
};

const parseValue = (rawValue: string) => {
  const value = rawValue.trim();
  if (value.length >= 2 && value[0] === "'" && value.at(-1) === "'") {
    return value.slice(1, -1).replace(/''/g, "'");
  }
  if (value.length >= 2 && value[0] === '"' && value.at(-1) === '"') {
    return value.slice(1, -1);
  }
  return value;
};

const parseQuery = (query: string, fields: EvalRunsFilterFieldDef[]): ParsedQuery => {
  const value = query.trim();
  if (!value || /\s+(?:AND|OR)\s+/i.test(value)) return { status: 'text' };
  const field = fields
    .slice()
    .sort((left, right) => right.id.length - left.id.length)
    .find(({ id }) => {
      const normalizedValue = value.toLocaleLowerCase();
      const normalizedId = id.toLocaleLowerCase();
      return normalizedValue === normalizedId || normalizedValue.startsWith(`${normalizedId} `);
    });
  if (!field) return { status: 'text' };

  const remainder = value.slice(field.id.length).trimStart();
  if (!remainder) return { status: 'field', field };
  const operator = getOperatorsForFieldType(field.type)
    .slice()
    .sort((left, right) => right.length - left.length)
    .find((candidate) => {
      const normalizedRemainder = remainder.toLocaleLowerCase();
      const normalizedOperator = candidate.toLocaleLowerCase();
      return normalizedRemainder === normalizedOperator || normalizedRemainder.startsWith(`${normalizedOperator} `);
    });
  if (!operator) return { status: 'field', field };

  const rawClauseValue = remainder.slice(operator.length).trim();
  if (!rawClauseValue) return { status: 'operator', field, operator };
  const clauseValue = parseValue(rawClauseValue);
  if (field.type === 'number' && !Number.isFinite(Number(clauseValue))) {
    return { status: 'operator', field, operator };
  }
  return {
    status: 'complete',
    field,
    clause: { field: field.id, operator, value: clauseValue },
  };
};

// Accepts every form shown in the search help (`RunsSearchTooltipContent`), e.g. `metric.` and `IN (...)`.
const isSearchRunsExpression = (value: string) =>
  /^(?:attributes?|metrics?|params?|tags?|datasets?)\./i.test(value) &&
  /\s(?:=|!=|>=|<=|>|<|(?:NOT\s+)?(?:LIKE|ILIKE|RLIKE|IN)|IS\s+(?:NOT\s+)?NULL)(?:\s|$|\()/i.test(value);

const buildObservedExamples = (runs: RunEntity[], fields: EvalRunsFilterFieldDef[]): EvalRunsFilterClause[] => {
  const examples: EvalRunsFilterClause[] = [];
  const seen = new Set<string>();
  const add = (clause: EvalRunsFilterClause) => {
    if (!seen.has(clause.field) && fieldForClause(fields, clause)) {
      seen.add(clause.field);
      examples.push(clause);
    }
  };

  for (const run of runs) {
    if (run.info.runName) {
      add({
        field: 'attributes.run_name',
        operator: EvalRunsFilterOperator.EQUALS,
        value: run.info.runName,
      });
    }
    if (run.info.status) {
      add({
        field: 'attributes.status',
        operator: EvalRunsFilterOperator.EQUALS,
        value: run.info.status,
      });
    }
    const metric = examples.some(({ field }) => field.startsWith('metrics.')) ? undefined : run.data?.metrics?.[0];
    if (metric) {
      add({
        field: `metrics.\`${metric.key}\``,
        operator: EvalRunsFilterOperator.EQUALS,
        value: String(metric.value),
      });
    }
    if (run.info.startTime != null) {
      add({
        field: 'attributes.start_time',
        operator: EvalRunsFilterOperator.EQUALS,
        value: String(run.info.startTime),
      });
    }
    const datasetName = run.inputs?.datasetInputs?.[0]?.dataset?.name;
    if (datasetName) {
      add({
        field: 'dataset.name',
        operator: EvalRunsFilterOperator.EQUALS,
        value: datasetName,
      });
    }
    if (examples.length >= MAX_OPTIONS) break;
  }
  return examples.slice(0, MAX_OPTIONS);
};

export const EvalRunsUnifiedSearch = ({
  runs,
  runsData,
  searchFilter,
  onSearchFilterChange,
  clauses,
  setClauses,
  openFilter,
  requestError,
  suffix,
  isAIMode = false,
  aiPlaceholder,
  onAISubmit,
}: EvalRunsUnifiedSearchProps) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  const { operatorLabels, getFieldLabel } = useEvalRunsFilterLabels();
  const [value, setValue] = useState(searchFilter);
  const [open, setOpen] = useState(false);
  const [focused, setFocused] = useState(false);
  const [recent, setRecent] = useState<RecentSearch[]>([]);
  const previousAIMode = useRef(isAIMode);
  const submittedSearch = useRef<{ backend: string; display: string }>();
  const fields = useMemo(() => {
    const available = getEvalRunsFilterFields(runsData);
    for (const clause of clauses) {
      if (!available.some(({ id }) => id === clause.field)) {
        const field = getEvalRunsFilterField(clause.field);
        if (field) available.push(field);
      }
    }
    return available;
  }, [clauses, runsData]);
  const examples = useMemo(() => buildObservedExamples(runs, fields), [fields, runs]);
  const parsed = useMemo(() => parseQuery(value, fields), [fields, value]);

  useEffect(() => {
    if (previousAIMode.current !== isAIMode) {
      previousAIMode.current = isAIMode;
      setValue(isAIMode ? '' : searchFilter);
      setOpen(false);
      return;
    }
    if (!isAIMode) {
      setValue(submittedSearch.current?.backend === searchFilter ? submittedSearch.current.display : searchFilter);
    }
  }, [isAIMode, searchFilter]);

  const remember = (entry: RecentSearch) =>
    setRecent((current) => [entry, ...current.filter(({ text }) => text !== entry.text)].slice(0, MAX_OPTIONS));
  const applyClause = (clause: EvalRunsFilterClause) => {
    const text = formatClause(fields, clause);
    setClauses([...clauses, clause]);
    submittedSearch.current = undefined;
    onSearchFilterChange('');
    setValue('');
    remember({ text, clause });
    setOpen(false);
  };
  const runTextSearch = (text: string) => {
    const trimmed = text.trim();
    if (!trimmed) return;
    const backend = isSearchRunsExpression(trimmed)
      ? trimmed
      : `attributes.run_name ILIKE ${quoteFilterValue(`%${trimmed}%`)}`;
    submittedSearch.current = { backend, display: trimmed };
    onSearchFilterChange(backend);
    setValue(trimmed);
    remember({ text: trimmed });
    setOpen(false);
  };
  const submit = () => {
    const trimmed = value.trim();
    if (!trimmed) {
      if (!isAIMode) {
        submittedSearch.current = undefined;
        onSearchFilterChange('');
        setValue('');
        setOpen(false);
      }
      return;
    }
    if (isAIMode) {
      onAISubmit?.(trimmed);
      setOpen(false);
      return;
    }
    if (parsed.status === 'complete') {
      applyClause(parsed.clause);
      return;
    }
    if ((parsed.status === 'field' || parsed.status === 'operator') && !isSearchRunsExpression(trimmed)) return;
    runTextSearch(trimmed);
  };
  const openEditor = (index?: number) => {
    flushSync(() => setOpen(false));
    openFilter(index);
  };

  const suggestionOptions: UnifiedSearchOption[] = [];
  if (!value.trim()) {
    if (examples.length) {
      suggestionOptions.push(
        ...examples.map((clause) => ({
          id: `example:${clause.field}`,
          accessibleLabel: formatClause(fields, clause),
          label: formatClause(fields, clause),
          textSearch: false,
          select: () => applyClause(clause),
        })),
      );
    } else {
      suggestionOptions.push(
        ...fields.slice(0, MAX_OPTIONS).map((field) => ({
          id: `field:${field.id}`,
          accessibleLabel: `${field.id} …`,
          label: `${field.id} …`,
          textSearch: false,
          select: () => {
            setValue(`${field.id} `);
            setOpen(true);
          },
        })),
      );
    }
  } else if (parsed.status === 'field') {
    suggestionOptions.push(
      ...getOperatorsForFieldType(parsed.field.type).map((operator) => ({
        id: `operator:${operator}`,
        accessibleLabel: `${parsed.field.id} ${EVAL_RUNS_FILTER_OPERATOR_LABELS[operator]} …`,
        label: `${parsed.field.id} ${EVAL_RUNS_FILTER_OPERATOR_LABELS[operator]} …`,
        textSearch: false,
        select: () => {
          setValue(`${parsed.field.id} ${operator} `);
          setOpen(true);
        },
      })),
    );
  } else if (parsed.status === 'operator' && parsed.field.options) {
    suggestionOptions.push(
      ...parsed.field.options.map((option) => {
        const clause = {
          field: parsed.field.id,
          operator: parsed.operator,
          value: option.value,
        };
        return {
          id: `value:${option.value}`,
          accessibleLabel: formatClause(fields, clause),
          label: formatClause(fields, clause),
          textSearch: false,
          select: () => applyClause(clause),
        };
      }),
    );
  } else if (parsed.status === 'complete') {
    suggestionOptions.push({
      id: 'apply-filter',
      accessibleLabel: formatClause(fields, parsed.clause),
      label: formatClause(fields, parsed.clause),
      textSearch: false,
      select: () => applyClause(parsed.clause),
    });
  } else {
    const query = value.trim().toLocaleLowerCase();
    suggestionOptions.push(
      ...fields
        .filter(({ id, label }) => `${id} ${label}`.toLocaleLowerCase().includes(query))
        .slice(0, MAX_OPTIONS - 1)
        .map((field) => ({
          id: `field:${field.id}`,
          accessibleLabel: `${field.id} …`,
          label: `${field.id} …`,
          textSearch: false,
          select: () => {
            setValue(`${field.id} `);
            setOpen(true);
          },
        })),
    );
    suggestionOptions.push({
      id: 'text-search',
      accessibleLabel: intl.formatMessage(
        {
          defaultMessage: 'Search run names for “{text}”',
          description: 'Run a plain text evaluation run-name search',
        },
        { text: value },
      ),
      label: intl.formatMessage(
        {
          defaultMessage: 'Search run names for “{text}”',
          description: 'Run a plain text evaluation run-name search',
        },
        { text: value },
      ),
      textSearch: true,
      select: submit,
    });
  }

  const recentOptions: UnifiedSearchOption[] = recent
    .filter(({ text }) => !value.trim() || text.toLocaleLowerCase().includes(value.trim().toLocaleLowerCase()))
    .slice(0, MAX_RECENT_OPTIONS)
    .map((entry) => ({
      id: `recent:${entry.text}`,
      label: <code>{entry.text}</code>,
      accessibleLabel: entry.text,
      textSearch: !entry.clause,
      select: () => {
        if (entry.clause) applyClause(entry.clause);
        else runTextSearch(entry.text);
      },
    }));
  const recentTexts = new Set(recentOptions.map(({ accessibleLabel }) => accessibleLabel));
  const suggestions = suggestionOptions
    .filter(({ accessibleLabel }) => !recentTexts.has(accessibleLabel))
    .slice(0, MAX_OPTIONS - recentOptions.length);
  const placeholder = isAIMode
    ? aiPlaceholder
    : focused
      ? intl.formatMessage({
          defaultMessage: 'Type something to search run names',
          description: 'Focused evaluation runs search placeholder',
        })
      : intl.formatMessage({
          defaultMessage: 'Add a filter',
          description: 'Idle evaluation runs search placeholder',
        });
  const requestErrorMessage = requestError instanceof Error ? requestError.message : requestError?.renderHttpError();

  const chips = clauses.map((clause, index) => {
    const field = fieldForClause(fields, clause);
    const text = formatClause(fields, clause);
    const label = `${field ? getFieldLabel(field) : clause.field} ${operatorLabels[clause.operator]} ${formatValue(
      field,
      clause.value,
    )}`;
    return {
      text,
      label: <code>{label}</code>,
      onEdit: () => openEditor(index),
      onRemove: () => setClauses(clauses.filter((_, clauseIndex) => clauseIndex !== index)),
    };
  });
  return (
    <UnifiedSearch
      componentId="mlflow.eval-runs.unified-search.input"
      value={value}
      open={open}
      setOpen={setOpen}
      isAIMode={isAIMode}
      placeholder={placeholder ?? ''}
      ariaLabel={
        isAIMode
          ? (aiPlaceholder ?? '')
          : intl.formatMessage({
              defaultMessage: 'Search evaluation runs',
              description: 'Evaluation runs search input',
            })
      }
      submitDisabled={
        !value.trim() ||
        (!isAIMode &&
          (parsed.status === 'field' || parsed.status === 'operator') &&
          !isSearchRunsExpression(value.trim()))
      }
      onSubmit={submit}
      onChange={setValue}
      onClear={() => {
        setValue('');
        if (!isAIMode) {
          submittedSearch.current = undefined;
          onSearchFilterChange('');
        }
      }}
      onOpenFilter={() => openEditor()}
      onFocusChange={setFocused}
      onInputClick={() => setOpen(!isAIMode)}
      recentOptions={recentOptions}
      suggestions={suggestions.map((option) => ({ ...option, label: <code>{option.label}</code> }))}
      chips={chips}
      suffix={
        <div css={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          {requestErrorMessage ? (
            <Tooltip componentId="mlflow.eval-runs.unified-search.error" content={requestErrorMessage}>
              <InfoFillIcon aria-label={requestErrorMessage} css={{ color: theme.colors.textValidationDanger }} />
            </Tooltip>
          ) : null}
          {!isAIMode && (
            <InfoPopover
              ariaLabel={intl.formatMessage({
                defaultMessage: 'Search syntax help',
                description: 'Accessible label for the evaluation runs search syntax help',
              })}
              popoverProps={{ side: 'right' }}
            >
              <FormattedMessage
                defaultMessage="Type part of a run name to search by name, or enter a filter expression."
                description="Evaluation runs search help: the search box takes run names and filter expressions"
              />
              <br />
              <RunsSearchTooltipContent />
            </InfoPopover>
          )}
          {suffix}
        </div>
      }
    />
  );
};
