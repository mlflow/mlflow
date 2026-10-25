import { useMemo } from 'react';
import { useIntl, type IntlShape } from 'react-intl';
import {
  FilterOp,
  isClauseComplete,
  type FilterClause,
  type FilterFieldDef,
  type TraceFilterModel,
} from '@databricks/web-shared/traces-table';
import type { ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';
import { shouldEnableNewEvalRunsUi } from '@mlflow/mlflow/src/common/utils/FeatureUtils';
import { escapeFilterValue } from './buildTracesV4SearchParams';
import { assessmentValueKindKey } from './assessmentColumns';
import {
  buildTraceFilterSuggestions,
  getTraceFilterValueOptions,
  type TraceFilterSuggestions,
} from './filterSuggestions';

/**
 * MLflow's trace-filter field set + server-clause compilation. The neutral multi-clause AST (
 * `FilterClause`/`FilterOp`/`TraceFilterModel`) and the popover UI live in the shared
 * `@databricks/web-shared/traces-table`; MLflow owns exactly the fields it emits and how each clause
 * compiles to a search-API clause string (matching the shared `createMlflowSearchFilter`).
 *
 * Scope: the trace-attribute fields available on the V4 API (State, Duration, Trace name, Service
 * name, User, Session, Run name, Source, Input, Output, Span name/type/status), the arbitrary Tag /
 * Metadata key fields (free-text key + value, mirroring v1's `handleTagKey` / `handleMetadataKey`),
 * plus "Feedback" and "Expectation" fields whose keys are rendered as comboboxes that suggest
 * the current page's candidate names and also accept a freeform-typed name (the v3 pattern). Null
 * (IS_NULL/IS_NOT_NULL) filters stay out of scope.
 *
 * With the new evaluation runs UI, State, User and Session also offer `!=`, value inputs suggest the
 * values on the loaded rows (while still accepting a typed value), and numeric feedback and
 * expectation values compare as numbers.
 */

const NUMERIC_OPERATORS: FilterOp[] = [
  FilterOp.EQUALS,
  FilterOp.NOT_EQUALS,
  FilterOp.GREATER_THAN,
  FilterOp.LESS_THAN,
  FilterOp.GREATER_THAN_OR_EQUALS,
  FilterOp.LESS_THAN_OR_EQUALS,
];

/** Session operators with the new evaluation runs UI (the server has no null checks for them here). */
const SESSION_OPERATORS: FilterOp[] = [FilterOp.EQUALS, FilterOp.NOT_EQUALS, FilterOp.CONTAINS];

/** Selectable trace states (value = the token the API matches on). */
const STATE_VALUES = ['OK', 'ERROR', 'IN_PROGRESS'] as const;

const stateLabel = (intl: IntlShape, value: string): string => {
  switch (value) {
    case 'OK':
      return intl.formatMessage({ defaultMessage: 'OK', description: 'Trace state filter value: successful' });
    case 'ERROR':
      return intl.formatMessage({ defaultMessage: 'Error', description: 'Trace state filter value: errored' });
    case 'IN_PROGRESS':
      return intl.formatMessage({ defaultMessage: 'In progress', description: 'Trace state filter value: running' });
    default:
      return value;
  }
};

/**
 * The MLflow trace filter fields, in dropdown order, with localized labels — fed to the shared
 * `TraceFilterButton`. Operators per field mirror v1's `getAvailableOperators` for the V4 path; the
 * first operator is the default when a field is selected. A single "Feedback" field is appended
 * whose key (the feedback name) is a freeform-capable combobox suggesting the candidate names.
 */
export const useMlflowTraceFilterFields = (
  assessmentNames: string[] = [],
  /** Rows on the current page, used for value suggestions with the new evaluation runs UI. */
  traces?: ModelTraceInfoV3[],
): FilterFieldDef[] => {
  const intl = useIntl();
  const enhanced = shouldEnableNewEvalRunsUi();
  const suggestionTraces = enhanced ? traces : undefined;
  const traceSuggestions = useMemo<TraceFilterSuggestions | undefined>(
    () => (suggestionTraces !== undefined ? buildTraceFilterSuggestions(suggestionTraces) : undefined),
    [suggestionTraces],
  );
  const availableAssessmentNames = useMemo(
    () =>
      [
        ...new Set([
          ...assessmentNames.filter((name) => !traceSuggestions?.keyedValues.expectation.valuesByKey.has(name)),
          ...(traceSuggestions?.assessmentNames.map(({ value }) => value) ?? []),
        ]),
      ]
        .sort((left, right) => left.localeCompare(right))
        .map((name) => ({ value: name, label: name })),
    [assessmentNames, traceSuggestions],
  );
  return useMemo(() => {
    const scalarValueInput = (field: keyof TraceFilterSuggestions['values']) =>
      enhanced
        ? { valueInput: 'combobox' as const, options: traceSuggestions?.values[field] ?? [] }
        : { valueInput: 'text' as const };
    const keyedValueInput = (field: 'tag' | 'metadata' | 'assessment' | 'expectation') =>
      enhanced
        ? {
            valueInput: 'combobox' as const,
            getValueOptions: (key: string) =>
              traceSuggestions ? getTraceFilterValueOptions(traceSuggestions, field, key) : [],
          }
        : { valueInput: 'text' as const };
    const keyedFieldInput = (field: 'tag' | 'metadata') =>
      enhanced ? { keyInput: 'combobox' as const, keyOptions: traceSuggestions?.keyedValues[field].keys ?? [] } : {};

    return [
      {
        id: 'state',
        label: intl.formatMessage({ defaultMessage: 'State', description: 'Trace filter field: run state' }),
        operators: enhanced ? [FilterOp.EQUALS, FilterOp.NOT_EQUALS] : [FilterOp.EQUALS],
        valueInput: 'select',
        options: STATE_VALUES.map((value) => ({ value, label: stateLabel(intl, value) })),
      },
      {
        id: 'duration',
        label: intl.formatMessage({ defaultMessage: 'Duration', description: 'Trace filter field: execution time' }),
        operators: NUMERIC_OPERATORS,
        valueInput: 'number',
        valuePlaceholder: intl.formatMessage({
          defaultMessage: 'Time in milliseconds',
          description: 'Placeholder for the duration filter value input',
        }),
      },
      {
        id: 'trace_name',
        label: intl.formatMessage({ defaultMessage: 'Trace name', description: 'Trace filter field: trace name' }),
        operators: [FilterOp.EQUALS, FilterOp.NOT_EQUALS, FilterOp.CONTAINS],
        ...scalarValueInput('trace_name'),
      },
      {
        id: 'service_name',
        label: intl.formatMessage({
          defaultMessage: 'Service name',
          description: 'Trace filter field: service name',
        }),
        operators: [FilterOp.EQUALS, FilterOp.NOT_EQUALS, FilterOp.CONTAINS],
        valueInput: 'text',
      },
      {
        id: 'user',
        label: intl.formatMessage({ defaultMessage: 'User', description: 'Trace filter field: user' }),
        operators: enhanced ? [FilterOp.EQUALS, FilterOp.NOT_EQUALS] : [FilterOp.EQUALS],
        ...scalarValueInput('user'),
      },
      {
        id: 'session',
        label: intl.formatMessage({ defaultMessage: 'Session', description: 'Trace filter field: session id' }),
        operators: enhanced ? SESSION_OPERATORS : [FilterOp.EQUALS, FilterOp.CONTAINS],
        ...scalarValueInput('session'),
      },
      {
        id: 'run_name',
        label: intl.formatMessage({ defaultMessage: 'Run name', description: 'Trace filter field: run' }),
        operators: [FilterOp.EQUALS],
        ...scalarValueInput('run_name'),
        ...(enhanced
          ? {
              valuePlaceholder: intl.formatMessage({
                defaultMessage: 'Select or type a run ID',
                description: 'Run filter accepts a suggested run ID or a typed one',
              }),
            }
          : {}),
      },
      {
        id: 'source',
        label: intl.formatMessage({ defaultMessage: 'Source', description: 'Trace filter field: source name' }),
        operators: [FilterOp.EQUALS, FilterOp.NOT_EQUALS, FilterOp.CONTAINS],
        ...scalarValueInput('source'),
      },
      {
        id: 'input',
        label: intl.formatMessage({ defaultMessage: 'Input', description: 'Trace filter field: request/input' }),
        operators: [FilterOp.CONTAINS],
        ...scalarValueInput('input'),
      },
      {
        id: 'output',
        label: intl.formatMessage({ defaultMessage: 'Output', description: 'Trace filter field: response/output' }),
        operators: [FilterOp.CONTAINS],
        ...scalarValueInput('output'),
      },
      {
        id: 'span_name',
        label: intl.formatMessage({ defaultMessage: 'Span name', description: 'Trace filter field: span name' }),
        operators: [FilterOp.EQUALS, FilterOp.NOT_EQUALS, FilterOp.CONTAINS],
        valueInput: 'text',
      },
      {
        id: 'span_type',
        label: intl.formatMessage({ defaultMessage: 'Span type', description: 'Trace filter field: span type' }),
        operators: [FilterOp.EQUALS, FilterOp.NOT_EQUALS, FilterOp.CONTAINS],
        valueInput: 'text',
      },
      {
        id: 'span_status',
        label: intl.formatMessage({ defaultMessage: 'Span status', description: 'Trace filter field: span status' }),
        operators: [FilterOp.EQUALS, FilterOp.NOT_EQUALS],
        valueInput: 'text',
      },
      // Arbitrary tag / trace-metadata key filters (v1's `handleTagKey` / `handleMetadataKey`): the
      // user types the key and value, so these carry a free-text key sub-input (`requiresKey`).
      {
        id: 'tag',
        label: intl.formatMessage({ defaultMessage: 'Tag', description: 'Trace filter field: arbitrary tag key' }),
        operators: [FilterOp.EQUALS, FilterOp.NOT_EQUALS],
        ...keyedValueInput('tag'),
        ...keyedFieldInput('tag'),
        requiresKey: true,
        keyPlaceholder: intl.formatMessage({
          defaultMessage: 'Tag key',
          description: 'Placeholder for the tag-key input in the traces filter',
        }),
      },
      {
        id: 'metadata',
        label: intl.formatMessage({
          defaultMessage: 'Metadata',
          description: 'Trace filter field: trace metadata key',
        }),
        operators: [FilterOp.EQUALS, FilterOp.NOT_EQUALS, FilterOp.CONTAINS],
        ...keyedValueInput('metadata'),
        ...keyedFieldInput('metadata'),
        requiresKey: true,
        keyPlaceholder: intl.formatMessage({
          defaultMessage: 'Metadata key',
          description: 'Placeholder for the metadata-key input in the traces filter',
        }),
      },
      // "Feedback" and "Expectation" fields: their keys are names (combobox suggesting the
      // candidate names, freeform-typing allowed), free-text values, and equality-only because the
      // managed V4 backend does not support null filters and their values are opaque.
      {
        id: 'assessment', // Stable for persisted filter models and v3 view compatibility.
        label: intl.formatMessage({
          defaultMessage: 'Feedback',
          description: 'Trace filter field: feedback assessment',
        }),
        operators: [FilterOp.EQUALS, FilterOp.NOT_EQUALS],
        ...keyedValueInput('assessment'),
        requiresKey: true,
        keyInput: 'combobox',
        keyOptions: enhanced ? availableAssessmentNames : assessmentNames.map((name) => ({ value: name, label: name })),
        keyPlaceholder: intl.formatMessage({
          defaultMessage: 'Feedback name',
          description: 'Placeholder for the feedback-name key input in the traces filter',
        }),
      },
      {
        id: 'expectation',
        label: intl.formatMessage({
          defaultMessage: 'Expectation',
          description: 'Trace filter field: expectation name',
        }),
        operators: [FilterOp.EQUALS, FilterOp.NOT_EQUALS],
        ...keyedValueInput('expectation'),
        requiresKey: true,
        keyInput: 'combobox',
        keyOptions:
          enhanced && traceSuggestions
            ? traceSuggestions.keyedValues.expectation.keys
            : assessmentNames.map((name) => ({ value: name, label: name })),
        keyPlaceholder: intl.formatMessage({
          defaultMessage: 'Expectation name',
          description: 'Placeholder for the expectation-name key input in the traces filter',
        }),
      },
    ];
  }, [intl, assessmentNames, enhanced, traceSuggestions, availableAssessmentNames]);
};

/**
 * Validate a persisted filter clause against the current field set: the field still exists, still
 * offers the clause's operator, and (for a `requiresKey` field) carries a non-blank key. Used when
 * restoring a saved view so a clause referencing a field/operator that no longer exists is dropped
 * rather than silently producing wrong results — and, symmetrically, so the dirty diff normalizes
 * the stored baseline the same way (an unsupported clause can't strand a view permanently dirty).
 */
export const isSupportedFilterClause = (fields: FilterFieldDef[], clause: FilterClause): boolean =>
  fields.some(
    (field) =>
      field.id === clause.field &&
      field.operators.includes(clause.operator) &&
      (!field.requiresKey || (typeof clause.key === 'string' && clause.key.trim() !== '')),
  );

/**
 * Compile a text field where only `CONTAINS` needs translation (to `ILIKE '%value%'`, since the
 * backend has no `CONTAINS` token); every other operator passes through as `field OP 'value'`.
 */
const compileContainsAwareClause = (field: string, operator: FilterOp, value: string): string => {
  const escaped = escapeFilterValue(value);
  return operator === FilterOp.CONTAINS ? `${field} ILIKE '%${escaped}%'` : `${field} ${operator} '${escaped}'`;
};

/**
 * Compile a span text field (name/type) clause, matching v3's span handling: the backend has no
 * `CONTAINS` token, so `EQUALS` becomes a case-insensitive exact `ILIKE 'value'` and `CONTAINS` a
 * substring `ILIKE '%value%'`; `NOT_EQUALS` passes through as `!=`.
 */
const compileSpanTextClause = (field: string, operator: FilterOp, value: string): string =>
  operator === FilterOp.EQUALS
    ? `${field} ILIKE '${escapeFilterValue(value)}'`
    : compileContainsAwareClause(field, operator, value);

/**
 * The `tags.<key>` field reference, backticked only when the key contains a `.` or a space (matching
 * the shared `createMlflowSearchFilter` tag clause), so dotted/spaced keys survive.
 */
const tagFieldRef = (key: string): string =>
  key.includes('.') || key.includes(' ') ? `tags.\`${key}\`` : `tags.${key}`;

const NO_NUMERIC_ASSESSMENT_KEYS: ReadonlySet<string> = new Set();

const NUMERIC_LITERAL = /^-?\d+(\.\d+)?$/;

/**
 * Compile a feedback or expectation clause. The server compares `=`/`!=` against the stored value's
 * text, so a number logged as `2.0` doesn't match `'2'`. For an assessment known to hold numbers
 * (`isNumeric`), a numeric `=` therefore compiles to a `>= n AND <= n` range, which the server
 * compares as numbers, and a numeric `!=` excludes both of a whole number's spellings (`'2'` and
 * `'2.0'`). Anything else, including an assessment whose type isn't known yet, keeps the literal
 * comparison, so a text value such as `'5'` still matches exactly.
 */
const compileAssessmentClause = (fieldRef: string, operator: FilterOp, value: string, isNumeric: boolean): string => {
  const trimmed = value.trim();
  if (
    isNumeric &&
    NUMERIC_LITERAL.test(trimmed) &&
    (operator === FilterOp.EQUALS || operator === FilterOp.NOT_EQUALS)
  ) {
    const number = Number(trimmed);
    if (operator === FilterOp.EQUALS) {
      // The server checks each half against any assessment with the name, so a trace with 3 from one run
      // and 5 from another also matches `= 4`. There's no single-assessment form to send instead (no `IN`
      // for assessments, and `RLIKE` fails on MSSQL). The evaluation run page matches these filters itself.
      return `${fieldRef} >= ${number} AND ${fieldRef} <= ${number}`;
    }
    const spellings = Number.isInteger(number) ? [`${number}`, `${number}.0`] : [`${number}`];
    return spellings.map((spelling) => `${fieldRef} != '${spelling}'`).join(' AND ');
  }
  return `${fieldRef} ${operator} '${escapeFilterValue(value)}'`;
};

/**
 * Compile a single clause into a search-API clause string, matching the per-field handling in the
 * shared `createMlflowSearchFilter` (`useMlflowTraces.tsx`). Returns `undefined` for an incomplete
 * clause so it's dropped rather than emitted half-formed. `enhanced` is the new evaluation runs UI.
 */
const compileClause = (
  clause: FilterClause,
  enhanced: boolean,
  numericAssessmentKeys: ReadonlySet<string>,
): string | undefined => {
  if (!isClauseComplete(clause)) {
    return undefined;
  }
  const { field, operator, value, key } = clause;
  // Every clause below wraps `value` in a single-quoted literal, so escape it once up front (the
  // numeric `duration` case doesn't quote and simply doesn't use this).
  const v = escapeFilterValue(value);
  switch (field) {
    case 'state':
      return enhanced ? `attributes.status ${operator} '${v}'` : `attributes.status = '${v}'`;
    case 'duration':
      return `attributes.execution_time_ms ${operator} ${value}`;
    case 'trace_name':
      return compileContainsAwareClause('attributes.name', operator, value);
    case 'service_name':
      return operator === FilterOp.CONTAINS
        ? `span.service_name LIKE '%${v}%'`
        : `span.service_name ${operator} '${v}'`;
    case 'user':
      return enhanced
        ? compileContainsAwareClause('request_metadata."mlflow.trace.user"', operator, value)
        : `request_metadata."mlflow.trace.user" = '${v}'`;
    case 'session':
      if (enhanced) {
        return compileContainsAwareClause('request_metadata.mlflow.trace.session', operator, value);
      }
      return operator === FilterOp.CONTAINS
        ? `request_metadata.mlflow.trace.session ILIKE '%${v}%'`
        : `request_metadata.mlflow.trace.session = '${v}'`;
    case 'run_name':
      return `attributes.run_id = '${v}'`;
    case 'source':
      return compileContainsAwareClause('request_metadata."mlflow.source.name"', operator, value);
    case 'input':
      return `span.content ILIKE '%${v}%'`;
    case 'output':
      return `span.content ILIKE '%${v}%'`;
    case 'span_name':
      return compileSpanTextClause('span.name', operator, value);
    case 'span_type':
      // v3 uses `span.type` for the span type column.
      return compileSpanTextClause('span.type', operator, value);
    case 'span_status':
      return `span.status ${operator} '${v}'`;
    case 'tag':
      // Arbitrary tag key=value; `isClauseComplete` guarantees a non-blank key here.
      return `${tagFieldRef(key ?? '')} ${operator} '${v}'`;
    case 'metadata':
      // Arbitrary trace-metadata key, matching v3's custom-metadata clause (CONTAINS → ILIKE).
      return compileContainsAwareClause(`request_metadata.${key ?? ''}`, operator, value);
    case 'assessment':
      // The key is the assessment name; `isClauseComplete` guarantees it's non-blank (requiresKey).
      // Compiles to a `feedback.\`<name>\`` clause (matching the shared `createMlflowSearchFilter`);
      // the name is backtick-escaped so dots/spaces survive.
      return enhanced
        ? compileAssessmentClause(
            `feedback.\`${key ?? ''}\``,
            operator,
            value,
            numericAssessmentKeys.has(assessmentValueKindKey('feedback', key ?? '')),
          )
        : `feedback.\`${key ?? ''}\` ${operator} '${v}'`;
    case 'expectation':
      // The key is the expectation name; `isClauseComplete` guarantees it's non-blank (requiresKey).
      // Compiles to an `expectation.\`<name>\`` clause, matching the trace-search DSL.
      return enhanced
        ? compileAssessmentClause(
            `expectation.\`${key ?? ''}\``,
            operator,
            value,
            numericAssessmentKeys.has(assessmentValueKindKey('expectation', key ?? '')),
          )
        : `expectation.\`${key ?? ''}\` ${operator} '${v}'`;
    default:
      return undefined;
  }
};

/**
 * Compile the filter model into search-API clause strings (one per complete clause). These flow
 * through `buildFilter({ extraClauses })` unchanged, ANDed with the search + time-range clauses.
 */
export const compileFilterModel = (
  model: TraceFilterModel,
  /**
   * Feedback and expectation names known to hold numbers (see `useNumericAssessmentKeys`); only these
   * compare as numbers with the new evaluation runs UI.
   */
  numericAssessmentKeys: ReadonlySet<string> = NO_NUMERIC_ASSESSMENT_KEYS,
): string[] => {
  const enhanced = shouldEnableNewEvalRunsUi();
  return model
    .map((clause) => compileClause(clause, enhanced, numericAssessmentKeys))
    .filter((clause): clause is string => clause !== undefined);
};

/**
 * Compile URL-backed click-to-filter tag constraints into `tags.<key> = '<value>'` clauses. The key
 * is backticked only when it contains a `.` or a space, matching the shared `createMlflowSearchFilter`
 * tag clause. Structural `{ key, value }` input keeps this util free of a hook import.
 */
export const compileTagFilters = (tagFilters: { key: string; value: string }[]): string[] =>
  tagFilters.map(({ key, value }) => `${tagFieldRef(key)} = '${escapeFilterValue(value)}'`);
