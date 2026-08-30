import { useMemo } from 'react';
import { useIntl, type IntlShape } from 'react-intl';
import {
  FilterOp,
  isClauseComplete,
  type FilterClause,
  type FilterFieldDef,
  type TraceFilterModel,
} from '@databricks/web-shared/traces-table';
import { escapeFilterValue } from './buildTracesV4SearchParams';

/**
 * MLflow's trace-filter field set + server-clause compilation. The neutral multi-clause AST (
 * `FilterClause`/`FilterOp`/`TraceFilterModel`) and the popover UI live in the shared
 * `@databricks/web-shared/traces-table`; MLflow owns exactly the fields it emits and how each clause
 * compiles to a search-API clause string (matching the shared `createMlflowSearchFilter`).
 *
 * Scope: the trace-attribute fields available on the V4 API (State, Duration, Trace name, Service
 * name, User, Session, Run name, Source, Input, Output, Span name/type/status), the arbitrary Tag /
 * Metadata key fields (free-text key + value, mirroring v1's `handleTagKey` / `handleMetadataKey`),
 * plus one "Assessment" field whose key is the assessment name, rendered as a combobox that suggests
 * the current page's candidate names and also accepts a freeform-typed name (the v3 pattern). Null
 * (IS_NULL/IS_NOT_NULL) assessment filters stay out of scope (backend-unsupported).
 */

const NUMERIC_OPERATORS: FilterOp[] = [
  FilterOp.EQUALS,
  FilterOp.NOT_EQUALS,
  FilterOp.GREATER_THAN,
  FilterOp.LESS_THAN,
  FilterOp.GREATER_THAN_OR_EQUALS,
  FilterOp.LESS_THAN_OR_EQUALS,
];

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
 * first operator is the default when a field is selected. A single "Assessment" field is appended
 * whose key (the assessment name) is a freeform-capable combobox suggesting the candidate names.
 */
export const useMlflowTraceFilterFields = (assessmentNames: string[] = []): FilterFieldDef[] => {
  const intl = useIntl();
  return useMemo(
    () => [
      {
        id: 'state',
        label: intl.formatMessage({ defaultMessage: 'State', description: 'Trace filter field: run state' }),
        operators: [FilterOp.EQUALS],
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
        valueInput: 'text',
      },
      {
        id: 'user',
        label: intl.formatMessage({ defaultMessage: 'User', description: 'Trace filter field: user' }),
        operators: [FilterOp.EQUALS],
        valueInput: 'text',
      },
      {
        id: 'session',
        label: intl.formatMessage({ defaultMessage: 'Session', description: 'Trace filter field: session id' }),
        operators: [FilterOp.EQUALS, FilterOp.CONTAINS],
        valueInput: 'text',
      },
      {
        id: 'run_name',
        label: intl.formatMessage({ defaultMessage: 'Run name', description: 'Trace filter field: run' }),
        operators: [FilterOp.EQUALS],
        valueInput: 'text',
      },
      {
        id: 'source',
        label: intl.formatMessage({ defaultMessage: 'Source', description: 'Trace filter field: source name' }),
        operators: [FilterOp.EQUALS, FilterOp.NOT_EQUALS, FilterOp.CONTAINS],
        valueInput: 'text',
      },
      {
        id: 'input',
        label: intl.formatMessage({ defaultMessage: 'Input', description: 'Trace filter field: request/input' }),
        operators: [FilterOp.CONTAINS],
        valueInput: 'text',
      },
      {
        id: 'output',
        label: intl.formatMessage({ defaultMessage: 'Output', description: 'Trace filter field: response/output' }),
        operators: [FilterOp.CONTAINS],
        valueInput: 'text',
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
        valueInput: 'text',
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
        valueInput: 'text',
        requiresKey: true,
        keyPlaceholder: intl.formatMessage({
          defaultMessage: 'Metadata key',
          description: 'Placeholder for the metadata-key input in the traces filter',
        }),
      },
      // One "Assessment" field: the key is the assessment name (combobox suggesting the candidate
      // names, freeform-typing allowed), free-text value, equality only (the managed V4 backend does
      // not support null assessment filters, and the value is opaque so comparison ops don't apply).
      {
        id: 'assessment',
        label: intl.formatMessage({ defaultMessage: 'Assessment', description: 'Trace filter field: assessment name' }),
        operators: [FilterOp.EQUALS, FilterOp.NOT_EQUALS],
        valueInput: 'text',
        requiresKey: true,
        keyInput: 'combobox',
        keyOptions: assessmentNames.map((name) => ({ value: name, label: name })),
        keyPlaceholder: intl.formatMessage({
          defaultMessage: 'Assessment name',
          description: 'Placeholder for the assessment-name key input in the traces filter',
        }),
      },
    ],
    [intl, assessmentNames],
  );
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

/**
 * Compile a single clause into a search-API clause string, matching the per-field handling in the
 * shared `createMlflowSearchFilter` (`useMlflowTraces.tsx`). Returns `undefined` for an incomplete
 * clause so it's dropped rather than emitted half-formed.
 */
const compileClause = (clause: FilterClause): string | undefined => {
  if (!isClauseComplete(clause)) {
    return undefined;
  }
  const { field, operator, value, key } = clause;
  // Every clause below wraps `value` in a single-quoted literal, so escape it once up front (the
  // numeric `duration` case doesn't quote and simply doesn't use this).
  const v = escapeFilterValue(value);
  switch (field) {
    case 'state':
      return `attributes.status = '${v}'`;
    case 'duration':
      return `attributes.execution_time_ms ${operator} ${value}`;
    case 'trace_name':
      return compileContainsAwareClause('attributes.name', operator, value);
    case 'service_name':
      return operator === FilterOp.CONTAINS
        ? `span.service_name LIKE '%${v}%'`
        : `span.service_name ${operator} '${v}'`;
    case 'user':
      return `request_metadata."mlflow.trace.user" = '${v}'`;
    case 'session':
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
      return `feedback.\`${key ?? ''}\` ${operator} '${v}'`;
    default:
      return undefined;
  }
};

/**
 * Compile the filter model into search-API clause strings (one per complete clause). These flow
 * through `buildFilter({ extraClauses })` unchanged, ANDed with the search + time-range clauses.
 */
export const compileFilterModel = (model: TraceFilterModel): string[] =>
  model.map(compileClause).filter((clause): clause is string => clause !== undefined);

/**
 * Compile URL-backed click-to-filter tag constraints into `tags.<key> = '<value>'` clauses. The key
 * is backticked only when it contains a `.` or a space, matching the shared `createMlflowSearchFilter`
 * tag clause. Structural `{ key, value }` input keeps this util free of a hook import.
 */
export const compileTagFilters = (tagFilters: { key: string; value: string }[]): string[] =>
  tagFilters.map(({ key, value }) => `${tagFieldRef(key)} = '${escapeFilterValue(value)}'`);
