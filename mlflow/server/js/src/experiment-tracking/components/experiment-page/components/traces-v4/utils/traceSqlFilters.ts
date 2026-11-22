import { type FilterFieldDef, type TraceFilterModel } from '@databricks/web-shared/traces-table';
import { compileFilterModel } from './filterModel';

const COLUMN_FIELDS = new Map(
  Object.entries({
    'attributes.status': 'state',
    'attributes.execution_time_ms': 'duration',
    'attributes.name': 'trace_name',
    'attributes.run_id': 'run_name',
    'request_metadata."mlflow.trace.user"': 'user',
    'request_metadata.mlflow.trace.session': 'session',
    'request_metadata."mlflow.source.name"': 'source',
    'span.name': 'span_name',
    'span.type': 'span_type',
    'span.status': 'span_status',
  }),
);

const KEYED_FIELDS = new Map(
  Object.entries({
    tags: 'tag',
    request_metadata: 'metadata',
    feedback: 'assessment',
    expectation: 'expectation',
  }),
);

/** Convert only predicates the current Basic controls can reproduce without changing their meaning. */
export const parseBasicTraceFilter = (
  sql: string,
  fields: FilterFieldDef[],
  format: (model: TraceFilterModel) => string = (model) => compileFilterModel(model).join(' AND '),
): TraceFilterModel | undefined => {
  if (!sql.trim()) return [];
  // Backslash-quote spellings differ across server tokenizers; preserve them in SQL.
  if (/\\['"]/.test(sql)) return undefined;
  const terms: string[] = [];
  let quote = '';
  let start = 0;
  for (let index = 0; index < sql.length; index++) {
    const char = sql[index];
    if (quote) {
      if (char === quote) {
        if (sql[index + 1] === quote) index++;
        else quote = '';
      }
    } else if (char === "'" || char === '"' || char === '`') {
      quote = char;
    } else if (/^\s+AND\s+/i.test(sql.slice(index))) {
      terms.push(sql.slice(start, index).trim());
      index += (sql.slice(index).match(/^\s+AND\s+/i)?.[0].length ?? 0) - 1;
      start = index + 1;
    }
  }
  if (quote) return undefined;
  terms.push(sql.slice(start).trim());

  const result: TraceFilterModel = [];
  for (const term of terms) {
    const match = term.match(/^((?:[\w.]|`[^`]*`|"[^"]*")+)\s+(ILIKE|!=|>=|<=|=|>|<)\s*(.*)$/i);
    if (!match) return undefined;
    const [, column, rawOperator, rawValue] = match;
    const string = rawValue.match(/^'((?:[^']|'')*)'$/);
    if (!string && !/^-?\d+(?:\.\d+)?$/.test(rawValue)) return undefined;
    const value = string ? string[1].replace(/''/g, "'") : rawValue;
    let field = COLUMN_FIELDS.get(column);
    let key: string | undefined;
    if (!field) {
      const keyed = column.match(/^(tags|request_metadata|feedback|expectation)\.(.+)$/);
      if (!keyed) return undefined;
      field = KEYED_FIELDS.get(keyed[1]) ?? '';
      key = keyed[2].replace(/^[`"]|[`"]$/g, '');
    }
    const definition = fields.find(({ id }) => id === field);
    if (!definition) return undefined;
    const operator = rawOperator.toUpperCase();
    // Try the actual Basic operators, including CONTAINS and the escaped span ILIKE forms.
    const values = [value];
    if (operator === 'ILIKE') {
      const literal = value.replace(/\\([\\%_[])/g, '$1');
      values.push(literal);
      if (value.startsWith('%') && value.endsWith('%')) {
        values.push(value.slice(1, -1), literal.slice(1, -1));
      }
    }
    const candidate = definition.operators
      .flatMap((basicOperator) =>
        values.map((basicValue) => ({ field, key, operator: basicOperator, value: basicValue })),
      )
      .find((clause) => format([clause]) === term);
    if (!candidate) return undefined;
    result.push(candidate);
  }
  return result;
};
