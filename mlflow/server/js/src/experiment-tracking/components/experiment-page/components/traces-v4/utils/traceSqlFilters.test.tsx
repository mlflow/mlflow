import { describe, expect, test } from '@jest/globals';
import { renderHook } from '@testing-library/react';
import { IntlProvider } from 'react-intl';
import { FilterOp } from '@databricks/web-shared/traces-table';
import { compileFilterModel, useMlflowTraceFilterFields } from './filterModel';
import { parseBasicTraceFilter } from './traceSqlFilters';

describe('SQL to Basic conversion', () => {
  const fields = () =>
    renderHook(() => useMlflowTraceFilterFields(), {
      wrapper: ({ children }) => <IntlProvider locale="en">{children}</IntlProvider>,
    }).result.current;

  test('round trips multiple conditions and quoted AND without losing literal case', () => {
    const model = [
      { field: 'duration', operator: FilterOp.GREATER_THAN, value: '1000' },
      { field: 'tag', key: 'release name', operator: FilterOp.EQUALS, value: "O'Reilly AND QA" },
      { field: 'metadata', key: 'team owner', operator: FilterOp.EQUALS, value: 'Alice' },
    ];
    const sql = compileFilterModel(model).join(' AND ');
    const basic = parseBasicTraceFilter(sql, fields());
    expect(basic).toEqual(model);
    expect(compileFilterModel(basic ?? []).join(' AND ')).toBe(sql);
  });

  test.each([
    "attributes.status IN ('OK', 'ERROR')",
    "span.name RLIKE 'checkout.*'",
    "attributes.status = 'OK' OR attributes.status = 'ERROR'",
    "(tags.env = 'qa' OR tags.env = 'prod')",
    'feedback.`quality` >= 2.5',
    'tags.env IS NULL',
    'SELECT * FROM traces',
    "attributes.name = 'unfinished",
    'attributes.execution_time_ms > 1.5',
  ])('leaves a nonrepresentable predicate in SQL: %s', (sql) => {
    expect(parseBasicTraceFilter(sql, fields())).toBeUndefined();
  });

  test('retains SQL when reserialization would change comparison semantics', () => {
    // Basic span equality is case-insensitive and escapes LIKE wildcards.
    expect(parseBasicTraceFilter("span.name = 'A_B'", fields())).toBeUndefined();
    expect(parseBasicTraceFilter("span.name ILIKE 'A_B'", fields())).toBeUndefined();
    const sql = compileFilterModel([{ field: 'span_name', operator: FilterOp.CONTAINS, value: 'A_B' }]).join(' AND ');
    expect(compileFilterModel(parseBasicTraceFilter(sql, fields()) ?? []).join(' AND ')).toBe(sql);
  });

  test('empty SQL clears filters', () => {
    expect(parseBasicTraceFilter('  ', fields())).toEqual([]);
  });
});
