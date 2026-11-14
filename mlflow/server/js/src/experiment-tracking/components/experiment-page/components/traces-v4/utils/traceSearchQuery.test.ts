import { compileFilterModel } from './filterModel';
import type { ExpectationAssessment } from '@databricks/web-shared/model-trace-explorer';
import { describe, expect, test } from '@jest/globals';
import { FilterOp, type FilterFieldDef } from '@databricks/web-shared/traces-table';
import { makeFeedbackAssessment, makeTrace } from '../test-utils/mockTraces';
import { buildTraceSearchFields } from './traceSearchSuggestions';
import { getTraceSearchQuerySuggestions, parseTraceSearchQuery } from './traceSearchQuery';

const makeExpectation = (name: string, value: string | number | boolean): ExpectationAssessment => ({
  assessment_id: `expectation-${name}-${value}`,
  assessment_name: name,
  trace_id: 'trace-1',
  source: { source_type: 'HUMAN', source_id: 'test-user' },
  create_time: '2025-01-01T00:00:00Z',
  last_update_time: '2025-01-01T00:00:00Z',
  expectation: { value },
});

const fields: FilterFieldDef[] = [
  {
    id: 'duration',
    label: 'Duration',
    operators: [
      FilterOp.EQUALS,
      FilterOp.NOT_EQUALS,
      FilterOp.GREATER_THAN,
      FilterOp.LESS_THAN,
      FilterOp.GREATER_THAN_OR_EQUALS,
      FilterOp.LESS_THAN_OR_EQUALS,
    ],
    valueInput: 'number',
  },
  {
    id: 'state',
    label: 'State',
    operators: [FilterOp.EQUALS],
    valueInput: 'select',
    options: [
      { value: 'OK', label: 'OK' },
      { value: 'ERROR', label: 'Error' },
    ],
  },
  {
    id: 'assessment',
    label: 'Feedback',
    operators: [FilterOp.EQUALS],
    valueInput: 'text',
    requiresKey: true,
  },
  {
    id: 'expectation',
    label: 'Expectation',
    operators: [FilterOp.EQUALS],
    valueInput: 'text',
    requiresKey: true,
  },
  {
    id: 'rationale',
    label: 'Scorer rationale',
    operators: [FilterOp.CONTAINS],
    valueInput: 'text',
    requiresKey: true,
  },
  {
    id: 'tag',
    label: 'Tag',
    operators: [FilterOp.EQUALS, FilterOp.NOT_EQUALS],
    valueInput: 'text',
    requiresKey: true,
  },
];

const spanAttributeField: FilterFieldDef = {
  id: 'span_attribute',
  label: 'Span attribute',
  operators: [FilterOp.EQUALS, FilterOp.CONTAINS, FilterOp.RLIKE],
  valueInput: 'text',
  requiresKey: true,
};

const traces = [
  makeTrace('trace-1', {
    execution_duration: '2s',
    assessments: [
      makeFeedbackAssessment('groundedness score', 'very good', { rationale: 'The answer is supported' }),
      makeFeedbackAssessment('yes-key', 'yes'),
    ],
    tags: { env: 'prod', 'owner name': 'alice' },
  }),
];

const searchFields = buildTraceSearchFields(fields, traces);
const fieldsWithSpanAttributes = [...fields, spanAttributeField];
const searchFieldsWithSpanAttributes = buildTraceSearchFields(fieldsWithSpanAttributes, traces);

describe('parseTraceSearchQuery', () => {
  test('parses scorer rationale text as a structured filter', () => {
    const parsed = parseTraceSearchQuery('rationale."groundedness score" contains "supported"', fields, searchFields);
    expect(parsed).toMatchObject({
      status: 'complete',
      clause: {
        field: 'rationale',
        key: 'groundedness score',
        operator: FilterOp.CONTAINS,
        value: 'supported',
      },
    });
  });

  test('suggests and parses the attributes namespace without observed span keys', () => {
    expect(
      getTraceSearchQuerySuggestions('attributes', fieldsWithSpanAttributes, searchFieldsWithSpanAttributes),
    ).toEqual([
      {
        id: 'query-key:span_attribute:namespace',
        kind: 'key',
        label: 'attributes.',
        query: 'attributes.',
      },
    ]);

    const parsed = parseTraceSearchQuery(
      "attributes.sidekick.skills.name CONTAINS 'search'",
      fieldsWithSpanAttributes,
      searchFieldsWithSpanAttributes,
    );
    expect(parsed.clause).toEqual({
      field: 'span_attribute',
      key: 'sidekick.skills.name',
      operator: FilterOp.CONTAINS,
      value: 'search',
    });
  });

  test('parses quoted dotted keys and values into the existing filter clause shape', () => {
    expect(parseTraceSearchQuery('feedback."groundedness score" = "very good"', fields, searchFields)).toMatchObject({
      status: 'complete',
      clause: {
        field: 'assessment',
        key: 'groundedness score',
        operator: FilterOp.EQUALS,
        value: 'very good',
      },
    });
  });

  test('rejects a quoted span attribute key containing a backslash', () => {
    expect(
      parseTraceSearchQuery(
        String.raw`attributes."bad\key" = 'value'`,
        fieldsWithSpanAttributes,
        searchFieldsWithSpanAttributes,
      ),
    ).toMatchObject({
      status: 'invalid',
      stage: 'key',
      key: String.raw`bad\key`,
    });
  });

  test('narrows an unfinished quoted value using observed values', () => {
    const suggestions = getTraceSearchQuerySuggestions('feedback.yes-key = "y', fields, searchFields);

    expect(suggestions.some((suggestion) => suggestion.label.endsWith("= 'yes'"))).toBe(true);
    expect(suggestions.every((suggestion) => suggestion.kind === 'value' || suggestion.kind === 'apply')).toBe(true);
  });

  test('rejects duration equality operators using the narrowed search field definition', () => {
    expect(parseTraceSearchQuery('latency = 100', fields, searchFields)).toMatchObject({
      status: 'invalid',
      structured: true,
    });
    expect(parseTraceSearchQuery('latency != 100', fields, searchFields)).toMatchObject({
      status: 'invalid',
      structured: true,
    });
  });

  test('normalizes a supported latency range with a seconds unit', () => {
    expect(parseTraceSearchQuery('latency >= 2 s', fields, searchFields).clause).toEqual({
      field: 'duration',
      operator: FilterOp.GREATER_THAN_OR_EQUALS,
      value: '2000',
    });
  });

  test('recognizes an unknown operator without treating it as text search', () => {
    const parsed = parseTraceSearchQuery('feedback.groundedness ~= yes', fields, searchFields);
    expect(parsed).toMatchObject({
      status: 'invalid',
      recognized: true,
      structured: true,
    });

    const suggestions = getTraceSearchQuerySuggestions('feedback.groundedness ~= yes', fields, searchFields);
    expect(suggestions.some((suggestion) => suggestion.label === 'feedback.groundedness =')).toBe(true);
    expect(suggestions.some((suggestion) => suggestion.label.includes('~='))).toBe(false);
  });

  test('uses canonical syntax in keyed and value suggestions', () => {
    const suggestions = getTraceSearchQuerySuggestions('tag.', fields, searchFields);
    expect(suggestions[0]).toMatchObject({
      label: 'tag.env',
      query: 'tag.env',
    });

    const valueSuggestions = getTraceSearchQuerySuggestions('tag.env =', fields, searchFields);
    expect(valueSuggestions.some((suggestion) => suggestion.label === "tag.env = 'prod'")).toBe(true);
    expect(valueSuggestions.some((suggestion) => suggestion.query === "tag.env = 'prod'")).toBe(true);
  });

  test('caps keyed completions at five choices', () => {
    const manyKeys = Array.from({ length: 7 }, (_, index) => makeFeedbackAssessment(`key-${index}`, 'value'));
    const manySearchFields = buildTraceSearchFields(fields, [makeTrace('many-keys', { assessments: manyKeys })]);

    expect(getTraceSearchQuerySuggestions('feedback.', fields, manySearchFields)).toHaveLength(5);
  });

  test('does not expose cost as a query-language field', () => {
    expect(
      getTraceSearchQuerySuggestions('cost', [...fields, { ...fields[0], id: 'cost', label: 'Cost' }], searchFields),
    ).toEqual([]);
  });
});

describe('feedback and expectation query namespaces', () => {
  const sharedFields = buildTraceSearchFields(fields, [
    makeTrace('shared', {
      assessments: [makeFeedbackAssessment('quality', 'pass'), makeExpectation('quality', 'gold')],
    }),
  ]);

  test.each([
    ['feedback', 'assessment', 'pass', 'gold'],
    ['expectation', 'expectation', 'gold', 'pass'],
  ])('keeps %s values and compiled clauses separate', (namespace, field, value, otherValue) => {
    const suggestions = getTraceSearchQuerySuggestions(`${namespace}.quality =`, fields, sharedFields);
    expect(suggestions.map((suggestion) => suggestion.clause?.value)).toEqual([value]);
    expect(suggestions.some((suggestion) => suggestion.label.includes(otherValue))).toBe(false);
    const parsed = parseTraceSearchQuery(`${namespace}.quality = '${value}'`, fields, sharedFields);
    expect(parsed.clause).toEqual({
      field,
      key: 'quality',
      operator: FilterOp.EQUALS,
      value,
    });
    expect(compileFilterModel([parsed.clause!])).toEqual([`${namespace}.\`quality\` = '${value}'`]);
    expect(suggestions[0].clause).toEqual(parsed.clause);
  });

  test.each(['feedback', 'expectation'])('parses quoted and unobserved %s keys without rows', (namespace) => {
    const parsed = parseTraceSearchQuery(`${namespace}."judge.score name" = "very good"`, fields, []);
    expect(parsed.status).toBe('complete');
    expect(compileFilterModel([parsed.clause!])).toEqual([`${namespace}.\`judge.score name\` = 'very good'`]);
  });

  test('rejects the internal assessment ID as public syntax', () => {
    expect(parseTraceSearchQuery('assessment.quality = pass', fields, sharedFields)).toMatchObject({
      status: 'invalid',
      structured: true,
    });
    expect(getTraceSearchQuerySuggestions('assessment.', fields, sharedFields)).toEqual([]);
  });

  test.each(['feedback.', 'feedback.quality', 'feedback.quality =', 'expectation.', 'expectation.quality ='])(
    'never displays assessment suggestions for %s',
    (query) => {
      const suggestions = getTraceSearchQuerySuggestions(query, fields, sharedFields);
      expect(suggestions.length).toBeGreaterThan(0);
      expect(suggestions.every(({ label, query: completion }) => !/assessment/i.test(label + completion))).toBe(true);
    },
  );
});

describe('text-only progressive completions', () => {
  test.each(['latency', 'latency '])('offers operators immediately for %s', (query) => {
    expect(
      getTraceSearchQuerySuggestions(query, fields, searchFields).map(({ query: completion }) => completion),
    ).toEqual(['latency > ', 'latency < ', 'latency >= ', 'latency <= ']);
    expect(parseTraceSearchQuery(query, fields, searchFields)).toMatchObject({
      status: 'partial',
      structured: true,
      bareField: true,
    });
  });

  test.each(['state', 'feedback', 'tag'])('marks exact field alias %s as literal-search eligible', (query) => {
    expect(parseTraceSearchQuery(query, fields, searchFields)).toMatchObject({
      status: 'partial',
      structured: true,
      bareField: true,
    });
  });

  test.each(['latency >', 'latency >= 2 s', 'feedback.', 'feedback.quality', 'state ='])(
    'does not mark structured syntax %s as a bare field',
    (query) => {
      expect(parseTraceSearchQuery(query, fields, searchFields).bareField).toBeFalsy();
    },
  );

  test('offers observed keys immediately for a complete namespace', () => {
    expect(getTraceSearchQuerySuggestions('feedback', fields, searchFields).map(({ query }) => query)).toEqual([
      "feedback.'groundedness score'",
      'feedback.yes-key',
    ]);
  });

  test.each(['feedback', 'expectation'])('completes a partial inequality operator for %s', (namespace) => {
    const enabledFields = fields.map((field) =>
      field.requiresKey ? { ...field, operators: [FilterOp.EQUALS, FilterOp.NOT_EQUALS] } : field,
    );
    const enabledSearchFields = buildTraceSearchFields(enabledFields, traces);
    expect(
      getTraceSearchQuerySuggestions(`${namespace}.score !`, enabledFields, enabledSearchFields).map(
        ({ query }) => query,
      ),
    ).toEqual([`${namespace}.score != `]);
    expect(parseTraceSearchQuery(`${namespace}.score IS NULL`, enabledFields, enabledSearchFields)).toMatchObject({
      status: 'invalid',
      stage: 'operator',
    });
    const parsed = parseTraceSearchQuery(`${namespace}.score != 'pass'`, enabledFields, enabledSearchFields);
    expect(compileFilterModel([parsed.clause!])).toEqual([`${namespace}.\`score\` != 'pass'`]);
  });
});
