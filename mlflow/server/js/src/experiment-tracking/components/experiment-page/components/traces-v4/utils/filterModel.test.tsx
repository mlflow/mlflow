import { describe, expect, test } from '@jest/globals';
import { renderHook } from '@testing-library/react';
import { IntlProvider } from 'react-intl';
import { FilterOp, type FilterClause } from '@databricks/web-shared/traces-table';
import { compileFilterModel, compileTagFilters, useMlflowTraceFilterFields } from './filterModel';

describe('compileFilterModel', () => {
  test('the empty model compiles to no clauses', () => {
    expect(compileFilterModel([])).toEqual([]);
  });

  test('an incomplete clause (empty value) drops from compilation', () => {
    expect(compileFilterModel([{ field: 'state', operator: FilterOp.EQUALS, value: '' }])).toEqual([]);
  });

  describe('per-field clause compilation (matches createMlflowSearchFilter)', () => {
    test.each<{ name: string; clause: FilterClause; expected: string }>([
      {
        name: 'state → status equality',
        clause: { field: 'state', operator: FilterOp.EQUALS, value: 'ERROR' },
        expected: "attributes.status = 'ERROR'",
      },
      {
        name: 'duration → numeric execution_time_ms with the chosen operator',
        clause: { field: 'duration', operator: FilterOp.GREATER_THAN, value: '1000' },
        expected: 'attributes.execution_time_ms > 1000',
      },
      {
        name: 'trace name equals → attributes.name equality',
        clause: { field: 'trace_name', operator: FilterOp.EQUALS, value: 'chat' },
        expected: "attributes.name = 'chat'",
      },
      {
        name: 'trace name contains → attributes.name ILIKE substring (backend has no CONTAINS token)',
        clause: { field: 'trace_name', operator: FilterOp.CONTAINS, value: 'chat' },
        expected: "attributes.name ILIKE '%chat%'",
      },
      {
        name: 'user → quoted request_metadata user',
        clause: { field: 'user', operator: FilterOp.EQUALS, value: 'alice' },
        expected: `request_metadata."mlflow.trace.user" = 'alice'`,
      },
      {
        name: 'session equality → dotted session metadata',
        clause: { field: 'session', operator: FilterOp.EQUALS, value: 'sess-1' },
        expected: "request_metadata.mlflow.trace.session = 'sess-1'",
      },
      {
        name: 'session contains → ILIKE on session metadata',
        clause: { field: 'session', operator: FilterOp.CONTAINS, value: 'sess' },
        expected: "request_metadata.mlflow.trace.session ILIKE '%sess%'",
      },
      {
        name: 'run name → run_id equality',
        clause: { field: 'run_name', operator: FilterOp.EQUALS, value: 'run-9' },
        expected: "attributes.run_id = 'run-9'",
      },
      {
        name: 'source → quoted source name with the chosen operator',
        clause: { field: 'source', operator: FilterOp.NOT_EQUALS, value: 'app.py' },
        expected: `request_metadata."mlflow.source.name" != 'app.py'`,
      },
      {
        name: 'source contains → ILIKE substring (backend has no CONTAINS token)',
        clause: { field: 'source', operator: FilterOp.CONTAINS, value: 'app' },
        expected: `request_metadata."mlflow.source.name" ILIKE '%app%'`,
      },
      {
        name: 'input → span.content ILIKE substring',
        clause: { field: 'input', operator: FilterOp.CONTAINS, value: 'hello' },
        expected: "span.content ILIKE '%hello%'",
      },
      {
        name: 'output → span.content ILIKE substring',
        clause: { field: 'output', operator: FilterOp.CONTAINS, value: 'world' },
        expected: "span.content ILIKE '%world%'",
      },
      {
        name: 'span name equals → span.name ILIKE exact (case-insensitive, no CONTAINS token)',
        clause: { field: 'span_name', operator: FilterOp.EQUALS, value: 'my_span' },
        expected: "span.name ILIKE 'my_span'",
      },
      {
        name: 'span name contains → span.name ILIKE substring',
        clause: { field: 'span_name', operator: FilterOp.CONTAINS, value: 'span' },
        expected: "span.name ILIKE '%span%'",
      },
      {
        name: 'span name not-equals → span.name !=',
        clause: { field: 'span_name', operator: FilterOp.NOT_EQUALS, value: 'excluded' },
        expected: "span.name != 'excluded'",
      },
      {
        name: 'span type equals → span.type ILIKE exact',
        clause: { field: 'span_type', operator: FilterOp.EQUALS, value: 'LLM' },
        expected: "span.type ILIKE 'LLM'",
      },
      {
        name: 'span type contains → span.type ILIKE substring',
        clause: { field: 'span_type', operator: FilterOp.CONTAINS, value: 'chain' },
        expected: "span.type ILIKE '%chain%'",
      },
      {
        name: 'span status → span.status equality (exact match)',
        clause: { field: 'span_status', operator: FilterOp.EQUALS, value: 'ERROR' },
        expected: "span.status = 'ERROR'",
      },
      {
        name: 'span status not-equals → span.status !=',
        clause: { field: 'span_status', operator: FilterOp.NOT_EQUALS, value: 'OK' },
        expected: "span.status != 'OK'",
      },
    ])('$name', ({ clause, expected }) => {
      expect(compileFilterModel([clause])).toEqual([expected]);
    });
  });

  test('multiple complete clauses compile in order (buildFilter ANDs them)', () => {
    const model: FilterClause[] = [
      { field: 'state', operator: FilterOp.EQUALS, value: 'OK' },
      { field: 'duration', operator: FilterOp.LESS_THAN_OR_EQUALS, value: '500' },
    ];
    expect(compileFilterModel(model)).toEqual(["attributes.status = 'OK'", 'attributes.execution_time_ms <= 500']);
  });

  describe('arbitrary tag / metadata key clauses (read clause.key)', () => {
    test('a tag clause compiles to a tags.<key> clause with the chosen operator', () => {
      expect(compileFilterModel([{ field: 'tag', operator: FilterOp.EQUALS, value: 'prod', key: 'env' }])).toEqual([
        "tags.env = 'prod'",
      ]);
    });

    test('a tag key with a dot or a space is backtick-escaped', () => {
      expect(compileFilterModel([{ field: 'tag', operator: FilterOp.NOT_EQUALS, value: 'v', key: 'my.tag' }])).toEqual([
        "tags.`my.tag` != 'v'",
      ]);
    });

    test('a metadata clause compiles to a request_metadata.<key> clause', () => {
      expect(
        compileFilterModel([{ field: 'metadata', operator: FilterOp.EQUALS, value: 'v1', key: 'custom.key' }]),
      ).toEqual(["request_metadata.custom.key = 'v1'"]);
    });

    test('a metadata CONTAINS clause compiles to ILIKE (backend has no CONTAINS token)', () => {
      expect(
        compileFilterModel([{ field: 'metadata', operator: FilterOp.CONTAINS, value: 'abc', key: 'custom.key' }]),
      ).toEqual(["request_metadata.custom.key ILIKE '%abc%'"]);
    });

    test('a tag clause with a blank key is dropped (incomplete)', () => {
      expect(compileFilterModel([{ field: 'tag', operator: FilterOp.EQUALS, value: 'prod', key: '' }])).toEqual([]);
    });
  });

  describe('assessment clauses (single field, key = assessment name, feedback.`<name>` with backtick-escaping)', () => {
    test('compiles an assessment equality clause via the clause key', () => {
      expect(
        compileFilterModel([{ field: 'assessment', operator: FilterOp.EQUALS, value: 'yes', key: 'relevance' }]),
      ).toEqual(["feedback.`relevance` = 'yes'"]);
    });

    test('compiles an assessment not-equals clause via the clause key', () => {
      expect(
        compileFilterModel([{ field: 'assessment', operator: FilterOp.NOT_EQUALS, value: 'bad', key: 'safety' }]),
      ).toEqual(["feedback.`safety` != 'bad'"]);
    });

    test('backtick-escapes assessment names with dots or spaces', () => {
      expect(
        compileFilterModel([{ field: 'assessment', operator: FilterOp.EQUALS, value: '5', key: 'my.judge score' }]),
      ).toEqual(["feedback.`my.judge score` = '5'"]);
    });

    test('an assessment clause with a blank key is dropped (incomplete)', () => {
      expect(compileFilterModel([{ field: 'assessment', operator: FilterOp.EQUALS, value: 'yes', key: '' }])).toEqual(
        [],
      );
    });
  });
});

describe('useMlflowTraceFilterFields', () => {
  const wrapper = ({ children }: { children: React.ReactNode }) => <IntlProvider locale="en">{children}</IntlProvider>;

  test('offers one combobox "Assessment" field whose keyOptions are the assessment names', () => {
    const { result } = renderHook(() => useMlflowTraceFilterFields(['relevance', 'safety']), { wrapper });

    const assessmentFields = result.current.filter((field) => field.id === 'assessment');
    expect(assessmentFields).toHaveLength(1);
    expect(assessmentFields[0]).toEqual({
      id: 'assessment',
      label: 'Assessment',
      operators: [FilterOp.EQUALS, FilterOp.NOT_EQUALS],
      valueInput: 'text',
      requiresKey: true,
      keyInput: 'combobox',
      keyOptions: [
        { value: 'relevance', label: 'relevance' },
        { value: 'safety', label: 'safety' },
      ],
      keyPlaceholder: 'Assessment name',
    });
  });

  test('still offers the "Assessment" field with empty keyOptions when no names are given', () => {
    const { result } = renderHook(() => useMlflowTraceFilterFields(), { wrapper });
    const assessment = result.current.find((field) => field.id === 'assessment');
    expect(assessment?.keyInput).toBe('combobox');
    expect(assessment?.keyOptions).toEqual([]);
    // No legacy per-name assessment:<name> fields remain.
    expect(result.current.some((field) => field.id.startsWith('assessment:'))).toBe(false);
  });

  test('offers span fields and requiresKey tag/metadata fields', () => {
    const { result } = renderHook(() => useMlflowTraceFilterFields(), { wrapper });

    expect(result.current.map((field) => field.id)).toEqual(
      expect.arrayContaining(['span_name', 'span_type', 'span_status', 'tag', 'metadata']),
    );
    // Tag and Metadata carry the free-text key sub-input; the other fields do not.
    expect(result.current.find((field) => field.id === 'tag')?.requiresKey).toBe(true);
    expect(result.current.find((field) => field.id === 'metadata')?.requiresKey).toBe(true);
    expect(result.current.find((field) => field.id === 'span_name')?.requiresKey).toBeUndefined();
  });
});

describe('compileTagFilters', () => {
  test('compiles a simple tag key to a bare tags.<key> equality clause', () => {
    expect(compileTagFilters([{ key: 'env', value: 'prod' }])).toEqual(["tags.env = 'prod'"]);
  });

  test('backticks a key containing a dot or a space', () => {
    expect(compileTagFilters([{ key: 'my.tag', value: 'v' }])).toEqual(["tags.`my.tag` = 'v'"]);
    expect(compileTagFilters([{ key: 'has space', value: 'v' }])).toEqual(["tags.`has space` = 'v'"]);
  });

  test('compiles multiple tag filters in order', () => {
    expect(
      compileTagFilters([
        { key: 'env', value: 'prod' },
        { key: 'team', value: 'ml' },
      ]),
    ).toEqual(["tags.env = 'prod'", "tags.team = 'ml'"]);
  });

  test('an empty list compiles to no clauses', () => {
    expect(compileTagFilters([])).toEqual([]);
  });

  test('escapes single quotes in tag values so the literal stays well-formed', () => {
    expect(compileTagFilters([{ key: 'author', value: "O'Reilly" }])).toEqual(["tags.author = 'O''Reilly'"]);
  });
});

describe('single-quote escaping in compiled clauses', () => {
  test.each<{ name: string; clause: FilterClause; expected: string }>([
    {
      name: 'equality value',
      clause: { field: 'state', operator: FilterOp.EQUALS, value: "O'Reilly" },
      expected: "attributes.status = 'O''Reilly'",
    },
    {
      name: 'contains → ILIKE value',
      clause: { field: 'source', operator: FilterOp.CONTAINS, value: "O'Reilly" },
      expected: "request_metadata.\"mlflow.source.name\" ILIKE '%O''Reilly%'",
    },
    {
      name: 'span text exact-match value',
      clause: { field: 'span_name', operator: FilterOp.EQUALS, value: "O'Reilly" },
      expected: "span.name ILIKE 'O''Reilly'",
    },
    {
      name: 'arbitrary tag value',
      clause: { field: 'tag', operator: FilterOp.EQUALS, key: 'author', value: "O'Reilly" },
      expected: "tags.author = 'O''Reilly'",
    },
  ])('escapes the value for $name', ({ clause, expected }) => {
    expect(compileFilterModel([clause])).toEqual([expected]);
  });
});
