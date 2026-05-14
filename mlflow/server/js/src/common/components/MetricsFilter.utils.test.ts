import { describe, it, expect, jest } from '@jest/globals';
import {
  isCompleteFilter,
  translateToMetricsFilters,
  translateToTracesPageFilters,
  type MetricFilter,
} from './MetricsFilter.utils';

jest.mock('@databricks/web-shared/model-trace-explorer', () => ({
  MLFLOW_TRACE_USER_KEY: 'mlflow.trace.user',
  createTraceMetadataFilter: jest.fn((key: string, value: string) => `trace.metadata.\`${key}\` = "${value}"`),
}));

jest.mock('@databricks/web-shared/genai-traces-table', () => ({
  USER_COLUMN_ID: 'user',
  FilterOperator: { EQUALS: '=' },
}));

describe('isCompleteFilter', () => {
  it('returns true when both column and value are set', () => {
    const filter: MetricFilter = { column: 'user', value: 'alice' };
    expect(isCompleteFilter(filter)).toBe(true);
  });

  it('returns false when value is empty', () => {
    const filter: MetricFilter = { column: 'user', value: '' };
    expect(isCompleteFilter(filter)).toBe(false);
  });

  it('returns false when column is empty', () => {
    const filter = { column: '' as MetricFilter['column'], value: 'alice' };
    expect(isCompleteFilter(filter)).toBe(false);
  });
});

describe('translateToMetricsFilters', () => {
  it('returns undefined for an empty filters array', () => {
    expect(translateToMetricsFilters([])).toBeUndefined();
  });

  it('returns undefined when all filters are incomplete', () => {
    const filters: MetricFilter[] = [
      { column: 'user', value: '' },
      { column: '' as MetricFilter['column'], value: 'alice' },
    ];
    expect(translateToMetricsFilters(filters)).toBeUndefined();
  });

  it('translates a user filter to a DSL string', () => {
    const filters: MetricFilter[] = [{ column: 'user', value: 'alice' }];
    const result = translateToMetricsFilters(filters);
    expect(result).toEqual(['trace.metadata.`mlflow.trace.user` = "alice"']);
  });

  it('skips incomplete filters and returns the rest', () => {
    const filters: MetricFilter[] = [
      { column: 'user', value: 'alice' },
      { column: 'user', value: '' },
    ];
    const result = translateToMetricsFilters(filters);
    expect(result).toEqual(['trace.metadata.`mlflow.trace.user` = "alice"']);
  });

  it('translates multiple complete filters', () => {
    const filters: MetricFilter[] = [
      { column: 'user', value: 'alice' },
      { column: 'user', value: 'bob' },
    ];
    const result = translateToMetricsFilters(filters);
    expect(result).toEqual([
      'trace.metadata.`mlflow.trace.user` = "alice"',
      'trace.metadata.`mlflow.trace.user` = "bob"',
    ]);
  });
});

describe('translateToTracesPageFilters', () => {
  it('returns undefined for an empty filters array', () => {
    expect(translateToTracesPageFilters([])).toBeUndefined();
  });

  it('returns undefined when all filters are incomplete', () => {
    const filters: MetricFilter[] = [
      { column: 'user', value: '' },
      { column: '' as MetricFilter['column'], value: 'alice' },
    ];
    expect(translateToTracesPageFilters(filters)).toBeUndefined();
  });

  it('translates a user filter to a Traces page URL filter string', () => {
    const filters: MetricFilter[] = [{ column: 'user', value: 'alice' }];
    expect(translateToTracesPageFilters(filters)).toEqual(['user::=::alice']);
  });

  it('skips incomplete filters and returns the rest', () => {
    const filters: MetricFilter[] = [
      { column: 'user', value: 'alice' },
      { column: 'user', value: '' },
    ];
    expect(translateToTracesPageFilters(filters)).toEqual(['user::=::alice']);
  });

  it('translates multiple complete filters', () => {
    const filters: MetricFilter[] = [
      { column: 'user', value: 'alice' },
      { column: 'user', value: 'bob' },
    ];
    expect(translateToTracesPageFilters(filters)).toEqual(['user::=::alice', 'user::=::bob']);
  });
});
