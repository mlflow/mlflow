import { describe, it, expect, jest } from '@jest/globals';
import {
  isCompleteFilter,
  translateToMetricsFilters,
  translateToTracesPageFilters,
  type MetricFilter,
} from './MetricsFilter.utils';

jest.mock('@databricks/web-shared/model-trace-explorer', () => ({
  MLFLOW_TRACE_USER_KEY: 'mlflow.trace.user',
  SESSION_ID_METADATA_KEY: 'mlflow.trace.session',
  MLFLOW_GIT_BRANCH_KEY: 'mlflow.source.git.branch',
  MLFLOW_GIT_COMMIT_KEY: 'mlflow.source.git.commit',
  TraceFilterKey: { STATUS: 'status', TAG: 'tag', METADATA: 'metadata' },
  TraceStatus: { IN_PROGRESS: 'IN_PROGRESS', OK: 'OK', ERROR: 'ERROR' },
  createTraceMetadataFilter: jest.fn((key: string, value: string) => `trace.metadata.\`${key}\` = "${value}"`),
  createTraceFilter: jest.fn((field: string, value: string) => `trace.${field} = "${value}"`),
}));

jest.mock('@databricks/web-shared/genai-traces-table', () => ({
  USER_COLUMN_ID: 'user',
  SESSION_COLUMN_ID: 'session',
  STATE_COLUMN_ID: 'state',
  GIT_BRANCH_COLUMN_ID: 'git_branch',
  GIT_COMMIT_COLUMN_ID: 'git_commit',
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

  it('translates a session filter to a metadata DSL string with the session metadata key', () => {
    const filters: MetricFilter[] = [{ column: 'session', value: 'sess-123' }];
    expect(translateToMetricsFilters(filters)).toEqual(['trace.metadata.`mlflow.trace.session` = "sess-123"']);
  });

  it('translates a state filter to a trace.status DSL string', () => {
    expect(translateToMetricsFilters([{ column: 'state', value: 'OK' }])).toEqual(['trace.status = "OK"']);
    expect(translateToMetricsFilters([{ column: 'state', value: 'ERROR' }])).toEqual(['trace.status = "ERROR"']);
    expect(translateToMetricsFilters([{ column: 'state', value: 'IN_PROGRESS' }])).toEqual([
      'trace.status = "IN_PROGRESS"',
    ]);
  });

  it('translates a git_branch filter to a metadata DSL string with the git branch metadata key', () => {
    const filters: MetricFilter[] = [{ column: 'git_branch', value: 'main' }];
    expect(translateToMetricsFilters(filters)).toEqual(['trace.metadata.`mlflow.source.git.branch` = "main"']);
  });

  it('translates a git_commit filter to a metadata DSL string with the git commit metadata key', () => {
    const filters: MetricFilter[] = [{ column: 'git_commit', value: 'abc1234' }];
    expect(translateToMetricsFilters(filters)).toEqual(['trace.metadata.`mlflow.source.git.commit` = "abc1234"']);
  });

  it('translates a heterogeneous mix of complete filters preserving order', () => {
    const filters: MetricFilter[] = [
      { column: 'user', value: 'alice' },
      { column: 'session', value: 'sess-123' },
      { column: 'state', value: 'ERROR' },
      { column: 'git_branch', value: 'main' },
      { column: 'git_commit', value: 'abc1234' },
    ];
    expect(translateToMetricsFilters(filters)).toEqual([
      'trace.metadata.`mlflow.trace.user` = "alice"',
      'trace.metadata.`mlflow.trace.session` = "sess-123"',
      'trace.status = "ERROR"',
      'trace.metadata.`mlflow.source.git.branch` = "main"',
      'trace.metadata.`mlflow.source.git.commit` = "abc1234"',
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

  it('translates a session filter using the session column id', () => {
    expect(translateToTracesPageFilters([{ column: 'session', value: 'sess-123' }])).toEqual(['session::=::sess-123']);
  });

  it('translates a state filter using the state column id', () => {
    expect(translateToTracesPageFilters([{ column: 'state', value: 'OK' }])).toEqual(['state::=::OK']);
    expect(translateToTracesPageFilters([{ column: 'state', value: 'ERROR' }])).toEqual(['state::=::ERROR']);
    expect(translateToTracesPageFilters([{ column: 'state', value: 'IN_PROGRESS' }])).toEqual([
      'state::=::IN_PROGRESS',
    ]);
  });

  it('translates a git_branch filter using the git_branch column id', () => {
    expect(translateToTracesPageFilters([{ column: 'git_branch', value: 'main' }])).toEqual(['git_branch::=::main']);
  });

  it('translates a git_commit filter using the git_commit column id', () => {
    expect(translateToTracesPageFilters([{ column: 'git_commit', value: 'abc1234' }])).toEqual([
      'git_commit::=::abc1234',
    ]);
  });

  it('translates a heterogeneous mix preserving order', () => {
    const filters: MetricFilter[] = [
      { column: 'user', value: 'alice' },
      { column: 'session', value: 'sess-123' },
      { column: 'state', value: 'ERROR' },
      { column: 'git_branch', value: 'main' },
      { column: 'git_commit', value: 'abc1234' },
    ];
    expect(translateToTracesPageFilters(filters)).toEqual([
      'user::=::alice',
      'session::=::sess-123',
      'state::=::ERROR',
      'git_branch::=::main',
      'git_commit::=::abc1234',
    ]);
  });
});
