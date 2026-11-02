import { describe, expect, test } from '@jest/globals';
import { buildEvalRunTraceFilter } from './useEvalRunTracesV4Controller';

describe('eval-run V4 trace filters', () => {
  test('always scopes user search and filter clauses to the escaped run id without a time range', () => {
    expect(
      buildEvalRunTraceFilter({
        runUuid: "run'id",
        searchQuery: "user's request",
        extraClauses: ["status = 'OK'"],
      }),
    ).toBe(
      "attributes.run_id = 'run''id' AND tags.`mlflow.trace.sourceScorer` IS NULL AND trace.text ILIKE '%user''s request%' AND status = 'OK'",
    );
  });

  test('scopes exact-id validation to both the run and the trace id', () => {
    expect(buildEvalRunTraceFilter({ runUuid: "run'id", searchQuery: 'tr-0123456789abcdef0123456789ABCDEF' })).toBe(
      "attributes.run_id = 'run''id' AND tags.`mlflow.trace.sourceScorer` IS NULL AND attributes.request_id = 'tr-0123456789abcdef0123456789abcdef'",
    );
  });
});
