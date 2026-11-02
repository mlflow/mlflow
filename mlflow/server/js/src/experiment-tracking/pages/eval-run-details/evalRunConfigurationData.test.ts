import { describe, expect, test } from '@jest/globals';
import type { ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';
import { aggregateEvalRunSidebarTraceStats, aggregateEvalRunConfigurationTraces } from './evalRunConfigurationData';

const createTrace = (
  runUuid: string,
  scores: Record<string, boolean | number | string>,
  index = 0,
): ModelTraceInfoV3 => ({
  trace_id: `trace-${runUuid}-${index}`,
  trace_location: { type: 'MLFLOW_EXPERIMENT', mlflow_experiment: { experiment_id: 'experiment-id' } },
  request_time: '2026-01-01T00:00:00Z',
  state: 'OK',
  tags: {},
  trace_metadata: { 'mlflow.sourceRun': runUuid },
  assessments: Object.entries(scores).map(([name, value]) => ({
    assessment_id: `${runUuid}-${index}-${name}`,
    assessment_name: name,
    trace_id: `trace-${runUuid}-${index}`,
    create_time: '2026-01-01T00:00:00Z',
    last_update_time: '2026-01-01T00:00:00Z',
    feedback: { value },
    source: { source_type: 'CODE', source_id: 'scorer' },
    rationale: 'Sampled feedback',
  })),
});

describe('evaluation configuration data', () => {
  test('sums trace usage and counts execution errors, not failed judgments or assessment errors', () => {
    const first = createTrace('a', { quality: false });
    first.trace_metadata = {
      'mlflow.trace.tokenUsage': '{"total_tokens":10}',
      'mlflow.trace.cost': '{"total_cost":0.2}',
    };
    if (first.assessments?.[0]) first.assessments[0].error = { error_code: 'FAILED' };
    const second = createTrace('a', {}, 1);
    second.state = 'ERROR';
    second.trace_metadata = {
      'mlflow.trace.tokenUsage': '{"total_tokens":20}',
      'mlflow.trace.cost': '{"total_cost":0.3}',
    };
    expect(aggregateEvalRunConfigurationTraces([first, second], false)).toEqual({
      traceCount: 2,
      totalTokens: 30,
      totalCost: 0.5,
      errors: 1,
      tracesWithTokens: 2,
      tracesWithCost: 2,
      isTruncated: false,
    });
    // A trace linked to two runs contributes to each run, without assessment-level multiplication.
    expect(aggregateEvalRunConfigurationTraces([first], false).totalTokens).toBe(10);
  });

  test('provides sidebar stats with trace truncation and incomplete usage coverage', () => {
    const first = createTrace('a', {});
    first.trace_metadata = {
      'mlflow.trace.tokenUsage': '{"total_tokens":10}',
      'mlflow.trace.cost': '{"total_cost":0.2}',
    };
    const second = createTrace('a', {}, 1);
    second.state = 'ERROR';

    expect(aggregateEvalRunSidebarTraceStats([first, second], true)).toEqual({
      traceCount: 2,
      totalTokens: 10,
      totalCost: 0.2,
      errors: 1,
      tracesWithTokens: 1,
      tracesWithCost: 1,
      isTruncated: true,
      hasPartialUsage: true,
    });
  });

  test('reports zero traces as loaded without inventing token or cost usage', () => {
    expect(aggregateEvalRunSidebarTraceStats([], false)).toEqual({
      traceCount: 0,
      totalTokens: undefined,
      totalCost: undefined,
      errors: 0,
      tracesWithTokens: 0,
      tracesWithCost: 0,
      isTruncated: false,
      hasPartialUsage: false,
    });
  });

  test.each([
    '{}',
    'null',
    'invalid',
    '{"total_tokens":-1,"total_cost":-1}',
    '{"total_tokens":"3","total_cost":true}',
    '{"total_tokens":1e999,"total_cost":1e999}',
  ])('excludes missing or malformed usage: %s', (metadata) => {
    const trace = createTrace('a', {});
    trace.trace_metadata = { 'mlflow.trace.tokenUsage': metadata, 'mlflow.trace.cost': metadata };
    expect(aggregateEvalRunConfigurationTraces([trace], false)).toMatchObject({
      totalTokens: undefined,
      totalCost: undefined,
      tracesWithTokens: 0,
      tracesWithCost: 0,
    });
  });

  test('distinguishes recorded zero, empty results, and partial coverage', () => {
    const trace = createTrace('a', {});
    trace.trace_metadata = { 'mlflow.trace.tokenUsage': '{"total_tokens":0}', 'mlflow.trace.cost': '{"total_cost":0}' };
    expect(aggregateEvalRunConfigurationTraces([trace], false)).toMatchObject({
      totalTokens: 0,
      totalCost: 0,
      tracesWithTokens: 1,
      tracesWithCost: 1,
    });
    expect(aggregateEvalRunConfigurationTraces([], false)).toMatchObject({
      traceCount: 0,
      errors: 0,
      totalTokens: undefined,
      totalCost: undefined,
    });
    const partial = aggregateEvalRunConfigurationTraces([trace, createTrace('a', {}, 1)], false);
    expect(partial).toMatchObject({ totalTokens: 0, tracesWithTokens: 1, traceCount: 2 });
  });
});
