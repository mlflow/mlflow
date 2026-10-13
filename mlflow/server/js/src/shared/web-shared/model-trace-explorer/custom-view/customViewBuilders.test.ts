import { describe, expect, it } from '@jest/globals';

import type { Assessment, ModelTrace, ModelTraceSpanNode } from '../ModelTrace.types';
import {
  buildAssessmentCardComponents,
  collectTraceAssessments,
  getAgentAssessments,
  getAssessmentBoardItems,
  getMetricsFromTraceInfo,
  getSpanAttributes,
  mapToAgentAssessments,
} from './customViewBuilders';

const makeNode = (node: Partial<ModelTraceSpanNode> & { key: string }): ModelTraceSpanNode =>
  ({
    title: node.key,
    start: 0,
    end: 0,
    type: 'UNKNOWN',
    events: [],
    attributes: {},
    assessments: [],
    ...node,
  }) as unknown as ModelTraceSpanNode;

const asInfo = (info: Record<string, unknown>): ModelTrace['info'] => info as unknown as ModelTrace['info'];

const feedback = (over: Record<string, unknown> = {}): Assessment =>
  ({
    assessment_id: 'a1',
    assessment_name: 'correctness',
    trace_id: 'trace-1',
    source: { source_type: 'LLM_JUDGE', source_id: 'judge' },
    create_time: '',
    last_update_time: '',
    feedback: { value: true },
    ...over,
  }) as unknown as Assessment;

describe('getMetricsFromTraceInfo', () => {
  it('derives V3 metrics (status, latency, token count) and uses the passed-in assessment count', () => {
    const info = asInfo({
      trace_location: {},
      state: 'OK',
      execution_duration: '1.23s',
      trace_metadata: { 'mlflow.trace.tokenUsage': JSON.stringify({ total_tokens: 999 }) },
      assessments: [feedback(), feedback({ assessment_id: 'a2' })],
    });
    // The count is passed in (post-merge with cached actions), not read off info.
    expect(getMetricsFromTraceInfo(info, 3)).toEqual({
      status: 'OK',
      latency: '1.23s',
      totalTokens: '999',
      assessments: '3',
    });
  });

  it('falls back to N/A token count when trace metadata is absent', () => {
    const info = asInfo({ trace_location: {}, state: 'ERROR' });
    expect(getMetricsFromTraceInfo(info, 0)).toEqual({
      status: 'ERROR',
      latency: 'N/A',
      totalTokens: 'N/A',
      assessments: '0',
    });
  });

  it('formats legacy trace-info latency (seconds >= 1s, else ms) and reports the passed-in count', () => {
    expect(getMetricsFromTraceInfo(asInfo({ status: 'OK', execution_time_ms: 1500 }), 2)).toEqual({
      status: 'OK',
      latency: '1.50s',
      totalTokens: 'N/A',
      assessments: '2',
    });
    expect(getMetricsFromTraceInfo(asInfo({ execution_time_ms: 250 }), 0).latency).toBe('250ms');
    expect(getMetricsFromTraceInfo(asInfo({}), 0).latency).toBe('N/A');
  });
});

describe('getAgentAssessments', () => {
  it('collects trace- and span-level assessments, deduped by id and skipping invalid ones', () => {
    const info = asInfo({
      trace_location: {},
      assessments: [feedback({ assessment_id: 'a1', feedback: { value: true }, rationale: 'looks good' })],
    });
    const nodeMap = {
      s1: makeNode({
        key: 's1',
        assessments: [
          // duplicate id -> ignored
          feedback({ assessment_id: 'a1', feedback: { value: false } }),
          // invalid -> ignored
          feedback({ assessment_id: 'a2', valid: false }),
          feedback({
            assessment_id: 'a3',
            assessment_name: 'relevance',
            feedback: { value: 'no', error: { error_code: 'E', error_message: 'boom' } },
            span_id: 's1',
          }),
        ],
      }),
    } satisfies Record<string, ModelTraceSpanNode>;
    expect(getAgentAssessments(info, nodeMap)).toEqual([
      {
        name: 'correctness',
        value: true,
        rationale: 'looks good',
        source: 'LLM_JUDGE',
        spanId: undefined,
        error: undefined,
      },
      { name: 'relevance', value: 'no', rationale: undefined, source: 'LLM_JUDGE', spanId: 's1', error: 'boom' },
    ]);
  });

  it('reads the value from expectation assessments (plain and serialized)', () => {
    const nodeMap = {
      s1: makeNode({
        key: 's1',
        assessments: [
          feedback({ assessment_id: 'e1', expectation: { value: 'ground-truth' }, feedback: undefined }),
          feedback({
            assessment_id: 'e2',
            expectation: { serialized_value: { value: '42', serialization_format: 'json' } },
            feedback: undefined,
          }),
        ],
      }),
    } satisfies Record<string, ModelTraceSpanNode>;
    const result = getAgentAssessments(asInfo({}), nodeMap);
    expect(result.map((a) => a.value)).toEqual(['ground-truth', '42']);
  });

  // `assessment_name` holds the issue id, so the readable `issue_name` must survive.
  it('names issue-reference assessments after their issue name, falling back to the issue id', () => {
    const info = asInfo({
      trace_location: {},
      assessments: [
        feedback({
          assessment_id: 'i1',
          assessment_name: 'issue-abc-123',
          issue: { issue_name: 'Hallucinated citation' },
          feedback: undefined,
          rationale: 'cited a nonexistent doc',
        }),
        feedback({
          assessment_id: 'i2',
          assessment_name: 'issue-def-456',
          issue: { issue_name: '' },
          feedback: undefined,
        }),
      ],
    });
    expect(getAgentAssessments(info, {})).toEqual([
      {
        name: 'Hallucinated citation',
        value: undefined,
        rationale: 'cited a nonexistent doc',
        source: 'LLM_JUDGE',
        spanId: undefined,
        error: undefined,
      },
      {
        name: 'issue-def-456',
        value: undefined,
        rationale: undefined,
        source: 'LLM_JUDGE',
        spanId: undefined,
        error: undefined,
      },
    ]);
  });
});

describe('collectTraceAssessments + mapToAgentAssessments', () => {
  it('collectTraceAssessments returns RAW trace- and span-level assessments, deduped by id', () => {
    const info = asInfo({
      trace_location: {},
      assessments: [feedback({ assessment_id: 'a1' })],
    });
    const nodeMap = {
      s1: makeNode({
        key: 's1',
        assessments: [
          // duplicate id -> ignored
          feedback({ assessment_id: 'a1' }),
          // invalid -> ignored
          feedback({ assessment_id: 'a2', valid: false }),
          feedback({ assessment_id: 'a3', span_id: 's1' }),
        ],
      }),
    } satisfies Record<string, ModelTraceSpanNode>;
    const raw = collectTraceAssessments(info, nodeMap);
    expect(raw.map((a) => a.assessment_id)).toEqual(['a1', 'a3']);
  });

  it('mapToAgentAssessments maps raw assessments, deduping ids and skipping invalid ones', () => {
    const mapped = mapToAgentAssessments([
      feedback({ assessment_id: 'a1', assessment_name: 'correctness', feedback: { value: true } }),
      // an in-session addition that overlaps a base id is deduped
      feedback({ assessment_id: 'a1', assessment_name: 'correctness', feedback: { value: false } }),
      feedback({ assessment_id: 'a2', valid: false }),
    ]);
    expect(mapped).toEqual([
      {
        name: 'correctness',
        value: true,
        rationale: undefined,
        source: 'LLM_JUDGE',
        spanId: undefined,
        error: undefined,
      },
    ]);
  });
});

describe('getAssessmentBoardItems', () => {
  it('derives sentiment and badge value from each assessment', () => {
    const items = getAssessmentBoardItems([
      { name: 'bool-true', value: true, source: 's' },
      { name: 'bool-false', value: false, source: 's' },
      { name: 'str-pass', value: 'pass', source: 's' },
      { name: 'str-fail', value: 'fail', source: 's' },
      { name: 'neutral', value: 'partial', source: 's' },
      { name: 'errored', value: 'ignored', source: 's', error: 'kaboom' },
      { name: 'empty', value: undefined, source: 's' },
    ]);
    expect(items).toEqual([
      { name: 'bool-true', value: 'true', rationale: undefined, source: 's', sentiment: 'positive' },
      { name: 'bool-false', value: 'false', rationale: undefined, source: 's', sentiment: 'negative' },
      { name: 'str-pass', value: 'pass', rationale: undefined, source: 's', sentiment: 'positive' },
      { name: 'str-fail', value: 'fail', rationale: undefined, source: 's', sentiment: 'negative' },
      { name: 'neutral', value: 'partial', rationale: undefined, source: 's', sentiment: 'neutral' },
      { name: 'errored', value: 'Error', rationale: 'kaboom', source: 's', sentiment: 'error' },
      { name: 'empty', value: undefined, rationale: undefined, source: 's', sentiment: 'neutral' },
    ]);
  });
});

describe('getSpanAttributes', () => {
  it('drops mlflow.* attributes and returns {} for a missing span', () => {
    const span = makeNode({ key: 's', attributes: { 'mlflow.spanType': 'TOOL', model: 'gpt' } });
    expect(getSpanAttributes(span)).toEqual({ model: 'gpt' });
    expect(getSpanAttributes(undefined)).toEqual({});
  });
});

describe('buildAssessmentCardComponents', () => {
  it('maps board items to AssessmentCard components with only present optional fields', () => {
    const { childIds, components } = buildAssessmentCardComponents(
      [
        { name: 'full', value: 'yes', rationale: 'because', source: 'judge', sentiment: 'positive' },
        { name: 'sparse', sentiment: 'neutral' },
      ],
      { idPrefix: 'board' },
    );
    expect(childIds).toEqual(['board-card-0', 'board-card-1']);
    expect(components[0]).toEqual({
      id: 'board-card-0',
      component: 'AssessmentCard',
      name: 'full',
      value: 'yes',
      rationale: 'because',
      source: 'judge',
      sentiment: 'positive',
    });
    expect(components[1]).toEqual({
      id: 'board-card-1',
      component: 'AssessmentCard',
      name: 'sparse',
      sentiment: 'neutral',
    });
  });
});
