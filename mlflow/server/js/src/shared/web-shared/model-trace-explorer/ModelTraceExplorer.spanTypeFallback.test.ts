import { beforeEach, describe, expect, it, jest } from '@jest/globals';

import { shouldEnableNewEvalRunsUi } from '../../../common/utils/FeatureUtils';
import type { ModelTraceSpanNode } from './ModelTrace.types';
import { ModelSpanType } from './ModelTrace.types';
import { normalizeNewSpanData, searchTree } from './ModelTraceExplorer.utils';
import { MOCK_V3_SPANS } from './ModelTraceExplorer.test-utils';
import { TEST_SPAN_FILTER_STATE } from './timeline-tree/TimelineTree.test-utils';

jest.mock('../../../common/utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('../../../common/utils/FeatureUtils')>('../../../common/utils/FeatureUtils'),
  shouldEnableNewEvalRunsUi: jest.fn(() => false),
}));

const spanWithAttributes = (attributes: Record<string, string>) => ({ ...MOCK_V3_SPANS[0], attributes });

// ModelSpanType.MEMORY is absent from TEST_SPAN_FILTER_STATE.spanTypeDisplayState.
const memorySpanNode: ModelTraceSpanNode = {
  key: 'memory-span',
  title: 'memory-span',
  children: [],
  inputs: undefined,
  outputs: undefined,
  attributes: undefined,
  start: 0,
  end: 1,
  type: ModelSpanType.MEMORY,
  assessments: [],
  traceId: 'test',
};

const memoryFilterState = { ...TEST_SPAN_FILTER_STATE, showParents: false, showExceptions: false };

describe('span type resolution with the new eval runs UI', () => {
  beforeEach(() => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
  });

  it('uses mlflow.spanType directly when present (raw value wins, no inference)', () => {
    const normalized = normalizeNewSpanData(
      spanWithAttributes({ 'mlflow.spanType': JSON.stringify(ModelSpanType.TOOL), 'gen_ai.system': 'anthropic' }),
      0,
      0,
      [],
      {},
      '',
    );
    expect(normalized.type).toBe(ModelSpanType.TOOL);
  });

  it('infers CHAT_MODEL from gen_ai.system when mlflow.spanType is absent', () => {
    const normalized = normalizeNewSpanData(spanWithAttributes({ 'gen_ai.system': 'anthropic' }), 0, 0, [], {}, '');
    expect(normalized.type).toBe(ModelSpanType.CHAT_MODEL);
  });

  it('infers AGENT from span.type=interaction when mlflow.spanType is absent', () => {
    const normalized = normalizeNewSpanData(spanWithAttributes({ 'span.type': 'interaction' }), 0, 0, [], {}, '');
    expect(normalized.type).toBe(ModelSpanType.AGENT);
  });

  it('falls back to UNKNOWN when mlflow.spanType is absent and no gen_ai/span.type attrs are present', () => {
    const normalized = normalizeNewSpanData(spanWithAttributes({}), 0, 0, [], {}, '');
    expect(normalized.type).toBe(ModelSpanType.UNKNOWN);
  });

  it('keeps a span visible when its type is absent from spanTypeDisplayState and the search filter matches', () => {
    const { filteredTreeNodes } = searchTree(memorySpanNode, 'memory', memoryFilterState);
    expect(filteredTreeNodes).toEqual([expect.objectContaining({ key: 'memory-span' })]);
  });
});

describe('span type resolution without the new eval runs UI', () => {
  beforeEach(() => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(false);
  });

  it('leaves the type unset when mlflow.spanType is absent, even with gen_ai attributes', () => {
    expect(normalizeNewSpanData(spanWithAttributes({ 'gen_ai.system': 'anthropic' }), 0, 0, [], {}, '').type).toBe(
      undefined,
    );
    expect(normalizeNewSpanData(spanWithAttributes({}), 0, 0, [], {}, '').type).toBe(undefined);
  });

  it('filters out a span whose type is absent from spanTypeDisplayState', () => {
    const { filteredTreeNodes } = searchTree(memorySpanNode, 'memory', memoryFilterState);
    expect(filteredTreeNodes).toEqual([]);
  });
});
