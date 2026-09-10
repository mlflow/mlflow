import { beforeAll, describe, expect, it } from '@jest/globals';

import type { IntlShape } from '@databricks/i18n';
import { I18nUtils } from '@databricks/i18n';

import { ModelSpanType, type ModelTraceSpanNode } from '../ModelTrace.types';
import type { TimelineTreeMetric } from '../ModelTraceExplorerPreferencesContext';
import { getTimelineTreeMetricValue } from './TimelineTreeMetrics';

const span: ModelTraceSpanNode = {
  key: 'span-1',
  title: 'Generate answer',
  type: ModelSpanType.LLM,
  start: 0,
  end: 3_200_000,
  inputs: {},
  outputs: { usage: { total_tokens: 1980 } },
  attributes: {},
  assessments: [],
  traceId: 'trace-1',
  tokenUsage: { total_tokens: 1980 },
  cost: { input_cost: 0.01, output_cost: 0.02, total_cost: 0.03 },
};

const otelGenAiSpan: ModelTraceSpanNode = {
  ...span,
  attributes: [
    { key: 'gen_ai.usage.input_tokens', value: { int_value: 156 } },
    { key: 'gen_ai.usage.output_tokens', value: { int_value: 17 } },
  ],
  tokenUsage: undefined,
};

describe('getTimelineTreeMetricValue', () => {
  let intl: IntlShape;

  beforeAll(() => {
    intl = I18nUtils.createIntlWithLocale();
  });

  it.each<{ metric: TimelineTreeMetric; expected: string }>([
    { metric: 'duration', expected: '3.20s' },
    { metric: 'tokens', expected: '2K' },
    { metric: 'cost', expected: '$0.03' },
  ])('formats the available $metric metric', ({ metric, expected }) => {
    expect(getTimelineTreeMetricValue(metric, span, intl)?.value).toBe(expected);
  });

  it('formats token usage from OTel GenAI span attributes', () => {
    expect(getTimelineTreeMetricValue('tokens', otelGenAiSpan, intl)?.value).toBe('173');
  });

  it('ignores provider-specific token usage payloads', () => {
    const spanWithProviderTokenUsage: ModelTraceSpanNode = {
      ...span,
      outputs: { usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 } },
      tokenUsage: undefined,
    };

    expect(getTimelineTreeMetricValue('tokens', spanWithProviderTokenUsage, intl)).toBeUndefined();
  });

  it('uses token usage normalized onto the span node', () => {
    const normalizedSpan: ModelTraceSpanNode = {
      ...span,
      outputs: {},
      tokenUsage: { total_tokens: 42 },
    };

    expect(getTimelineTreeMetricValue('tokens', normalizedSpan, intl)?.value).toBe('42');
  });

  it.each<TimelineTreeMetric>(['tokens', 'cost'])('omits the unavailable %s metric', (metric) => {
    const spanWithoutOptionalMetrics: ModelTraceSpanNode = {
      key: 'span-2',
      title: 'Tool call',
      type: ModelSpanType.TOOL,
      start: 0,
      end: 1000,
      attributes: {},
      assessments: [],
      traceId: 'trace-1',
    };

    expect(getTimelineTreeMetricValue(metric, spanWithoutOptionalMetrics, intl)).toBeUndefined();
  });
});
