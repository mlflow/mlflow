import { beforeAll, describe, expect, it, jest } from '@jest/globals';

import type { IntlShape } from '@databricks/i18n';
import { I18nUtils } from '@databricks/i18n';

import { ModelSpanType, type ModelTraceSpanNode } from '../ModelTrace.types';
import type { TimelineTreeMetric } from '../ModelTraceExplorerPreferencesContext';
import { getTimelineTreeComparisonMetricValues, getTimelineTreeMetricValue } from './TimelineTreeMetrics';
import * as tokenUsageUtils from '../ModelTraceTokenUsage.utils';
import * as timelineUtils from './TimelineTree.utils';
import { shouldEnableNewEvalRunsUi } from '../../../../../common/utils/FeatureUtils';

jest.mock('../../../../../common/utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('../../../../../common/utils/FeatureUtils')>(
    '../../../../../common/utils/FeatureUtils',
  ),
  shouldEnableNewEvalRunsUi: jest.fn(() => false),
}));

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

  describe('cost precision', () => {
    const lowCostSpan: ModelTraceSpanNode = {
      ...span,
      cost: { input_cost: 0.00005, output_cost: 0.000073, total_cost: 0.000123 },
    };

    it('flag on: limits cost metrics to four decimal places and marks them as estimated', () => {
      jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
      expect(getTimelineTreeMetricValue('cost', lowCostSpan, intl)?.value).toBe('≈$0.0001');
    });

    it('flag on: includes the estimation explanation in the cost metric title', () => {
      jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
      expect(getTimelineTreeMetricValue('cost', lowCostSpan, intl)?.title).toContain(
        'Cost is estimated from token usage and public model pricing.',
      );
    });

    it('flag off: keeps up to six decimal places', () => {
      jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(false);
      expect(getTimelineTreeMetricValue('cost', lowCostSpan, intl)?.value).toBe('$0.000123');
    });
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

  it('does not repeatedly inspect attributes for a span without token usage', () => {
    const node = { ...span, tokenUsage: undefined, attributes: {} };
    const read = jest.spyOn(tokenUsageUtils, 'getSpanTokenUsage');
    try {
      for (let render = 0; render < 8; render++) {
        expect(getTimelineTreeMetricValue('tokens', node, intl)).toBeUndefined();
      }
      expect(read).toHaveBeenCalledTimes(1);
    } finally {
      read.mockRestore();
    }
  });

  it.each([0, 42])('reuses fallback token usage including %s across metric reads', (total) => {
    const node = {
      ...span,
      tokenUsage: undefined,
      attributes: { 'gen_ai.usage.input_tokens': total, 'gen_ai.usage.output_tokens': 0 },
    };
    const read = jest.spyOn(tokenUsageUtils, 'getSpanTokenUsage');
    try {
      expect(getTimelineTreeMetricValue('tokens', node, intl)?.value).toBe(String(total));
      expect(getTimelineTreeMetricValue('tokens', node, intl)?.value).toBe(String(total));
      expect(read).toHaveBeenCalledTimes(1);
    } finally {
      read.mockRestore();
    }
  });

  it('shares a span token result between its row and comparison tooltip', () => {
    const node = { ...otelGenAiSpan };
    const read = jest.spyOn(tokenUsageUtils, 'getSpanTokenUsage');
    try {
      expect(getTimelineTreeMetricValue('tokens', node, intl)?.value).toBe('173');
      expect(getTimelineTreeComparisonMetricValues(node, undefined, intl)).toContainEqual(
        expect.objectContaining({ key: 'tokens', baseline: '173', candidate: '—' }),
      );
      expect(read).toHaveBeenCalledTimes(1);
    } finally {
      read.mockRestore();
    }
  });

  it('reads replacement data for nodes with the same span id', () => {
    const original = { ...span, tokenUsage: undefined, attributes: {} };
    const replacement = { ...original, tokenUsage: { total_tokens: 42 } };
    expect(original.key).toBe(replacement.key);
    expect(getTimelineTreeMetricValue('tokens', original, intl)).toBeUndefined();
    expect(getTimelineTreeMetricValue('tokens', replacement, intl)?.value).toBe('42');
  });

  it('formats duration only when the duration metric is requested', () => {
    const format = jest.spyOn(timelineUtils, 'spanTimeFormatter');
    try {
      getTimelineTreeMetricValue('tokens', span, intl);
      getTimelineTreeMetricValue('cost', span, intl);
      expect(format).not.toHaveBeenCalled();
      expect(getTimelineTreeMetricValue('duration', span, intl)?.value).toBe('3.20s');
      expect(format).toHaveBeenCalledTimes(1);
    } finally {
      format.mockRestore();
    }
  });
});
