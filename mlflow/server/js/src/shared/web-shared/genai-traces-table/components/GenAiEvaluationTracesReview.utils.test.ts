import { describe, it, jest, expect } from '@jest/globals';

import type { ThemeType } from '@databricks/design-system';
import { I18nUtils } from '@databricks/i18n';

import {
  autoSelectFirstNonEmptyEvaluationId,
  getAssessmentValueLabel,
  getEvaluationResultInputTitle,
  getEvaluationResultTitle,
  stringifyValue,
  tryExtractUserMessageContent,
} from './GenAiEvaluationTracesReview.utils';
import type { AssessmentInfo, RunEvaluationTracesDataEntry } from '../types';

const buildExampleRunEvaluationTracesDataEntry = ({
  evaluationId,
  inputs,
}: {
  evaluationId: string | undefined | null;
  inputs?: any;
}): RunEvaluationTracesDataEntry => {
  return {
    requestId: '',
    // @ts-expect-error test the case where evaluationId is undefined or null
    evaluationId: evaluationId,
    inputs: inputs || {},
    inputsId: '',
    outputs: {},
    targets: {},
    overallAssessments: [],
    responseAssessmentsByName: { overall_assessment: [] },
    metrics: {},
    retrievalChunks: [],
  };
};

describe('EvaluationsReview utils', () => {
  describe('autoSelectFirstNonEmptyEvaluationId', () => {
    const exampleEvaluations = [
      buildExampleRunEvaluationTracesDataEntry({ evaluationId: undefined }),
      buildExampleRunEvaluationTracesDataEntry({ evaluationId: null }),
      buildExampleRunEvaluationTracesDataEntry({ evaluationId: '' }),
      buildExampleRunEvaluationTracesDataEntry({ evaluationId: 'evaluation-1' }),
      buildExampleRunEvaluationTracesDataEntry({ evaluationId: 'evaluation-2' }),
    ];

    it('should select the first non-empty evaluation id', () => {
      const mockSetter = jest.fn();
      autoSelectFirstNonEmptyEvaluationId(exampleEvaluations, undefined, mockSetter);
      expect(mockSetter).toHaveBeenCalledWith('evaluation-1');
      expect(mockSetter).toHaveBeenCalledTimes(1);
    });

    it.each([
      [],
      null,
      // All evaluations have empty evaluation id
      [
        buildExampleRunEvaluationTracesDataEntry({ evaluationId: undefined }),
        buildExampleRunEvaluationTracesDataEntry({ evaluationId: null }),
        buildExampleRunEvaluationTracesDataEntry({ evaluationId: '' }),
      ],
    ])('should not set if no non-empty evaluation id', (evaluations: RunEvaluationTracesDataEntry[] | null) => {
      const mockSetter = jest.fn();
      autoSelectFirstNonEmptyEvaluationId(evaluations, undefined, mockSetter);
      expect(mockSetter).not.toHaveBeenCalled();
    });

    it('should not set if selectedEvaluationId is already set', () => {
      const mockSetter = jest.fn();
      autoSelectFirstNonEmptyEvaluationId(exampleEvaluations, 'some-eval-id', mockSetter);
      expect(mockSetter).not.toHaveBeenCalled();
    });
  });

  describe('getEvaluationResultTitle', () => {
    it('should get the inputs.request if present', () => {
      const actual = getEvaluationResultTitle(
        buildExampleRunEvaluationTracesDataEntry({ evaluationId: 'evaluation-1', inputs: { request: 'request' } }),
      );
      expect(actual).toEqual('request');
    });

    it('should get the evaluation id if inputs.request not present', () => {
      const actual = getEvaluationResultTitle(
        buildExampleRunEvaluationTracesDataEntry({ evaluationId: 'evaluation-1' }),
      );
      expect(actual).toEqual('evaluation-1');
    });

    it('should get the stringify inputs.request if not a string', () => {
      const actual = getEvaluationResultTitle(
        buildExampleRunEvaluationTracesDataEntry({
          evaluationId: 'evaluation-1',
          inputs: { arg1: { foo: 'bar' } },
        }),
      );
      expect(actual).toEqual(stringifyValue({ arg1: { foo: 'bar' } }));
    });

    it('should parse openai messages', () => {
      const actual = getEvaluationResultTitle(
        buildExampleRunEvaluationTracesDataEntry({
          evaluationId: 'evaluation-1',
          inputs: { messages: [{ role: 'user', content: 'foo' }] },
        }),
      );
      expect(actual).toEqual('foo');
    });

    it('should read the last message from openai messages', () => {
      const actual = getEvaluationResultTitle(
        buildExampleRunEvaluationTracesDataEntry({
          evaluationId: 'evaluation-1',
          inputs: {
            messages: [
              { role: 'user', content: 'foo' },
              { role: 'assistant', content: 'yes?' },
              { role: 'user', content: 'bar' },
            ],
          },
        }),
      );
      expect(actual).toEqual('bar');
    });
  });

  describe('getEvaluationResultInputTitle', () => {
    it('should return the input if it is a string', () => {
      const actual = getEvaluationResultInputTitle(
        buildExampleRunEvaluationTracesDataEntry({ evaluationId: 'evaluation-1', inputs: { request: 'request' } }),
        'request',
      );
      expect(actual).toEqual('request');
    });

    it('should stringify the input if it is not a string', () => {
      const actual = getEvaluationResultInputTitle(
        buildExampleRunEvaluationTracesDataEntry({
          evaluationId: 'evaluation-1',
          inputs: { customKey: { custom: 'custom' } },
        }),
        'customKey',
      );
      expect(actual).toEqual(stringifyValue({ custom: 'custom' }));
    });

    it('should return undefined if the key does not exist', () => {
      const actual = getEvaluationResultInputTitle(
        buildExampleRunEvaluationTracesDataEntry({ evaluationId: 'evaluation-1' }),
        'request',
      );
      expect(actual).toBeUndefined();
    });

    it('should parse openai messages', () => {
      const actual = getEvaluationResultInputTitle(
        buildExampleRunEvaluationTracesDataEntry({
          evaluationId: 'evaluation-1',
          inputs: { messages: [{ role: 'user', content: 'foo' }] },
        }),
        'messages',
      );
      expect(actual).toEqual('foo');
    });

    it('should read the last message from openai messages', () => {
      const actual = getEvaluationResultInputTitle(
        buildExampleRunEvaluationTracesDataEntry({
          evaluationId: 'evaluation-1',
          inputs: {
            messages: [
              { role: 'user', content: 'foo' },
              { role: 'assistant', content: 'yes?' },
              { role: 'user', content: 'bar' },
            ],
          },
        }),
        'messages',
      );
      expect(actual).toEqual('bar');
    });

    it('should parse openai messages without messages key', () => {
      const actual = getEvaluationResultInputTitle(
        buildExampleRunEvaluationTracesDataEntry({
          evaluationId: 'evaluation-1',
          inputs: {
            messages_custom: [
              { role: 'user', content: 'foo' },
              { role: 'assistant', content: 'yes?' },
              { role: 'user', content: 'bar' },
            ],
          },
        }),
        'messages_custom',
      );
      expect(actual).toEqual('bar');
    });
  });

  it('should read the last message from openai messages, when nested under request', () => {
    const actual = getEvaluationResultInputTitle(
      buildExampleRunEvaluationTracesDataEntry({
        evaluationId: 'evaluation-1',
        inputs: {
          request: {
            messages: [
              { role: 'user', content: 'foo' },
              { role: 'assistant', content: 'yes?' },
              { role: 'user', content: 'bar' },
            ],
          },
        },
      }),
      'request',
    );
    expect(actual).toEqual('bar');
  });

  describe('tryExtractUserMessageContent', () => {
    it('should return undefined for null/undefined input', () => {
      expect(tryExtractUserMessageContent(null)).toBeUndefined();
      expect(tryExtractUserMessageContent(undefined)).toBeUndefined();
    });

    it('should extract user message from Langchain nested array format', () => {
      const langchainInput = [
        [
          {
            content: 'How to use Databricks?',
            additional_kwargs: {},
            response_metadata: {},
            type: 'human',
            name: null,
            id: null,
          },
        ],
      ];
      expect(tryExtractUserMessageContent(langchainInput)).toEqual('How to use Databricks?');
    });

    it('should extract last user message from Langchain format with multiple messages', () => {
      const langchainInput = [
        [
          { content: 'First question', type: 'human' },
          { content: 'Response here', type: 'ai' },
          { content: 'Follow up question', type: 'human' },
        ],
      ];
      expect(tryExtractUserMessageContent(langchainInput)).toEqual('Follow up question');
    });

    it('should extract user message from OpenAI format', () => {
      const openaiInput = {
        messages: [
          { role: 'system', content: 'You are a helpful assistant' },
          { role: 'user', content: 'Hello there' },
        ],
      };
      expect(tryExtractUserMessageContent(openaiInput)).toEqual('Hello there');
    });

    it('should return undefined for non-chat-like objects', () => {
      expect(tryExtractUserMessageContent({ foo: 'bar' })).toBeUndefined();
      expect(tryExtractUserMessageContent({ data: [1, 2, 3] })).toBeUndefined();
    });

    it('should return undefined for empty arrays', () => {
      expect(tryExtractUserMessageContent([])).toBeUndefined();
      expect(tryExtractUserMessageContent([[]])).toBeUndefined();
    });

    it('should return undefined for primitive values', () => {
      expect(tryExtractUserMessageContent(123)).toBeUndefined();
      expect(tryExtractUserMessageContent(true)).toBeUndefined();
    });

    it('should return undefined for string inputs', () => {
      // This can happen when request_preview is truncated and JSON.parse fails
      expect(tryExtractUserMessageContent('just a plain string')).toBeUndefined();
    });
  });
});

describe('getAssessmentValueLabel', () => {
  const intl = I18nUtils.createIntlWithLocale();
  const mockTheme = {} as ThemeType;

  const createMockAssessmentInfo = (dtype: AssessmentInfo['dtype']): AssessmentInfo => ({
    name: 'test_assessment',
    displayName: 'Test Assessment',
    isKnown: false,
    isOverall: false,
    metricName: 'test_metric',
    source: undefined,
    isCustomMetric: false,
    isEditable: false,
    isRetrievalAssessment: false,
    isSessionLevelAssessment: false,
    dtype,
    uniqueValues: new Set(),
    docsLink: '',
    missingTooltip: '',
    description: '',
  });

  it('should return "Error" for error value regardless of dtype', () => {
    const booleanAssessment = createMockAssessmentInfo('boolean');
    const result = getAssessmentValueLabel(intl, mockTheme, booleanAssessment, 'Error');
    expect(result.content).toBe('Error');
  });

  it('should return "Error" for pass-fail dtype with error value', () => {
    const passFail = createMockAssessmentInfo('pass-fail');
    const result = getAssessmentValueLabel(intl, mockTheme, passFail, 'Error');
    expect(result.content).toBe('Error');
  });

  it('should return "True" for boolean dtype with true value', () => {
    const booleanAssessment = createMockAssessmentInfo('boolean');
    const result = getAssessmentValueLabel(intl, mockTheme, booleanAssessment, true);
    expect(result.content).toBe('True');
  });

  it('should return "False" for boolean dtype with false value', () => {
    const booleanAssessment = createMockAssessmentInfo('boolean');
    const result = getAssessmentValueLabel(intl, mockTheme, booleanAssessment, false);
    expect(result.content).toBe('False');
  });

  it('should return "null" for boolean dtype with undefined value', () => {
    const booleanAssessment = createMockAssessmentInfo('boolean');
    const result = getAssessmentValueLabel(intl, mockTheme, booleanAssessment, undefined);
    expect(result.content).toBe('null');
  });

  it('should return string representation for other dtypes', () => {
    const stringAssessment = createMockAssessmentInfo('string');
    const result = getAssessmentValueLabel(intl, mockTheme, stringAssessment, 'custom_value');
    expect(result.content).toBe('custom_value');
  });
});
