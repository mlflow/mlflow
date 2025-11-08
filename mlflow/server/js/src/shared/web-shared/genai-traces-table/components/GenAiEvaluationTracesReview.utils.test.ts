import {
  autoSelectFirstNonEmptyEvaluationId,
  getEvaluationResultInputTitle,
  getEvaluationResultTitle,
  stringifyValue,
} from './GenAiEvaluationTracesReview.utils';
import type { RunEvaluationTracesDataEntry } from '../types';

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
});
