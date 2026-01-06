import { render, waitFor } from '@testing-library/react';
import { IntlProvider } from '@databricks/i18n';
import { FormProvider, useForm } from 'react-hook-form';
import SampleScorerOutputPanelContainer from './SampleScorerOutputPanelContainer';
import { useEvaluateTraces } from './useEvaluateTraces';
import { TraceJudgeEvaluationResult, type JudgeEvaluationResult } from './useEvaluateTraces.common';
import { type ModelTrace } from '@databricks/web-shared/model-trace-explorer';
import SampleScorerOutputPanelRenderer from './SampleScorerOutputPanelRenderer';
import type { ScorerFormData } from './utils/scorerTransformUtils';
import { LLM_TEMPLATE, type EvaluateTracesParams } from './types';
import { jest } from '@jest/globals';
import { describe } from '@jest/globals';
import { beforeEach } from '@jest/globals';
import { it } from '@jest/globals';
import { expect } from '@jest/globals';
import { ScorerEvaluationScope } from './constants';

jest.mock('./useEvaluateTraces');
jest.mock('./SampleScorerOutputPanelRenderer');

const mockedUseEvaluateTraces = jest.mocked(useEvaluateTraces);
const mockedRenderer = jest.mocked(SampleScorerOutputPanelRenderer);

const experimentId = 'exp-123';

function createMockTrace(traceId: string): ModelTrace {
  return {
    info: { trace_id: traceId } as any,
    data: { spans: [] },
  } as ModelTrace;
}

function createMockEvalResult(traceId: string): TraceJudgeEvaluationResult {
  return {
    trace: createMockTrace(traceId),
    results: [
      {
        assessment_id: `assessment-${traceId}`,
        result: 'PASS',
        rationale: 'Good quality',
        error: null,
      },
    ],
    error: null,
  };
}

// Test wrapper component that sets up a real form
interface TestWrapperProps {
  defaultValues?: Partial<ScorerFormData>;
  onScorerFinished?: () => void;
}

function TestWrapper({ defaultValues, onScorerFinished }: TestWrapperProps) {
  const form = useForm<ScorerFormData>({
    defaultValues: {
      name: 'Test Scorer',
      instructions: 'Test instructions',
      llmTemplate: LLM_TEMPLATE.CUSTOM,
      sampleRate: 100,
      scorerType: 'llm',
      ...defaultValues,
    },
  });

  return (
    <IntlProvider locale="en">
      <FormProvider {...form}>
        <SampleScorerOutputPanelContainer
          control={form.control}
          experimentId={experimentId}
          onScorerFinished={onScorerFinished}
        />
      </FormProvider>
    </IntlProvider>
  );
}

describe('SampleScorerOutputPanelContainer', () => {
  const mockEvaluateTraces: jest.MockedFunction<(params: EvaluateTracesParams) => Promise<JudgeEvaluationResult[]>> =
    jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();

    mockedUseEvaluateTraces.mockReturnValue([
      mockEvaluateTraces,
      {
        data: null,
        isLoading: false,
        error: null,
        reset: jest.fn(),
      },
    ]);

    mockedRenderer.mockReturnValue(null);
  });

  const renderComponent = (props: TestWrapperProps = {}) => {
    return render(<TestWrapper {...props} />);
  };

  describe('Golden Path - Successful Operations', () => {
    it('should render and call renderer with correct initial props', () => {
      renderComponent();

      expect(mockedRenderer).toHaveBeenCalledWith(
        expect.objectContaining({
          isLoading: false,
          isRunScorerDisabled: false,
          error: null,
          currentEvalResultIndex: 0,
          totalTraces: 0,
          itemsToEvaluate: { itemCount: 10, itemIds: [] },
        }),
        expect.anything(),
      );
    });

    it('should pass evaluation results to renderer when data is available', () => {
      const mockResults = [createMockEvalResult('trace-1'), createMockEvalResult('trace-2')];

      mockedUseEvaluateTraces.mockReturnValue([
        mockEvaluateTraces,
        {
          data: mockResults,
          isLoading: false,
          error: null,
          reset: jest.fn(),
        },
      ]);

      renderComponent();

      expect(mockedRenderer).toHaveBeenCalledWith(
        expect.objectContaining({
          totalTraces: 2,
          currentEvalResult: mockResults[0],
          assessments: expect.any(Array),
        }),
        expect.anything(),
      );
    });

    it('should pass loading state to renderer', () => {
      mockedUseEvaluateTraces.mockReturnValue([
        mockEvaluateTraces,
        {
          data: null,
          isLoading: true,
          error: null,
          reset: jest.fn(),
        },
      ]);

      renderComponent();

      expect(mockedRenderer).toHaveBeenCalledWith(
        expect.objectContaining({
          isLoading: true,
        }),
        expect.anything(),
      );
    });

    it('should pass error state to renderer when evaluation fails', () => {
      const mockError = new Error('API error');

      mockedUseEvaluateTraces.mockReturnValue([
        mockEvaluateTraces,
        {
          data: null,
          isLoading: false,
          error: mockError,
          reset: jest.fn(),
        },
      ]);

      renderComponent();

      expect(mockedRenderer).toHaveBeenCalledWith(
        expect.objectContaining({
          error: mockError,
        }),
        expect.anything(),
      );
    });

    it('should provide handleRunScorer callback that calls evaluateTraces', async () => {
      const mockResults = [createMockEvalResult('trace-1')];
      mockEvaluateTraces.mockResolvedValue(mockResults);

      renderComponent();

      const rendererProps = mockedRenderer.mock.calls[0][0];
      rendererProps.handleRunScorer();

      expect(mockEvaluateTraces).toHaveBeenCalledWith({
        evaluationScope: ScorerEvaluationScope.TRACES,
        itemCount: 10,
        itemIds: [],
        locations: [{ mlflow_experiment: { experiment_id: experimentId }, type: 'MLFLOW_EXPERIMENT' }],
        judgeInstructions: 'Test instructions',
        experimentId,
        serializedScorer: expect.any(String),
      });
    });

    it('should call onScorerFinished when evaluation succeeds with results', async () => {
      const onScorerFinished = jest.fn();
      const mockResults = [createMockEvalResult('trace-1')];
      mockEvaluateTraces.mockImplementation(() => {
        onScorerFinished();
        return Promise.resolve(mockResults);
      });

      renderComponent({ onScorerFinished });

      const rendererProps = mockedRenderer.mock.calls[0][0];
      await rendererProps.handleRunScorer();

      await waitFor(() => {
        expect(onScorerFinished).toHaveBeenCalledTimes(1);
      });
    });
  });

  describe('Edge Cases', () => {
    it('should disable run scorer button when instructions are empty', () => {
      renderComponent({ defaultValues: { instructions: '' } });

      expect(mockedRenderer).toHaveBeenCalledWith(
        expect.objectContaining({
          isRunScorerDisabled: true,
          runScorerDisabledTooltip: expect.any(String),
        }),
        expect.anything(),
      );
    });

    it('should disable run scorer button when instructions are undefined', () => {
      renderComponent({ defaultValues: { instructions: undefined } });

      expect(mockedRenderer).toHaveBeenCalledWith(
        expect.objectContaining({
          isRunScorerDisabled: true,
        }),
        expect.anything(),
      );
    });

    it('should not call onScorerFinished when evaluation returns empty results', async () => {
      const onScorerFinished = jest.fn();
      mockEvaluateTraces.mockResolvedValue([]);

      renderComponent({ onScorerFinished });

      const rendererProps = mockedRenderer.mock.calls[0][0];
      rendererProps.handleRunScorer();

      expect(onScorerFinished).not.toHaveBeenCalled();
    });

    it('should handle undefined currentTrace when no data is available', () => {
      mockedUseEvaluateTraces.mockReturnValue([
        mockEvaluateTraces,
        {
          data: null,
          isLoading: false,
          error: null,
          reset: jest.fn(),
        },
      ]);

      renderComponent();

      expect(mockedRenderer).toHaveBeenCalledWith(
        expect.objectContaining({
          currentEvalResult: undefined,
          assessments: undefined,
        }),
        expect.anything(),
      );
    });
  });

  describe('Error Conditions', () => {
    it('should not call evaluateTraces when instructions are missing', async () => {
      renderComponent({ defaultValues: { instructions: undefined } });

      const rendererProps = mockedRenderer.mock.calls[0][0];
      rendererProps.handleRunScorer();

      expect(mockEvaluateTraces).not.toHaveBeenCalled();
    });

    it('should handle evaluation errors gracefully', async () => {
      mockEvaluateTraces.mockImplementation(() => Promise.reject(new Error('API error')));

      renderComponent();

      const rendererProps = mockedRenderer.mock.calls[0][0];
      rendererProps.handleRunScorer();

      // Error is caught and handled by the hook
      expect(mockEvaluateTraces).toHaveBeenCalledTimes(1);
    });

    it('should not call onScorerFinished when evaluation fails', async () => {
      const onScorerFinished = jest.fn();
      mockEvaluateTraces.mockRejectedValue(new Error('Network error'));

      renderComponent({ onScorerFinished });

      const rendererProps = mockedRenderer.mock.calls[0][0];
      rendererProps.handleRunScorer();

      expect(onScorerFinished).not.toHaveBeenCalled();
    });
  });
});
