import { render, waitFor } from '@testing-library/react';
import { IntlProvider } from '@databricks/i18n';
import { FormProvider, useForm } from 'react-hook-form';
import SampleScorerOutputPanelContainer from './SampleScorerOutputPanelContainer';
import { useEvaluateTraces } from './useEvaluateTraces';
import type { TraceJudgeEvaluationResult } from './useEvaluateTraces.common';
import { type JudgeEvaluationResult } from './useEvaluateTraces.common';
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
import {
  createTraceLocationForExperiment,
  createTraceLocationForDestinationPath,
} from '@databricks/web-shared/genai-traces-table';
import type { ModelTraceSearchLocation } from '@databricks/web-shared/model-trace-explorer';

jest.mock('./useEvaluateTraces');
jest.mock('./SampleScorerOutputPanelRenderer');
jest.mock('./SampleScorerTracesToEvaluatePicker', () => ({
  SampleScorerTracesToEvaluatePicker: () => null,
}));
jest.mock('../../../common/utils/FeatureUtils', () => ({
  isRunningAgenticJudgesEnabled: () => false,
  isRunningAllScorerTemplatesEnabled: () => false,
  isEvaluatingSessionsInScorersEnabled: () => false,
  isScorerModelSelectionEnabled: () => false,
  shouldSupportRunningDatabricksProviderJudgesFromUI: () => true,
}));
jest.mock('../../../gateway/utils/gatewayUtils', () => ({
  ModelProvider: { GATEWAY: 'gateway', DATABRICKS: 'databricks', OTHER: 'other' },
  getModelProvider: (model: string | undefined) => {
    if (!model || model.startsWith('gateway:/')) return 'gateway';
    if (model.startsWith('databricks:/')) return 'databricks';
    return 'other';
  },
}));
const experimentId = 'exp-123';

const mockedUseEvaluateTraces = jest.mocked(useEvaluateTraces);
const mockedRenderer = jest.mocked(SampleScorerOutputPanelRenderer);

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
  selectedItemIds?: string[];
  onSelectedItemIdsChange?: (itemIds: string[]) => void;
  isSessionLevelScorer?: boolean;
}

function TestWrapper({
  defaultValues,
  onScorerFinished,
  selectedItemIds = [],
  onSelectedItemIdsChange = jest.fn(),
  isSessionLevelScorer,
}: TestWrapperProps) {
  const form = useForm<ScorerFormData>({
    defaultValues: {
      name: 'Test Scorer',
      instructions: 'Test instructions',
      llmTemplate: LLM_TEMPLATE.CUSTOM,
      isInstructionsJudge: true,
      sampleRate: 100,
      scorerType: 'llm',
      model: 'gateway:/some-model',
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
          selectedItemIds={selectedItemIds}
          onSelectedItemIdsChange={onSelectedItemIdsChange}
          isSessionLevelScorer={isSessionLevelScorer}
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
        latestEvaluation: null,
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
          // Button is disabled initially because no traces are selected
          isRunScorerDisabled: true,
          error: null,
          currentEvalResultIndex: 0,
          totalTraces: 0,
          selectedItemIds: [],
        }),
        expect.anything(),
      );
    });

    it('should pass evaluation results to renderer when data is available', () => {
      const mockResults = [createMockEvalResult('trace-1'), createMockEvalResult('trace-2')];

      mockedUseEvaluateTraces.mockReturnValue([
        mockEvaluateTraces,
        {
          latestEvaluation: mockResults,
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
          latestEvaluation: null,
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
          latestEvaluation: null,
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
          latestEvaluation: null,
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

  describe('Run scorer disabled reason precedence', () => {
    // Helper to get the tooltip from the last renderer call
    const getDisabledTooltip = () => {
      const lastCall = mockedRenderer.mock.calls[mockedRenderer.mock.calls.length - 1];
      return lastCall[0].runScorerDisabledTooltip;
    };

    it('should show "enter instructions" before "select traces" for instructions judge with no instructions', () => {
      // Both: no instructions AND no traces selected
      renderComponent({
        defaultValues: { instructions: '', isInstructionsJudge: true, llmTemplate: LLM_TEMPLATE.CUSTOM },
        selectedItemIds: [],
      });

      expect(getDisabledTooltip()).toMatch(/enter instructions/i);
    });

    it('should show "unsupported template" as highest precedence (before session-level and select traces)', () => {
      // Unsupported template AND session-level AND no traces selected — unsupported template wins
      renderComponent({
        defaultValues: {
          isInstructionsJudge: false,
          llmTemplate: LLM_TEMPLATE.EQUIVALENCE,
          instructions: 'some instructions',
          evaluationScope: ScorerEvaluationScope.SESSIONS,
        },
        selectedItemIds: [],
      });

      expect(getDisabledTooltip()).toMatch(/not yet supported/i);
    });

    it('should show "session level not supported" before "enter instructions" and "select traces"', () => {
      // Session-level AND no instructions AND no traces — session-level wins over instructions/traces
      renderComponent({
        defaultValues: {
          instructions: '',
          isInstructionsJudge: true,
          llmTemplate: LLM_TEMPLATE.CUSTOM,
        },
        isSessionLevelScorer: true,
        selectedItemIds: [],
      });

      expect(getDisabledTooltip()).toMatch(/session/i);
    });

    it('should show "select traces" when judge config is valid but no traces selected', () => {
      // Valid instructions judge, but no traces
      renderComponent({
        defaultValues: {
          instructions: 'Evaluate the response',
          isInstructionsJudge: true,
          llmTemplate: LLM_TEMPLATE.CUSTOM,
        },
        selectedItemIds: [],
      });

      expect(getDisabledTooltip()).toMatch(/select traces/i);
    });

    it('should not be disabled when judge config is valid and traces are selected', () => {
      renderComponent({
        defaultValues: {
          instructions: 'Evaluate the response',
          isInstructionsJudge: true,
          llmTemplate: LLM_TEMPLATE.CUSTOM,
        },
        selectedItemIds: ['trace-1'],
      });

      const lastCall = mockedRenderer.mock.calls[mockedRenderer.mock.calls.length - 1];
      expect(lastCall[0].isRunScorerDisabled).toBe(false);
      expect(lastCall[0].runScorerDisabledTooltip).toBeUndefined();
    });

    it('should show "session level not supported" for session-level scorer with valid config', () => {
      renderComponent({
        defaultValues: {
          instructions: 'Evaluate the session',
          isInstructionsJudge: true,
          llmTemplate: LLM_TEMPLATE.CUSTOM,
        },
        isSessionLevelScorer: true,
        selectedItemIds: ['trace-1'],
      });

      expect(getDisabledTooltip()).toMatch(/session/i);
    });

    it('should show "trace variable not supported" for instructions containing {{ trace }}', () => {
      // isRunningAgenticJudgesEnabled is mocked to false
      renderComponent({
        defaultValues: {
          instructions: 'Evaluate {{ trace }} for quality',
          isInstructionsJudge: true,
          llmTemplate: LLM_TEMPLATE.CUSTOM,
        },
        selectedItemIds: ['trace-1'],
      });

      expect(getDisabledTooltip()).toMatch(/trace variable/i);
    });

    it('should show "guidelines empty" before "select traces" for guidelines template with no guidelines', () => {
      renderComponent({
        defaultValues: {
          isInstructionsJudge: false,
          llmTemplate: LLM_TEMPLATE.GUIDELINES,
          guidelines: '',
          instructions: '',
        },
        selectedItemIds: [],
      });

      expect(getDisabledTooltip()).toMatch(/guidelines.*empty/i);
    });

    it('should show "unsupported template" for retrieval relevance (caught by unsupported template check)', () => {
      // RETRIEVAL_RELEVANCE has no ASSESSMENT_NAME_TEMPLATE_MAPPING entry,
      // so isUnsupportedTemplate fires before the specific retrieval relevance check
      renderComponent({
        defaultValues: {
          isInstructionsJudge: false,
          llmTemplate: LLM_TEMPLATE.RETRIEVAL_RELEVANCE,
          instructions: '',
        },
        selectedItemIds: [],
      });

      expect(getDisabledTooltip()).toMatch(/not yet supported/i);
    });
  });
});
