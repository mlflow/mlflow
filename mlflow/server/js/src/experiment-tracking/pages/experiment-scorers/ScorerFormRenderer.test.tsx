import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { IntlProvider } from '@databricks/i18n';
import { FormProvider, useForm } from 'react-hook-form';
import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';
import { DesignSystemProvider } from '@databricks/design-system';
import ScorerFormRenderer from './ScorerFormRenderer';
import type { ScorerFormData } from './utils/scorerTransformUtils';
import { SCORER_FORM_MODE, ScorerEvaluationScope } from './constants';
import { jest, describe, beforeEach, it, expect } from '@jest/globals';
import type { Endpoint } from '../../../gateway/types';

let mockEndpoints: Endpoint[] = [];

// Mock the feature flag
jest.mock('../../../common/utils/FeatureUtils', () => ({
  isRunningScorersEnabled: () => true,
  isEvaluatingSessionsInScorersEnabled: () => true,
  isScorerModelSelectionEnabled: () => true,
  isScorerOutputTypeSelectorEnabled: () => false,
}));

// Mock the endpoint selector to avoid API calls (forbidden in unit tests)
jest.mock('../../components/EndpointSelector', () => ({
  EndpointSelector: ({
    allowSystemOne,
    showDisabledSystemOne,
    systemOneDisabledReason,
  }: {
    allowSystemOne?: boolean;
    showDisabledSystemOne?: boolean;
    systemOneDisabledReason?: string;
  }) => (
    <div
      data-testid="endpoint-selector"
      data-allow-system-one={String(allowSystemOne ?? false)}
      data-show-disabled-system-one={String(showDisabledSystemOne ?? false)}
      data-system-one-disabled-reason={systemOneDisabledReason}
    />
  ),
}));

jest.mock('../../../gateway/hooks/useEndpointsQuery', () => ({
  useEndpointsQuery: () => ({ data: mockEndpoints }),
}));

// Mock useExperimentIds used by ModelSectionRenderer for cache invalidation
jest.mock('../../components/experiment-page/hooks/useExperimentIds', () => ({
  useExperimentIds: () => ['exp-123'],
}));

// Mock to avoid transitive @databricks/web-shared/comlink resolution failure via SelectTracesModal
jest.mock('./SampleScorerOutputPanelContainer', () => ({
  __esModule: true,
  default: ({
    selectedItemIds,
    isSessionLevelScorer,
  }: {
    selectedItemIds: string[];
    isSessionLevelScorer: boolean;
  }) => (
    <div data-testid="sample-scorer-output-panel">
      {selectedItemIds.length > 0
        ? `${selectedItemIds.length} session selected`
        : isSessionLevelScorer
          ? 'Select sessions'
          : 'Select traces'}
    </div>
  ),
}));

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { retry: false },
  },
});

interface TestWrapperProps {
  defaultValues?: Partial<ScorerFormData>;
  initialSelectedItemIds?: string[];
  onFormSubmit?: (data: ScorerFormData) => void;
}

function TestWrapper({ defaultValues, initialSelectedItemIds, onFormSubmit = jest.fn() }: TestWrapperProps) {
  const form = useForm<ScorerFormData>({
    defaultValues: {
      name: 'Test Scorer',
      instructions: 'Test instructions',
      llmTemplate: 'Custom',
      sampleRate: 100,
      scorerType: 'llm',
      model: 'gateway:/some-model',
      evaluationScope: ScorerEvaluationScope.TRACES,
      ...defaultValues,
    },
  });

  return (
    <QueryClientProvider client={queryClient}>
      <DesignSystemProvider>
        <IntlProvider locale="en">
          <FormProvider {...form}>
            <ScorerFormRenderer
              mode={SCORER_FORM_MODE.CREATE}
              handleSubmit={form.handleSubmit}
              onFormSubmit={onFormSubmit}
              control={form.control}
              setValue={form.setValue}
              getValues={form.getValues}
              scorerType="llm"
              mutation={{ isLoading: false, error: null }}
              componentError={null}
              handleCancel={jest.fn()}
              isSubmitDisabled={false}
              experimentId="exp-123"
              initialSelectedItemIds={initialSelectedItemIds}
            />
          </FormProvider>
        </IntlProvider>
      </DesignSystemProvider>
    </QueryClientProvider>
  );
}

describe('ScorerFormRenderer', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockEndpoints = [];
  });

  it('waits for name entry to finish before opening evaluation criteria', async () => {
    const user = userEvent.setup();
    render(<TestWrapper defaultValues={{ name: '' }} />);

    const generalSection = screen.getByRole('button', { name: /General/ });
    const evaluationCriteriaSection = screen.getByRole('button', { name: /Evaluation criteria/ });
    await user.type(screen.getByPlaceholderText('Custom'), 'My judge');

    expect(generalSection).toHaveAttribute('aria-expanded', 'true');
    expect(evaluationCriteriaSection).toHaveAttribute('aria-expanded', 'false');

    await user.tab();

    expect(generalSection).toHaveAttribute('aria-expanded', 'false');
    expect(evaluationCriteriaSection).toHaveAttribute('aria-expanded', 'true');
  });

  it('explains that TypeSafe endpoints do not support the trace variable', async () => {
    const user = userEvent.setup();
    render(<TestWrapper defaultValues={{ instructions: 'Inspect {{ trace }}', outputTypeKind: 'bool' }} />);
    await user.click(screen.getByRole('button', { name: /Evaluation criteria/ }));

    expect(screen.getByTestId('endpoint-selector')).toHaveAttribute('data-allow-system-one', 'false');
    expect(screen.getByTestId('endpoint-selector')).toHaveAttribute('data-show-disabled-system-one', 'true');
    expect(screen.getByTestId('endpoint-selector')).toHaveAttribute(
      'data-system-one-disabled-reason',
      'System One endpoints do not support {{ trace }}. Remove {{ trace }} from the instructions to use a System One endpoint.',
    );
  });

  it('allows TypeSafe endpoints for structured boolean judges', async () => {
    const user = userEvent.setup();
    render(<TestWrapper defaultValues={{ instructions: 'Inspect {{ outputs }}', outputTypeKind: 'bool' }} />);
    await user.click(screen.getByRole('button', { name: /Evaluation criteria/ }));

    expect(screen.getByTestId('endpoint-selector')).toHaveAttribute('data-allow-system-one', 'true');
  });

  it('uses the known categorical output type when Completeness is selected', async () => {
    const user = userEvent.setup();
    render(
      <TestWrapper
        defaultValues={{
          llmTemplate: 'Custom',
          instructions: '',
          outputTypeKind: 'default',
          isInstructionsJudge: true,
        }}
      />,
    );

    await user.click(screen.getByRole('button', { name: /Evaluation criteria/ }));
    expect(screen.getByTestId('endpoint-selector')).toHaveAttribute('data-allow-system-one', 'false');
    await user.click(screen.getByRole('combobox', { name: 'LLM judge' }));
    await user.click(screen.getByText('Completeness'));

    expect(screen.getByTestId('endpoint-selector')).toHaveAttribute('data-allow-system-one', 'true');
  });

  it('uses the known categorical output type when Equivalence is selected', async () => {
    const user = userEvent.setup();
    render(
      <TestWrapper
        defaultValues={{
          llmTemplate: 'Custom',
          instructions: '',
          outputTypeKind: 'default',
          isInstructionsJudge: true,
        }}
      />,
    );

    await user.click(screen.getByRole('button', { name: /Evaluation criteria/ }));
    expect(screen.getByTestId('endpoint-selector')).toHaveAttribute('data-allow-system-one', 'false');
    await user.click(screen.getByRole('combobox', { name: 'LLM judge' }));
    await user.click(screen.getByText('Equivalence'));

    expect(screen.getByTestId('endpoint-selector')).toHaveAttribute('data-allow-system-one', 'true');
  });

  it('excludes TypeSafe endpoints until a categorical judge has nonblank options', async () => {
    const user = userEvent.setup();
    const { unmount } = render(
      <TestWrapper
        defaultValues={{
          instructions: 'Inspect {{ outputs }}',
          outputTypeKind: 'categorical',
          categoricalOptions: ' \n ',
        }}
      />,
    );
    await user.click(screen.getByRole('button', { name: /Evaluation criteria/ }));
    expect(screen.getByTestId('endpoint-selector')).toHaveAttribute('data-allow-system-one', 'false');
    expect(screen.getByTestId('endpoint-selector')).toHaveAttribute(
      'data-system-one-disabled-reason',
      'System One endpoints require Boolean output or Categorical output with at least one option.',
    );
    unmount();

    render(
      <TestWrapper
        defaultValues={{
          instructions: 'Inspect {{ outputs }}',
          outputTypeKind: 'categorical',
          categoricalOptions: 'good\nbad',
        }}
      />,
    );
    await user.click(screen.getByRole('button', { name: /Evaluation criteria/ }));
    expect(screen.getByTestId('endpoint-selector')).toHaveAttribute('data-allow-system-one', 'true');
  });

  it('rejects a mixed TypeSafe and chat endpoint for a structured boolean judge', async () => {
    const user = userEvent.setup();
    const onFormSubmit = jest.fn();
    mockEndpoints = [
      {
        name: 'some-model',
        model_mappings: [{ model_definition: { provider: 'typesafe' } }, { model_definition: { provider: 'openai' } }],
      } as Endpoint,
    ];

    render(
      <TestWrapper
        defaultValues={{ instructions: 'Inspect {{ outputs }}', outputTypeKind: 'bool' }}
        onFormSubmit={onFormSubmit}
      />,
    );
    await user.click(screen.getByRole('button', { name: /Evaluation criteria/ }));
    await user.click(screen.getByRole('button', { name: 'Create judge' }));

    expect(await screen.findByText('Mixed TypeSafe and chat endpoints are not supported.')).toBeInTheDocument();
    expect(onFormSubmit).not.toHaveBeenCalled();
  });

  describe('Preset Modal Behavior', () => {
    it('should preset scope to SESSIONS when initialScope is SESSIONS', () => {
      render(<TestWrapper defaultValues={{ evaluationScope: ScorerEvaluationScope.SESSIONS }} />);

      const sessionsRadio = screen.getByRole('radio', { name: /sessions/i });
      expect(sessionsRadio).toBeChecked();
    });

    it('should preset selectedItemIds when initialSelectedItemIds is provided', () => {
      render(
        <TestWrapper
          defaultValues={{ evaluationScope: ScorerEvaluationScope.SESSIONS }}
          initialSelectedItemIds={['session-123']}
        />,
      );

      // Button should show "1 session selected" instead of "Select sessions"
      expect(screen.getByText('1 session selected')).toBeInTheDocument();
    });

    it('should clear selectedItemIds when user changes scope', async () => {
      const user = userEvent.setup();

      render(
        <TestWrapper
          defaultValues={{ evaluationScope: ScorerEvaluationScope.SESSIONS }}
          initialSelectedItemIds={['session-123']}
        />,
      );

      // Initially shows "1 session selected"
      expect(screen.getByText('1 session selected')).toBeInTheDocument();

      // Click on Traces radio to change scope
      const tracesRadio = screen.getByRole('radio', { name: /traces/i });
      await user.click(tracesRadio);

      // After scope change, selected items should be cleared, showing "Select traces"
      expect(screen.getByText('Select traces')).toBeInTheDocument();
    });
  });
});
