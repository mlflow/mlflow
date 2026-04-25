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

// Mock the feature flag
jest.mock('../../../common/utils/FeatureUtils', () => ({
  isRunningScorersEnabled: () => true,
  isEvaluatingSessionsInScorersEnabled: () => true,
  isScorerModelSelectionEnabled: () => true,
  isScorerOutputTypeSelectorEnabled: () => false,
}));

// Mock the endpoint selector to avoid API calls (forbidden in unit tests)
jest.mock('../../components/EndpointSelector', () => ({
  EndpointSelector: () => <div data-testid="endpoint-selector" />,
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
}

function TestWrapper({ defaultValues, initialSelectedItemIds }: TestWrapperProps) {
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
              onFormSubmit={jest.fn()}
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
