import { jest, beforeEach, describe, it, expect } from '@jest/globals';
import { screen, waitFor } from '@testing-library/react';
import { render } from '@databricks/testing-library';
import userEvent from '@testing-library/user-event';
import React, { useState } from 'react';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';
import { QueryClient, QueryClientProvider } from '../../../query-client/queryClient';

import { ModelTraceExplorerRightPaneTabs } from './ModelTraceExplorerRightPaneTabs';
import type { Assessment, ModelTrace, ModelTraceExplorerTab } from '../ModelTrace.types';
import { ModelSpanType } from '../ModelTrace.types';
import {
  ModelTraceExplorerViewStateProvider,
  useModelTraceExplorerViewState,
} from '../ModelTraceExplorerViewStateContext';
import { LabelingSchemaContextProvider, type LabelingSchema } from '../../assessments-pane-v2/LabelingSchemaContext';

// Mock the feature flags
jest.mock('../../FeatureUtils', () => ({
  shouldEnableTracesTabLabelingSchemas: () => true,
  shouldUseTracesV4API: () => false,
}));

// Mock global-settings for user
jest.mock('../../../global-settings/getUser', () => ({
  ...jest.requireActual<typeof import('../../../global-settings/getUser')>('../../../global-settings/getUser'),
  getUser: jest.fn(() => 'test-user@databricks.com'),
}));
jest.mock('../../../global-settings/getOrgID', () => ({
  ...jest.requireActual<typeof import('../../../global-settings/getOrgID')>('../../../global-settings/getOrgID'),
  getOrgID: jest.fn(() => '123456'),
}));

// Mock assessment that matches the categorical schema with "Good" selected
const MOCK_EXISTING_ASSESSMENT: Assessment = {
  assessment_id: 'assessment-1',
  assessment_name: 'quality-rating', // Must match schema name
  trace_id: 'trace-with-assessment',
  source: {
    source_type: 'HUMAN',
    source_id: 'test-user@databricks.com', // Must match mocked getUser()
  },
  create_time: '2025-01-01T00:00:00.000Z',
  last_update_time: '2025-01-01T00:00:00.000Z',
  feedback: {
    value: 'Good', // Pre-selected value
  },
};

// Create ModelTrace with a single span - assessments come from trace info
const createMockTrace = (traceId: string, spanId: string, assessments: Assessment[] = []): ModelTrace => ({
  data: {
    spans: [
      {
        context: { span_id: spanId, trace_id: traceId },
        parent_id: null,
        name: `span-${spanId}`,
        start_time: 0,
        end_time: 1000 * 1e6, // in nanoseconds
        span_type: ModelSpanType.CHAIN,
        status: { description: 'OK', status_code: 1 },
        attributes: {
          'mlflow.spanInputs': JSON.stringify({ query: `input-${spanId}` }),
          'mlflow.spanOutputs': JSON.stringify({ response: `output-${spanId}` }),
        },
        events: [],
      },
    ],
  },
  info: {
    trace_id: traceId,
    trace_location: {
      type: 'MLFLOW_EXPERIMENT',
      mlflow_experiment: { experiment_id: 'test-experiment' },
    },
    request_time: '2025-01-01T00:00:00.000Z',
    state: 'OK',
    trace_metadata: {},
    tags: {},
    assessments,
  },
});

// Trace with an existing assessment with "Good" selected
const MOCK_TRACE_WITH_ASSESSMENT = createMockTrace('trace-with-assessment', 'span-1', [MOCK_EXISTING_ASSESSMENT]);
// Trace without assessments
const MOCK_TRACE_WITHOUT_ASSESSMENT = createMockTrace('trace-without-assessment', 'span-2', []);

// Mock categorical labeling schema with 2 options
const MOCK_CATEGORICAL_SCHEMA: LabelingSchema = {
  name: 'quality-rating',
  type: 'FEEDBACK',
  title: 'Quality Rating',
  instruction: 'Rate the quality of the response',
  dataType: 'categorical',
  categorical: {
    options: ['Good', 'Bad'],
  },
};

// Custom wrapper that provides all necessary context
const TestWrapper = ({
  children,
  modelTrace,
  schemas = [MOCK_CATEGORICAL_SCHEMA],
}: {
  children: React.ReactNode;
  modelTrace: ModelTrace;
  schemas?: LabelingSchema[];
}) => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });

  return (
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <QueryClientProvider client={queryClient}>
          <ModelTraceExplorerViewStateProvider
            modelTrace={modelTrace}
            assessmentsPaneEnabled
            initialAssessmentsPaneCollapsed="force-open"
          >
            <LabelingSchemaContextProvider
              schemas={schemas}
              allAvailableSchemas={schemas}
              isLoading={false}
              onAddSchema={async () => {}}
              onRemoveSchema={async () => {}}
            >
              {children}
            </LabelingSchemaContextProvider>
          </ModelTraceExplorerViewStateProvider>
        </QueryClientProvider>
      </DesignSystemProvider>
    </IntlProvider>
  );
};

// Inner component that uses the view state context to get the active span
const RightPaneWithActiveSpan = ({
  activeTab,
  setActiveTab,
}: {
  activeTab: ModelTraceExplorerTab;
  setActiveTab: (tab: ModelTraceExplorerTab) => void;
}) => {
  const { rootNode } = useModelTraceExplorerViewState();
  return (
    <ModelTraceExplorerRightPaneTabs
      activeSpan={rootNode ?? undefined}
      searchFilter=""
      activeMatch={null}
      activeTab={activeTab}
      setActiveTab={setActiveTab}
    />
  );
};

// Component wrapper that allows changing traces (which changes the active span)
const TestComponent = ({
  initialTrace = MOCK_TRACE_WITH_ASSESSMENT,
  schemas = [MOCK_CATEGORICAL_SCHEMA],
}: {
  initialTrace?: ModelTrace;
  schemas?: LabelingSchema[];
}) => {
  const [currentTrace, setCurrentTrace] = useState(initialTrace);
  const [activeTab, setActiveTab] = useState<ModelTraceExplorerTab>('content');

  return (
    <div>
      <button data-testid="switch-to-trace-with-assessment" onClick={() => setCurrentTrace(MOCK_TRACE_WITH_ASSESSMENT)}>
        Switch to Trace with Assessment
      </button>
      <button
        data-testid="switch-to-trace-without-assessment"
        onClick={() => setCurrentTrace(MOCK_TRACE_WITHOUT_ASSESSMENT)}
      >
        Switch to Trace without Assessment
      </button>
      <TestWrapper modelTrace={currentTrace} schemas={schemas}>
        <RightPaneWithActiveSpan activeTab={activeTab} setActiveTab={setActiveTab} />
      </TestWrapper>
    </div>
  );
};

describe('ModelTraceExplorerRightPaneTabs - Assessment Reset on Span Change', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should show pre-populated assessment on trace with existing assessment', async () => {
    render(<TestComponent initialTrace={MOCK_TRACE_WITH_ASSESSMENT} />);

    // Wait for the assessments pane to be rendered
    await waitFor(() => {
      expect(screen.getByTestId('assessments-pane-v2')).toBeInTheDocument();
    });

    // The span with assessment should have "Good" pre-selected
    const goodRadio = screen.getByRole('radio', { name: 'Good' });
    const badRadio = screen.getByRole('radio', { name: 'Bad' });

    await waitFor(() => {
      expect(goodRadio).toBeChecked();
    });
    expect(badRadio).not.toBeChecked();
  });

  it('should reset form when switching from trace with assessment to trace without assessment', async () => {
    const user = userEvent.setup();

    render(<TestComponent initialTrace={MOCK_TRACE_WITH_ASSESSMENT} />);

    // Wait for the assessments pane to be rendered
    await waitFor(() => {
      expect(screen.getByTestId('assessments-pane-v2')).toBeInTheDocument();
    });

    // Initially "Good" should be pre-selected from the existing assessment
    const goodRadio = screen.getByRole('radio', { name: 'Good' });
    await waitFor(() => {
      expect(goodRadio).toBeChecked();
    });

    // Switch to trace without assessment
    await user.click(screen.getByTestId('switch-to-trace-without-assessment'));

    // Wait for the component to re-render - the form should be reset (no selection)
    await waitFor(() => {
      const newGoodRadio = screen.getByRole('radio', { name: 'Good' });
      const newBadRadio = screen.getByRole('radio', { name: 'Bad' });
      expect(newGoodRadio).not.toBeChecked();
      expect(newBadRadio).not.toBeChecked();
    });
  });

  it('should restore pre-populated assessment when switching back to trace with assessment', async () => {
    const user = userEvent.setup();

    render(<TestComponent initialTrace={MOCK_TRACE_WITH_ASSESSMENT} />);

    // Wait for the assessments pane to be rendered
    await waitFor(() => {
      expect(screen.getByTestId('assessments-pane-v2')).toBeInTheDocument();
    });

    // Initially "Good" should be pre-selected
    await waitFor(() => {
      expect(screen.getByRole('radio', { name: 'Good' })).toBeChecked();
    });

    // Switch to trace without assessment
    await user.click(screen.getByTestId('switch-to-trace-without-assessment'));

    // Verify form is reset
    await waitFor(() => {
      expect(screen.getByRole('radio', { name: 'Good' })).not.toBeChecked();
    });

    // Select "Bad" on the trace without assessment
    await user.click(screen.getByRole('radio', { name: 'Bad' }));
    await waitFor(() => {
      expect(screen.getByRole('radio', { name: 'Bad' })).toBeChecked();
    });

    // Switch back to trace with assessment
    await user.click(screen.getByTestId('switch-to-trace-with-assessment'));

    // "Good" should be pre-selected again (from the assessment data)
    await waitFor(() => {
      const goodRadio = screen.getByRole('radio', { name: 'Good' });
      const badRadio = screen.getByRole('radio', { name: 'Bad' });
      expect(goodRadio).toBeChecked();
      expect(badRadio).not.toBeChecked();
    });
  });
});
