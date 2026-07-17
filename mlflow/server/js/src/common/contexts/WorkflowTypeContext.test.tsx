/* eslint-disable jest/no-standalone-expect */
import { describe, jest, test, expect, beforeEach } from '@jest/globals';
import { act, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Route, Routes } from '../utils/RoutingUtils';
import { QueryClient, QueryClientProvider } from '../utils/reactQueryHooks';
import { renderWithDesignSystem } from '../utils/TestUtils.react18';
import { WorkflowType, WorkflowTypeProvider, useWorkflowType } from './WorkflowTypeContext';
import { useGetExperimentQuery } from '../../experiment-tracking/hooks/useExperimentQuery';
import { EXPERIMENT_KIND_TAG_KEY } from '../../experiment-tracking/utils/ExperimentKindUtils';

const WORKFLOW_TYPE_STORAGE_KEY_V1 = 'mlflow.workflowType_v1';
const seedWorkflowType = (value: WorkflowType) =>
  window.localStorage.setItem(WORKFLOW_TYPE_STORAGE_KEY_V1, JSON.stringify(value));

const mockNavigate = jest.fn();

jest.mock('../../experiment-tracking/hooks/useExperimentQuery', () => ({
  useGetExperimentQuery: jest.fn(() => ({ data: undefined, loading: false })),
}));

// Mock the same module specifier the provider imports (``../utils/RoutingUtils``)
// so the ``useNavigate`` override is guaranteed to apply to the code under test.
jest.mock('../utils/RoutingUtils', () => ({
  ...jest.requireActual<typeof import('../utils/RoutingUtils')>('../utils/RoutingUtils'),
  useNavigate: () => mockNavigate,
}));

const mockUseGetExperimentQuery = jest.mocked(useGetExperimentQuery);

const experimentWithKind = (kind: string | undefined, experimentId = '42') => ({
  data: {
    experimentId,
    tags: kind === undefined ? [] : [{ key: EXPERIMENT_KIND_TAG_KEY, value: kind }],
  },
  loading: false,
});

const WorkflowTypeProbe = () => {
  const { workflowType, setWorkflowType } = useWorkflowType();
  return (
    <div>
      <span data-testid="workflow-type">{workflowType}</span>
      <button type="button" onClick={() => setWorkflowType(WorkflowType.MACHINE_LEARNING)}>
        pick-ml
      </button>
      <button type="button" onClick={() => setWorkflowType(WorkflowType.GENAI)}>
        pick-genai
      </button>
    </div>
  );
};

describe('WorkflowTypeProvider experiment-kind sync', () => {
  const routerTree = (path: string) => {
    const queryClient = new QueryClient();
    return (
      <QueryClientProvider client={queryClient}>
        <MemoryRouter initialEntries={[path]}>
          <Routes>
            <Route
              path="/experiments/:experimentId/*"
              element={
                <WorkflowTypeProvider>
                  <WorkflowTypeProbe />
                </WorkflowTypeProvider>
              }
            />
            <Route
              path="*"
              element={
                <WorkflowTypeProvider>
                  <WorkflowTypeProbe />
                </WorkflowTypeProvider>
              }
            />
          </Routes>
        </MemoryRouter>
      </QueryClientProvider>
    );
  };

  const renderAt = (path: string) => renderWithDesignSystem(routerTree(path));

  beforeEach(() => {
    jest.clearAllMocks();
    // Reset the persisted workflow type between tests.
    seedWorkflowType(WorkflowType.GENAI);
    mockUseGetExperimentQuery.mockReturnValue({ data: undefined, loading: false } as any);
  });

  test('syncs to MACHINE_LEARNING when opening a custom_model_development experiment', async () => {
    mockUseGetExperimentQuery.mockReturnValue(experimentWithKind('custom_model_development') as any);
    renderAt('/experiments/42/runs');
    expect(await screen.findByTestId('workflow-type')).toHaveTextContent(WorkflowType.MACHINE_LEARNING);
  });

  test('syncs to MACHINE_LEARNING for inferred + other ML kinds', async () => {
    for (const kind of [
      'custom_model_development_inferred',
      'finetuning',
      'regression',
      'classification',
      'forecasting',
      'automl',
    ]) {
      mockUseGetExperimentQuery.mockReturnValue(experimentWithKind(kind) as any);
      const { unmount } = renderAt('/experiments/42/runs');
      expect(await screen.findByTestId('workflow-type')).toHaveTextContent(WorkflowType.MACHINE_LEARNING);
      unmount();
    }
  });

  test('syncs to GENAI when opening a genai_development experiment', async () => {
    // Start persisted in ML so we can observe the flip back to GENAI.
    seedWorkflowType(WorkflowType.MACHINE_LEARNING);
    mockUseGetExperimentQuery.mockReturnValue(experimentWithKind('genai_development', '7') as any);
    renderAt('/experiments/7/traces');
    expect(await screen.findByTestId('workflow-type')).toHaveTextContent(WorkflowType.GENAI);
  });

  test('leaves the current workflow type untouched when the experiment kind tag is absent', async () => {
    seedWorkflowType(WorkflowType.MACHINE_LEARNING);
    mockUseGetExperimentQuery.mockReturnValue(experimentWithKind(undefined, '9') as any);
    renderAt('/experiments/9/runs');
    // No definite mapping -> keep whatever was there (persisted ML here), do not force GENAI.
    expect(await screen.findByTestId('workflow-type')).toHaveTextContent(WorkflowType.MACHINE_LEARNING);
  });

  test('leaves the current workflow type untouched for no_inferred_type / empty kinds', async () => {
    seedWorkflowType(WorkflowType.MACHINE_LEARNING);
    for (const kind of ['no_inferred_type', '']) {
      mockUseGetExperimentQuery.mockReturnValue(experimentWithKind(kind, '9') as any);
      const { unmount } = renderAt('/experiments/9/runs');
      expect(await screen.findByTestId('workflow-type')).toHaveTextContent(WorkflowType.MACHINE_LEARNING);
      unmount();
    }
  });

  test('does not sync to a stale experiment kind when the query still holds the previous experiment', async () => {
    // Apollo keeps returning the previously-observed experiment until the new
    // one resolves. If the route is already on the genai experiment (7) but the
    // query data still reflects the classic-ML experiment (id 42), the sync must
    // NOT flip the toggle to MACHINE_LEARNING based on the stale record.
    seedWorkflowType(WorkflowType.GENAI);
    mockUseGetExperimentQuery.mockReturnValue(experimentWithKind('custom_model_development', '42') as any);
    renderAt('/experiments/7/traces');
    expect(await screen.findByTestId('workflow-type')).toHaveTextContent(WorkflowType.GENAI);
  });

  test('does not sync (or navigate) when not inside an experiment', async () => {
    mockUseGetExperimentQuery.mockReturnValue(experimentWithKind('custom_model_development') as any);
    renderAt('/experiments');
    expect(await screen.findByTestId('workflow-type')).toHaveTextContent(WorkflowType.GENAI);
    expect(mockNavigate).not.toHaveBeenCalled();
  });

  test('auto-sync does not trigger a navigation redirect', async () => {
    mockUseGetExperimentQuery.mockReturnValue(experimentWithKind('custom_model_development') as any);
    renderAt('/experiments/42/runs');
    expect(await screen.findByTestId('workflow-type')).toHaveTextContent(WorkflowType.MACHINE_LEARNING);
    // The auto-sync must NOT bounce the user off their deep link.
    expect(mockNavigate).not.toHaveBeenCalled();
  });

  test('manual override sticks within the same experiment (kind tag does not clobber it)', async () => {
    mockUseGetExperimentQuery.mockReturnValue(experimentWithKind('genai_development', '7') as any);
    renderAt('/experiments/7/traces');
    expect(await screen.findByTestId('workflow-type')).toHaveTextContent(WorkflowType.GENAI);

    // User manually flips to Model training while staying on the same experiment.
    await act(async () => {
      await userEvent.click(screen.getByText('pick-ml'));
    });
    expect(screen.getByTestId('workflow-type')).toHaveTextContent(WorkflowType.MACHINE_LEARNING);
  });

  test('manual toggle still navigates to the experiment page (existing behavior preserved)', async () => {
    mockUseGetExperimentQuery.mockReturnValue(experimentWithKind('genai_development', '7') as any);
    renderAt('/experiments/7/traces');
    expect(await screen.findByTestId('workflow-type')).toHaveTextContent(WorkflowType.GENAI);

    await act(async () => {
      await userEvent.click(screen.getByText('pick-ml'));
    });
    // The user-driven toggle should redirect to the experiment default tab.
    expect(mockNavigate).toHaveBeenCalled();
  });

  test('reconciles when the experiment kind tag is populated later on the same experiment', async () => {
    seedWorkflowType(WorkflowType.GENAI);
    // First render: experiment loaded but kind tag not yet populated (ambiguous).
    mockUseGetExperimentQuery.mockReturnValue(experimentWithKind(undefined, '42') as any);
    const { rerender } = renderAt('/experiments/42/runs');
    expect(await screen.findByTestId('workflow-type')).toHaveTextContent(WorkflowType.GENAI);

    // Kind inference lands the tag afterwards while staying on the same experiment.
    mockUseGetExperimentQuery.mockReturnValue(experimentWithKind('custom_model_development', '42') as any);
    rerender(routerTree('/experiments/42/runs'));
    expect(await screen.findByTestId('workflow-type')).toHaveTextContent(WorkflowType.MACHINE_LEARNING);
  });
});
