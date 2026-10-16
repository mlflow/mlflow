import { describe, expect, jest, test, beforeEach } from '@jest/globals';
import { renderWithIntl, waitFor } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { DesignSystemProvider } from '@databricks/design-system';
import { useSearchMlflowTraces } from '@databricks/web-shared/genai-traces-table';
import type { ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';

import ExperimentSingleChatSessionPage, { findSessionRedirectTrace } from './ExperimentSingleChatSessionPage';

const mockNavigate = jest.fn();
let mockSearch = '';

jest.mock('@mlflow/mlflow/src/common/utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('@mlflow/mlflow/src/common/utils/FeatureUtils')>(
    '@mlflow/mlflow/src/common/utils/FeatureUtils',
  ),
  shouldEnableNewEvalRunsUi: jest.fn(() => true),
}));

jest.mock('@databricks/web-shared/genai-traces-table', () => ({
  ...jest.requireActual<typeof import('@databricks/web-shared/genai-traces-table')>(
    '@databricks/web-shared/genai-traces-table',
  ),
  useSearchMlflowTraces: jest.fn(() => ({ data: undefined, isLoading: true })),
}));

jest.mock('@mlflow/mlflow/src/common/utils/RoutingUtils', () => ({
  ...jest.requireActual<typeof import('@mlflow/mlflow/src/common/utils/RoutingUtils')>(
    '@mlflow/mlflow/src/common/utils/RoutingUtils',
  ),
  useLocation: () => ({ search: mockSearch, hash: '' }),
  useNavigate: () => mockNavigate,
  useParams: () => ({ experimentId: 'exp-1', sessionId: 'session-1' }),
}));

jest.mock('../../../hooks/useExperimentQuery', () => ({
  useGetExperimentQuery: jest.fn(() => ({ loading: false })),
}));

const makeTurn = (traceId: string, requestTime: string) =>
  ({
    trace_id: traceId,
    trace_location: { type: 'MLFLOW_EXPERIMENT', mlflow_experiment: { experiment_id: 'exp-1' } },
    request_time: requestTime,
  }) as ModelTraceInfoV3;
const TURNS = [makeTurn('tr-2', '2025-01-01T00:01:00.000Z'), makeTurn('tr-1', '2025-01-01T00:00:00.000Z')];

const renderPage = () =>
  renderWithIntl(
    <DesignSystemProvider>
      <ExperimentSingleChatSessionPage />
    </DesignSystemProvider>,
  );

describe('ExperimentSingleChatSessionPage with the new eval runs UI', () => {
  beforeEach(() => {
    mockNavigate.mockClear();
    mockSearch = '';
  });

  test('waits for the session traces before redirecting', () => {
    jest.mocked(useSearchMlflowTraces).mockReturnValue({ data: undefined, isLoading: true, isFetching: true });
    renderPage();
    expect(mockNavigate).not.toHaveBeenCalled();
  });

  test('opens the session in the Traces tab drawer on its first turn, keeping the link parameters', async () => {
    jest.mocked(useSearchMlflowTraces).mockReturnValue({ data: TURNS, isLoading: false, isFetching: false });
    mockSearch = '?q=hello';
    renderPage();

    await waitFor(() => expect(mockNavigate).toHaveBeenCalled());
    const [target, options] = mockNavigate.mock.calls[0] as [string, { replace: boolean }];
    const url = new URL(target, 'http://localhost');
    expect(url.pathname).toBe('/experiments/exp-1/traces');
    expect(Object.fromEntries(url.searchParams)).toEqual({
      q: 'hello',
      groupBy: 'session',
      startTimeLabel: 'ALL',
      traceDrawerView: 'session',
      traceDrawerSessionId: 'session-1',
      traceId: 'tr-1',
    });
    expect(options).toEqual({ replace: true });
  });

  test('opens the linked turn when the link names one', async () => {
    jest.mocked(useSearchMlflowTraces).mockReturnValue({ data: TURNS, isLoading: false, isFetching: false });
    mockSearch = '?selectedTraceId=tr-2';
    renderPage();

    await waitFor(() => expect(mockNavigate).toHaveBeenCalled());
    const url = new URL(mockNavigate.mock.calls[0][0] as string, 'http://localhost');
    expect(url.searchParams.get('traceId')).toBe('tr-2');
    expect(url.searchParams.get('selectedTraceId')).toBeNull();
  });
});

describe('findSessionRedirectTrace', () => {
  test('prefers the selected trace and falls back to the earliest turn', () => {
    expect(findSessionRedirectTrace(TURNS, 'tr-2')?.trace_id).toBe('tr-2');
    expect(findSessionRedirectTrace(TURNS, 'missing')?.trace_id).toBe('tr-1');
    expect(findSessionRedirectTrace([], null)).toBeUndefined();
  });
});
