import { jest, describe, beforeEach, test, expect } from '@jest/globals';
import { render } from '@testing-library/react';
import ExperimentTraceDetailRedirect from './ExperimentTraceDetailRedirect';
import { useNavigate, useParams, useSearchParams } from '../../../common/utils/RoutingUtils';
import { useGetExperimentQuery } from '../../hooks/useExperimentQuery';

jest.mock('../../../common/utils/RoutingUtils', () => ({
  useNavigate: jest.fn(),
  useParams: jest.fn(),
  useSearchParams: jest.fn(),
}));

jest.mock('../../hooks/useExperimentQuery', () => ({
  useGetExperimentQuery: jest.fn(),
}));

jest.mock('../../routes', () => ({
  __esModule: true,
  default: {
    getExperimentPageTracesTabRoute: (experimentId: string) => `/experiments/${experimentId}/traces`,
  },
}));

describe('ExperimentTraceDetailRedirect', () => {
  const mockNavigate = jest.fn<ReturnType<typeof useNavigate>>();

  const mockExperimentQuery = (overrides: Partial<ReturnType<typeof useGetExperimentQuery>> = {}) => {
    jest.mocked(useGetExperimentQuery).mockReturnValue({
      data: undefined,
      loading: false,
      refetch: jest.fn(),
      apolloError: undefined,
      apiError: undefined,
      ...overrides,
    } as ReturnType<typeof useGetExperimentQuery>);
  };

  beforeEach(() => {
    jest.clearAllMocks();
    jest.mocked(useNavigate).mockReturnValue(mockNavigate as any);
    mockExperimentQuery();
  });

  test('redirects V3 trace immediately without fetching experiment', () => {
    jest.mocked(useParams).mockReturnValue({ experimentId: '123', traceId: 'tr-abc123' });
    jest.mocked(useSearchParams).mockReturnValue([new URLSearchParams('o=999'), jest.fn()]);

    render(<ExperimentTraceDetailRedirect />);

    expect(useGetExperimentQuery).toHaveBeenCalledWith(expect.objectContaining({ options: { skip: true } }));
    expect(mockNavigate).toHaveBeenCalledWith('/experiments/123/traces?o=999&selectedEvaluationId=tr-abc123', {
      replace: true,
    });
  });

  test('redirects V4 trace with UC schema prefix', () => {
    jest.mocked(useParams).mockReturnValue({ experimentId: '456', traceId: 'deadbeef' });
    jest.mocked(useSearchParams).mockReturnValue([new URLSearchParams('o=999'), jest.fn()]);
    const mockExperimentData = {
      data: {
        tags: [{ key: 'mlflow.experiment.databricksTraceDestinationPath', value: 'main.default' }],
      },
    } as any;
    mockExperimentQuery(mockExperimentData);

    render(<ExperimentTraceDetailRedirect />);

    expect(useGetExperimentQuery).toHaveBeenCalledWith(expect.objectContaining({ options: { skip: false } }));
    expect(mockNavigate).toHaveBeenCalledWith(
      '/experiments/456/traces?o=999&selectedEvaluationId=trace%3A%2Fmain.default%2Fdeadbeef',
      { replace: true },
    );
  });

  test('waits for experiment data before redirecting V4 trace', () => {
    jest.mocked(useParams).mockReturnValue({ experimentId: '456', traceId: 'deadbeef' });
    jest.mocked(useSearchParams).mockReturnValue([new URLSearchParams('o=999'), jest.fn()]);
    mockExperimentQuery({ loading: true });

    render(<ExperimentTraceDetailRedirect />);

    expect(mockNavigate).not.toHaveBeenCalled();
  });

  test('redirects V4 trace without prefix when UC schema tag is missing', () => {
    jest.mocked(useParams).mockReturnValue({ experimentId: '456', traceId: 'deadbeef' });
    jest.mocked(useSearchParams).mockReturnValue([new URLSearchParams('o=999'), jest.fn()]);
    mockExperimentQuery({
      data: {
        experiment: { tags: [] },
      } as unknown as ReturnType<typeof useGetExperimentQuery>['data'],
    });

    render(<ExperimentTraceDetailRedirect />);

    expect(mockNavigate).toHaveBeenCalledWith('/experiments/456/traces?o=999&selectedEvaluationId=deadbeef', {
      replace: true,
    });
  });

  test('preserves existing search params through redirect', () => {
    jest.mocked(useParams).mockReturnValue({ experimentId: '123', traceId: 'tr-abc123' });
    jest
      .mocked(useSearchParams)
      .mockReturnValue([new URLSearchParams('o=999&startTimeLabel=LAST_7_DAYS&someFilter=value'), jest.fn()]);

    render(<ExperimentTraceDetailRedirect />);

    const navigatedUrl = mockNavigate.mock.calls[0][0] as unknown as string;
    const params = new URLSearchParams(navigatedUrl.split('?')[1]);
    expect(params.get('o')).toBe('999');
    expect(params.get('startTimeLabel')).toBe('LAST_7_DAYS');
    expect(params.get('someFilter')).toBe('value');
    expect(params.get('selectedEvaluationId')).toBe('tr-abc123');
  });

  test('does not redirect when experimentId is missing', () => {
    jest.mocked(useParams).mockReturnValue({ traceId: 'tr-abc123' });
    jest.mocked(useSearchParams).mockReturnValue([new URLSearchParams(), jest.fn()]);

    render(<ExperimentTraceDetailRedirect />);

    expect(mockNavigate).not.toHaveBeenCalled();
  });

  test('does not redirect when traceId is missing', () => {
    jest.mocked(useParams).mockReturnValue({ experimentId: '123' });
    jest.mocked(useSearchParams).mockReturnValue([new URLSearchParams(), jest.fn()]);

    render(<ExperimentTraceDetailRedirect />);

    expect(mockNavigate).not.toHaveBeenCalled();
  });

  test('renders nothing', () => {
    jest.mocked(useParams).mockReturnValue({ experimentId: '123', traceId: 'tr-abc123' });
    jest.mocked(useSearchParams).mockReturnValue([new URLSearchParams(), jest.fn()]);

    const { container } = render(<ExperimentTraceDetailRedirect />);

    expect(container.innerHTML).toBe('');
  });
});
