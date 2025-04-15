import { renderHook, act } from '@testing-library/react';
import { useActiveExperimentTrace } from './useActiveExperimentTrace';
import { useSearchParams } from '../../../../common/utils/RoutingUtils';

jest.mock('../../../../common/utils/RoutingUtils', () => ({
  useSearchParams: jest.fn(),
}));

describe('useActiveExperimentTrace', () => {
  let mockSearchParams = new URLSearchParams();
  const mockSetSearchParams = jest.fn((setter) => {
    mockSearchParams = setter(mockSearchParams);
  });

  beforeEach(() => {
    mockSearchParams = new URLSearchParams();
    jest.clearAllMocks();
    jest.mocked(useSearchParams).mockReturnValue([new URLSearchParams(), mockSetSearchParams]);
  });

  test('should return selectedTraceId', () => {
    jest
      .mocked(useSearchParams)
      .mockReturnValue([new URLSearchParams({ selectedTraceId: 'tr-initial' }), mockSetSearchParams]);

    const {
      result: {
        current: [resultTraceId],
      },
    } = renderHook(() => useActiveExperimentTrace());

    expect(resultTraceId).toEqual('tr-initial');
  });

  test('should set selectedTraceId to undefined', () => {
    const {
      result: {
        current: [, setSelectedTraceId],
      },
    } = renderHook(() => useActiveExperimentTrace());

    act(() => {
      setSelectedTraceId(undefined);
    });

    expect(mockSetSearchParams).toHaveBeenCalledTimes(1);
    expect(mockSearchParams.get('selectedTraceId')).toBeNull();
  });

  test('should set selectedTraceId to a value', () => {
    const {
      result: {
        current: [, setSelectedTraceId],
      },
    } = renderHook(() => useActiveExperimentTrace());

    act(() => {
      setSelectedTraceId('tr-12345');
    });

    expect(mockSetSearchParams).toHaveBeenCalledTimes(1);
    expect(mockSearchParams.get('selectedTraceId')).toEqual('tr-12345');
  });
});
