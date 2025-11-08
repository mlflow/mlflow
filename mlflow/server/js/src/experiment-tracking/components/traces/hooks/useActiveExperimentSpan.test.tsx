import { renderHook, act } from '@testing-library/react';
import { useActiveExperimentSpan } from './useActiveExperimentSpan';
import { useSearchParams } from '../../../../common/utils/RoutingUtils';

jest.mock('../../../../common/utils/RoutingUtils', () => ({
  useSearchParams: jest.fn(),
}));

describe('useActiveExperimentSpan', () => {
  let mockSearchParams = new URLSearchParams();
  const mockSetSearchParams = jest.fn((setter) => {
    mockSearchParams = setter(mockSearchParams);
  });

  beforeEach(() => {
    mockSearchParams = new URLSearchParams();
    jest.clearAllMocks();
    jest.mocked(useSearchParams).mockReturnValue([new URLSearchParams(), mockSetSearchParams]);
  });

  test('should return selectedSpanId', () => {
    jest
      .mocked(useSearchParams)
      .mockReturnValue([new URLSearchParams({ selectedSpanId: 'span-initial' }), mockSetSearchParams]);

    const {
      result: {
        current: [resultSpanId],
      },
    } = renderHook(() => useActiveExperimentSpan());

    expect(resultSpanId).toEqual('span-initial');
  });

  test('should set selectedSpanId to undefined', () => {
    const {
      result: {
        current: [, setSelectedSpanId],
      },
    } = renderHook(() => useActiveExperimentSpan());

    act(() => {
      setSelectedSpanId(undefined);
    });

    expect(mockSetSearchParams).toHaveBeenCalledTimes(1);
    expect(mockSearchParams.get('selectedSpanId')).toBeNull();
  });

  test('should set selectedSpanId to a value', () => {
    const {
      result: {
        current: [, setSelectedSpanId],
      },
    } = renderHook(() => useActiveExperimentSpan());

    act(() => {
      setSelectedSpanId('span-12345');
    });

    expect(mockSetSearchParams).toHaveBeenCalledTimes(1);
    expect(mockSearchParams.get('selectedSpanId')).toEqual('span-12345');
  });
});
