import { renderHook, act } from '@testing-library/react';

import { useActiveEvaluation } from './useActiveEvaluation';
import { useSearchParams } from '../utils/RoutingUtils';

jest.mock('../utils/RoutingUtils', () => ({
  useSearchParams: jest.fn(),
}));

describe('useActiveEvaluation', () => {
  let mockSearchParams = new URLSearchParams();
  const mockSetSearchParams = jest.fn((setter) => {
    mockSearchParams = setter(mockSearchParams);
  });

  beforeEach(() => {
    mockSearchParams = new URLSearchParams();
    jest.clearAllMocks();
    jest.mocked(useSearchParams).mockReturnValue([new URLSearchParams(), mockSetSearchParams]);
  });

  test('should return selectedEvaluationId', () => {
    jest
      .mocked(useSearchParams)
      .mockReturnValue([new URLSearchParams({ selectedEvaluationId: 'some-eval-id' }), mockSetSearchParams]);

    const {
      result: {
        current: [resultEvaluationId],
      },
    } = renderHook(() => useActiveEvaluation());

    expect(resultEvaluationId).toEqual('some-eval-id');
  });

  test('should set selectedEvaluationId to undefined', () => {
    const {
      result: {
        current: [, setSelectedEvaluationId],
      },
    } = renderHook(() => useActiveEvaluation());

    act(() => {
      setSelectedEvaluationId(undefined);
    });

    expect(mockSetSearchParams).toHaveBeenCalledTimes(1);
    expect(mockSearchParams.get('selectedEvaluationId')).toBeNull();
  });

  test('should set selectedEvaluationId to a value', () => {
    const {
      result: {
        current: [, setSelectedEvaluationId],
      },
    } = renderHook(() => useActiveEvaluation());

    act(() => {
      setSelectedEvaluationId('another-eval-id');
    });

    expect(mockSetSearchParams).toHaveBeenCalledTimes(1);
    expect(mockSearchParams.get('selectedEvaluationId')).toEqual('another-eval-id');
  });
});
