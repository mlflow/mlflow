import { afterEach, describe, expect, it, jest } from '@jest/globals';
import { renderHook } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';

import { useSelectedIssueId } from '@mlflow/mlflow/src/experiment-tracking/components/run-page/hooks/useSelectedIssueId';

import { useSafeSelectedIssueId } from './useSafeSelectedIssueId';

jest.mock('@mlflow/mlflow/src/experiment-tracking/components/run-page/hooks/useSelectedIssueId', () => ({
  ...jest.requireActual<
    typeof import('@mlflow/mlflow/src/experiment-tracking/components/run-page/hooks/useSelectedIssueId')
  >('@mlflow/mlflow/src/experiment-tracking/components/run-page/hooks/useSelectedIssueId'),
  useSelectedIssueId: jest.fn(),
}));

const mockUseSelectedIssueId = jest.mocked(useSelectedIssueId);
const realUseSelectedIssueId = jest.requireActual<
  typeof import('@mlflow/mlflow/src/experiment-tracking/components/run-page/hooks/useSelectedIssueId')
>('@mlflow/mlflow/src/experiment-tracking/components/run-page/hooks/useSelectedIssueId').useSelectedIssueId;

describe('useSafeSelectedIssueId', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('returns the issue selected in the query string', () => {
    mockUseSelectedIssueId.mockImplementation(realUseSelectedIssueId);

    const { result } = renderHook(() => useSafeSelectedIssueId(), {
      wrapper: ({ children }) => (
        <MemoryRouter initialEntries={['/?selectedIssueId=iss-latency']}>{children}</MemoryRouter>
      ),
    });

    expect(result.current).toBe('iss-latency');
  });

  it('returns undefined instead of throwing when there is no router', () => {
    mockUseSelectedIssueId.mockImplementation(realUseSelectedIssueId);

    const { result } = renderHook(() => useSafeSelectedIssueId());

    expect(result.current).toBeUndefined();
  });

  it('rethrows an unrelated failure when a router is present', () => {
    // React logs the error it caught on its way out; the assertion below is the signal.
    jest.spyOn(console, 'error').mockImplementation(() => {});
    mockUseSelectedIssueId.mockImplementation(() => {
      throw new Error('Rendered more hooks than during the previous render');
    });

    expect(() =>
      renderHook(() => useSafeSelectedIssueId(), {
        wrapper: ({ children }) => <MemoryRouter>{children}</MemoryRouter>,
      }),
    ).toThrow('Rendered more hooks than during the previous render');
  });
});
