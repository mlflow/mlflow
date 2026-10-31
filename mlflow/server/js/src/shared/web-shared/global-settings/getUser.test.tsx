import { afterEach, beforeEach, describe, expect, it, jest } from '@jest/globals';
import { renderHook } from '@testing-library/react';

import { GenericNetworkRequestError } from '@databricks/web-shared/errors';
import { useCurrentUserQuery } from '@mlflow/mlflow/src/account/hooks';

import { useUserIdentity } from './getUser';

jest.mock('@mlflow/mlflow/src/account/hooks', () => ({
  useCurrentUserQuery: jest.fn(),
}));

const refetch = jest.fn();

const mockCurrentUserQuery = (state: { data?: unknown; error?: unknown; isFetching?: boolean }) => {
  jest.mocked(useCurrentUserQuery).mockReturnValue({
    data: undefined,
    error: null,
    errorUpdatedAt: state.error ? 1 : 0,
    isFetching: false,
    refetch,
    ...state,
  } as any);
};

describe('useUserIdentity', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it('uses the authenticated username', () => {
    mockCurrentUserQuery({ data: { user: { username: 'alice' } } });

    const { result } = renderHook(() => useUserIdentity());

    expect(result.current).toEqual({ user: 'alice', isResolved: true });
  });

  it('checks the identity again when the window gets focus', () => {
    mockCurrentUserQuery({ data: { user: { username: 'alice' } } });

    renderHook(() => useUserIdentity());

    expect(useCurrentUserQuery).toHaveBeenCalledWith({ refetchOnWindowFocus: true });
  });

  it('uses the OSS placeholder only when the current-user endpoint is unavailable', () => {
    mockCurrentUserQuery({ error: new GenericNetworkRequestError({ status: 404 }) });

    const { result } = renderHook(() => useUserIdentity());

    expect(result.current).toEqual({ user: 'User', isResolved: true });
    jest.advanceTimersByTime(5000);
    expect(refetch).not.toHaveBeenCalled();
  });

  it('does not resolve identity after another request error and retries it', () => {
    mockCurrentUserQuery({ error: new GenericNetworkRequestError({ status: 503 }) });

    const { result } = renderHook(() => useUserIdentity());

    expect(result.current).toEqual({ user: '', isResolved: false });
    jest.advanceTimersByTime(2000);
    expect(refetch).toHaveBeenCalledTimes(1);
  });

  it('does not resolve the cached user while the identity is refetched', () => {
    mockCurrentUserQuery({ data: { user: { username: 'alice' } }, isFetching: true });

    const { result } = renderHook(() => useUserIdentity());

    expect(result.current).toEqual({ user: 'alice', isResolved: false });
  });

  it('does not resolve the cached user after a refetch fails', () => {
    mockCurrentUserQuery({
      data: { user: { username: 'alice' } },
      error: new GenericNetworkRequestError({ status: 503 }),
    });

    const { result } = renderHook(() => useUserIdentity());

    expect(result.current).toEqual({ user: 'alice', isResolved: false });
  });
});
