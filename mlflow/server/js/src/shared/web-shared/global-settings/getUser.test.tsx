import { beforeEach, describe, expect, it, jest } from '@jest/globals';
import { renderHook } from '@testing-library/react';

import { GenericNetworkRequestError } from '@databricks/web-shared/errors';
import { useCurrentUserQuery } from '@mlflow/mlflow/src/account/hooks';

import { useUserIdentity } from './getUser';

jest.mock('@mlflow/mlflow/src/account/hooks', () => ({
  useCurrentUserQuery: jest.fn(),
}));

describe('useUserIdentity', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('uses the authenticated username', () => {
    jest.mocked(useCurrentUserQuery).mockReturnValue({
      data: { user: { username: 'alice' } },
      error: undefined,
    } as any);

    const { result } = renderHook(() => useUserIdentity());

    expect(result.current).toEqual({ user: 'alice', isResolved: true });
  });

  it('uses the OSS placeholder only when the current-user endpoint is unavailable', () => {
    jest.mocked(useCurrentUserQuery).mockReturnValue({
      data: undefined,
      error: new GenericNetworkRequestError({ status: 404 }),
    } as any);

    const { result } = renderHook(() => useUserIdentity());

    expect(result.current).toEqual({ user: 'User', isResolved: true });
  });

  it('does not resolve identity after another request error', () => {
    jest.mocked(useCurrentUserQuery).mockReturnValue({
      data: undefined,
      error: new GenericNetworkRequestError({ status: 503 }),
    } as any);

    const { result } = renderHook(() => useUserIdentity());

    expect(result.current).toEqual({ user: '', isResolved: false });
  });
});
