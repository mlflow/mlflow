import { describe, jest, it, expect, beforeEach } from '@jest/globals';

import { AdminApi } from './api';
import { fetchEndpoint } from '../common/utils/FetchUtils';

jest.mock('../common/utils/FetchUtils', () => ({
  fetchEndpoint: jest.fn(() => Promise.resolve({ roles: [] })),
}));

const mockedFetch = jest.mocked(fetchEndpoint);

const lastUrl = (): string => {
  const calls = mockedFetch.mock.calls;
  expect(calls.length).toBeGreaterThan(0);
  return (calls[calls.length - 1][0] as { relativeUrl: string }).relativeUrl;
};

beforeEach(() => {
  mockedFetch.mockClear();
});

describe('AdminApi.listRoles URL shape', () => {
  it('omits the workspace param entirely when called with no argument', () => {
    AdminApi.listRoles();
    // Bare path — no trailing ``?`` even when there are zero params, since the
    // earlier ``join('?')`` form polluted logs and broke some signature checks.
    expect(lastUrl()).toBe('ajax-api/3.0/mlflow/roles/list');
  });

  it('appends a single workspace when called with a string', () => {
    AdminApi.listRoles('ml-research');
    expect(lastUrl()).toBe('ajax-api/3.0/mlflow/roles/list?workspace=ml-research');
  });

  it('appends one workspace param per element when called with an array', () => {
    AdminApi.listRoles(['foo', 'bar']);
    // Repeated ``workspace=`` query params (one per element) is the wire shape
    // the server reads via ``request.args.getlist("workspace")``.
    expect(lastUrl()).toBe('ajax-api/3.0/mlflow/roles/list?workspace=foo&workspace=bar');
  });

  it('returns the bare URL (no trailing ``?``) when called with an empty array', () => {
    AdminApi.listRoles([]);
    // The validator denies empty unscoped lists for non-admins; the handler
    // treats no params as the unscoped admin path. Either way the URL is bare.
    expect(lastUrl()).toBe('ajax-api/3.0/mlflow/roles/list');
  });

  it('skips falsy entries within an array (defensive)', () => {
    AdminApi.listRoles(['foo', '']);
    expect(lastUrl()).toBe('ajax-api/3.0/mlflow/roles/list?workspace=foo');
  });

  it('URL-encodes workspace names containing reserved characters', () => {
    AdminApi.listRoles(['ws&with=reserved']);
    expect(lastUrl()).toBe('ajax-api/3.0/mlflow/roles/list?workspace=ws%26with%3Dreserved');
  });

  it('preserves caller-provided order in the URL (the cache key normalizes; the wire does not need to)', () => {
    AdminApi.listRoles(['b', 'a']);
    expect(lastUrl()).toBe('ajax-api/3.0/mlflow/roles/list?workspace=b&workspace=a');
  });
});
