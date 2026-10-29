import { describe, expect, jest, test } from '@jest/globals';
import { shouldEnableNewEvalRunsUi } from '@mlflow/mlflow/src/common/utils/FeatureUtils';
import { buildOrderBy } from './buildTracesV4SearchParams';

jest.mock('@mlflow/mlflow/src/common/utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('@mlflow/mlflow/src/common/utils/FeatureUtils')>(
    '@mlflow/mlflow/src/common/utils/FeatureUtils',
  ),
  shouldEnableNewEvalRunsUi: jest.fn(() => false),
}));

describe('buildOrderBy for state', () => {
  test('with the new eval runs UI, orders by status', () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
    expect(buildOrderBy('state', 'asc')).toEqual(['status ASC']);
    expect(buildOrderBy('state', 'desc')).toEqual(['status DESC']);
    // Tokens and cost have no server-side ordering, so they stay unsorted.
    expect(buildOrderBy('tokens', 'desc')).toBeUndefined();
  });

  test('without it, state stays unsorted', () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(false);
    expect(buildOrderBy('state', 'desc')).toBeUndefined();
  });
});
