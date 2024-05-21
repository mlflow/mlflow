import { renderHook } from '@testing-library/react-for-react-18';
import { useSearchParams, useNavigate } from '../../../../common/utils/RoutingUtils';

import { useUpdateExperimentPageSearchFacets } from './useExperimentPageSearchFacets';
import { useSharedExperimentViewState } from './useSharedExperimentViewState';
import { createExperimentPageUIState } from '../models/ExperimentPageUIState';
import { ExperimentEntity } from '../../../types';
import { createExperimentPageSearchFacetsState } from '../models/ExperimentPageSearchFacetsState';
import { isNil, omitBy } from 'lodash';
import { IntlProvider } from 'react-intl';

jest.mock('../../../../common/utils/RoutingUtils', () => ({
  ...jest.requireActual('../../../../common/utils/RoutingUtils'),
  useSearchParams: jest.fn(),
  useNavigate: jest.fn(),
}));

jest.mock('./useExperimentPageSearchFacets', () => ({
  ...jest.requireActual('./useExperimentPageSearchFacets'),
  useUpdateExperimentPageSearchFacets: jest.fn(),
}));

const testUIState = { ...createExperimentPageUIState(), selectedColumns: ['metrics.m2'], viewMaximized: true };
const testFacetsState = { ...createExperimentPageSearchFacetsState(), orderByKey: 'metrics.m1', orderByAsc: true };

const testSerializedShareViewState = JSON.stringify({
  ...testUIState,
  ...testFacetsState,
});
const testSerializedStateHash = 'abcdef123456789';

const testExperiment = {
  experimentId: 'experiment_1',
  tags: [{ key: `mlflow.sharedViewState.${testSerializedStateHash}`, value: testSerializedShareViewState }],
} as ExperimentEntity;

describe('useSharedExperimentViewState', () => {
  const uiStateSetterMock = jest.fn();
  const updateSearchFacetsMock = jest.fn();
  const navigateMock = jest.fn();

  const renderHookWithIntl = (hook: () => ReturnType<typeof useSharedExperimentViewState>) => {
    return renderHook(hook, { wrapper: ({ children }) => <IntlProvider locale="en">{children}</IntlProvider> });
  };

  beforeEach(() => {
    jest.clearAllMocks();
    jest.mocked(useSearchParams).mockReturnValue([new URLSearchParams(), jest.fn()]);
    jest.mocked(useNavigate).mockReturnValue(navigateMock);
    jest.mocked(useUpdateExperimentPageSearchFacets).mockReturnValue(updateSearchFacetsMock);
  });

  it('should return isViewStateShared as false when viewStateShareKey is not present', () => {
    const { result } = renderHookWithIntl(() => useSharedExperimentViewState(uiStateSetterMock));

    expect(result.current.isViewStateShared).toBe(false);
  });

  it('should return isViewStateShared as true when viewStateShareKey is present', () => {
    jest
      .mocked(useSearchParams)
      .mockReturnValue([new URLSearchParams(`viewStateShareKey=${testSerializedStateHash}`), jest.fn()]);

    const { result } = renderHookWithIntl(() => useSharedExperimentViewState(uiStateSetterMock));

    expect(result.current.isViewStateShared).toBe(true);
  });

  it('should update search facets and ui state when shared state is present', () => {
    jest
      .mocked(useSearchParams)
      .mockReturnValue([new URLSearchParams(`viewStateShareKey=${testSerializedStateHash}`), jest.fn()]);

    const { result } = renderHookWithIntl(() => useSharedExperimentViewState(uiStateSetterMock, testExperiment));

    // Expected state fields, undefined values are omitted
    const expectedFacetsState = omitBy(testFacetsState, isNil);
    const expectedUiState = omitBy(testUIState, isNil);

    expect(updateSearchFacetsMock).toHaveBeenCalledWith(expect.objectContaining(expectedFacetsState), {
      replace: true,
    });
    expect(uiStateSetterMock).toHaveBeenCalledWith(expect.objectContaining(expectedUiState));
    expect(result.current.sharedStateError).toBeNull();
  });

  it('should not update state when the hook is disabled', () => {
    jest
      .mocked(useSearchParams)
      .mockReturnValue([new URLSearchParams(`viewStateShareKey=${testSerializedStateHash}`), jest.fn()]);

    const { result } = renderHookWithIntl(() => useSharedExperimentViewState(uiStateSetterMock, testExperiment, true));

    expect(updateSearchFacetsMock).not.toHaveBeenCalled();
    expect(uiStateSetterMock).not.toHaveBeenCalled();
    expect(result.current.sharedStateError).toBeNull();
  });

  it('should report an error and navigate to experiment page when shared state is malformed', () => {
    jest.spyOn(console, 'error').mockImplementation(() => {});
    jest
      .mocked(useSearchParams)
      .mockReturnValue([new URLSearchParams(`viewStateShareKey=${testSerializedStateHash}`), jest.fn()]);
    const brokenExperiment = {
      ...testExperiment,
      tags: [{ key: `mlflow.sharedViewState.${testSerializedStateHash}`, value: 'broken' }],
    } as ExperimentEntity;

    const { result } = renderHookWithIntl(() => useSharedExperimentViewState(uiStateSetterMock, brokenExperiment));

    expect(updateSearchFacetsMock).not.toHaveBeenCalled();
    expect(uiStateSetterMock).not.toHaveBeenCalled();
    expect(result.current.sharedStateError).toMatch(/Error loading shared view state: share key is invalid/);
    expect(navigateMock).toHaveBeenCalledWith(expect.stringMatching(/\/experiments\/experiment_1$/), { replace: true });
    jest.mocked(console.error).mockRestore();
  });
});
