import { renderHook, waitFor } from '@testing-library/react';
import { useSearchParams, useNavigate } from '../../../../common/utils/RoutingUtils';

import { useUpdateExperimentPageSearchFacets } from './useExperimentPageSearchFacets';
import { useSharedExperimentViewState } from './useSharedExperimentViewState';
import { createExperimentPageUIState } from '../models/ExperimentPageUIState';
import type { ExperimentEntity } from '../../../types';
import { createExperimentPageSearchFacetsState } from '../models/ExperimentPageSearchFacetsState';
import { isNil, omitBy } from 'lodash';
import { IntlProvider } from 'react-intl';
import { shouldUseCompressedExperimentViewSharedState } from '../../../../common/utils/FeatureUtils';
import { textCompressDeflate } from '../../../../common/utils/StringUtils';

jest.mock('../../../../common/utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('../../../../common/utils/FeatureUtils')>(
    '../../../../common/utils/FeatureUtils',
  ),
  shouldUseCompressedExperimentViewSharedState: jest.fn(),
}));

jest.mock('../../../../common/utils/RoutingUtils', () => ({
  ...jest.requireActual<typeof import('../../../../common/utils/RoutingUtils')>(
    '../../../../common/utils/RoutingUtils',
  ),
  useSearchParams: jest.fn(),
  useNavigate: jest.fn(),
}));

jest.mock('./useExperimentPageSearchFacets', () => ({
  ...jest.requireActual<typeof import('./useExperimentPageSearchFacets')>('./useExperimentPageSearchFacets'),
  useUpdateExperimentPageSearchFacets: jest.fn(),
}));

const testUIState = { ...createExperimentPageUIState(), selectedColumns: ['metrics.m2'], viewMaximized: true };
const testFacetsState = { ...createExperimentPageSearchFacetsState(), orderByKey: 'metrics.m1', orderByAsc: true };

const testSerializedShareViewState = JSON.stringify({
  ...testUIState,
  ...testFacetsState,
});
const testSerializedStateHash = 'abcdef123456789';

const getTestExperiment = async (isCompressed: boolean) => {
  const tagValue = isCompressed
    ? await textCompressDeflate(testSerializedShareViewState)
    : testSerializedShareViewState;
  return {
    experimentId: 'experiment_1',
    tags: [{ key: `mlflow.sharedViewState.${testSerializedStateHash}`, value: tagValue }],
  } as ExperimentEntity;
};

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

  describe.each([true, false])('when state compression flag is set to %s', (compressionEnabled) => {
    beforeEach(() => {
      jest.mocked(shouldUseCompressedExperimentViewSharedState).mockImplementation(() => compressionEnabled);
    });

    it('should return isViewStateShared as false when viewStateShareKey is not present', async () => {
      const { result } = renderHookWithIntl(() => useSharedExperimentViewState(uiStateSetterMock));

      await waitFor(() => {
        expect(result.current.isViewStateShared).toBe(false);
      });
    });

    it('should return isViewStateShared as true when viewStateShareKey is present', async () => {
      jest
        .mocked(useSearchParams)
        .mockReturnValue([new URLSearchParams(`viewStateShareKey=${testSerializedStateHash}`), jest.fn()]);

      const { result } = renderHookWithIntl(() => useSharedExperimentViewState(uiStateSetterMock));

      await waitFor(() => {
        expect(result.current.isViewStateShared).toBe(true);
      });
    });

    it('should update search facets and ui state when shared state is present', async () => {
      jest
        .mocked(useSearchParams)
        .mockReturnValue([new URLSearchParams(`viewStateShareKey=${testSerializedStateHash}`), jest.fn()]);

      const testExperiment = await getTestExperiment(compressionEnabled);

      const { result } = renderHookWithIntl(() => useSharedExperimentViewState(uiStateSetterMock, testExperiment));

      // Expected state fields, undefined values are omitted
      const expectedFacetsState = omitBy(testFacetsState, isNil);
      const expectedUiState = omitBy(testUIState, isNil);

      await waitFor(() => {
        expect(updateSearchFacetsMock).toHaveBeenCalledWith(expect.objectContaining(expectedFacetsState), {
          replace: true,
        });
        expect(uiStateSetterMock).toHaveBeenCalledWith(expect.objectContaining(expectedUiState));
        expect(result.current.sharedStateError).toBeNull();
      });
    });

    it('should not update state when the hook is disabled', async () => {
      jest
        .mocked(useSearchParams)
        .mockReturnValue([new URLSearchParams(`viewStateShareKey=${testSerializedStateHash}`), jest.fn()]);

      const testExperiment = await getTestExperiment(compressionEnabled);

      const { result } = renderHookWithIntl(() =>
        useSharedExperimentViewState(uiStateSetterMock, testExperiment, true),
      );

      await waitFor(() => {
        expect(updateSearchFacetsMock).not.toHaveBeenCalled();
        expect(uiStateSetterMock).not.toHaveBeenCalled();
        expect(result.current.sharedStateError).toBeNull();
      });
    });

    it('should report an error and navigate to experiment page when shared state is malformed', async () => {
      jest.spyOn(console, 'error').mockImplementation(() => {});
      jest
        .mocked(useSearchParams)
        .mockReturnValue([new URLSearchParams(`viewStateShareKey=${testSerializedStateHash}`), jest.fn()]);

      const testExperiment = await getTestExperiment(compressionEnabled);

      const brokenExperiment = {
        ...testExperiment,
        tags: [{ key: `mlflow.sharedViewState.${testSerializedStateHash}`, value: 'broken' }],
      } as ExperimentEntity;

      const { result } = renderHookWithIntl(() => useSharedExperimentViewState(uiStateSetterMock, brokenExperiment));

      await waitFor(() => {
        expect(updateSearchFacetsMock).not.toHaveBeenCalled();
        expect(uiStateSetterMock).not.toHaveBeenCalled();
        expect(result.current.sharedStateError).toMatch(/Error loading shared view state: share key is invalid/);
        expect(navigateMock).toHaveBeenCalledWith(expect.stringMatching(/\/experiments\/experiment_1$/), {
          replace: true,
        });
        // eslint-disable-next-line no-console -- TODO(FEINF-3587)
        jest.mocked(console.error).mockRestore();
      });
    });
  });

  test('should recognize uncompressed state also when compression flag is enabled', async () => {
    jest.mocked(shouldUseCompressedExperimentViewSharedState).mockImplementation(() => true);

    jest
      .mocked(useSearchParams)
      .mockReturnValue([new URLSearchParams(`viewStateShareKey=${testSerializedStateHash}`), jest.fn()]);

    const testExperiment = await getTestExperiment(false);

    const { result } = renderHookWithIntl(() => useSharedExperimentViewState(uiStateSetterMock, testExperiment));

    // Expected state fields, undefined values are omitted
    const expectedFacetsState = omitBy(testFacetsState, isNil);
    const expectedUiState = omitBy(testUIState, isNil);

    await waitFor(() => {
      expect(updateSearchFacetsMock).toHaveBeenCalledWith(expect.objectContaining(expectedFacetsState), {
        replace: true,
      });
      expect(uiStateSetterMock).toHaveBeenCalledWith(expect.objectContaining(expectedUiState));
      expect(result.current.sharedStateError).toBeNull();
    });
  });
});
