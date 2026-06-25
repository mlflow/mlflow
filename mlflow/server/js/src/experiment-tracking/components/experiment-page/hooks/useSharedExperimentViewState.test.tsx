import { jest, describe, beforeEach, it, expect, test } from '@jest/globals';
import { act, renderHook, waitFor } from '@testing-library/react';
import type { NavigateFunction } from '../../../../common/utils/RoutingUtils';
import { useSearchParams, useNavigate } from '../../../../common/utils/RoutingUtils';

import { useUpdateExperimentPageSearchFacets } from './useExperimentPageSearchFacets';
import { useSharedExperimentViewState } from './useSharedExperimentViewState';
import { createExperimentPageUIState } from '../models/ExperimentPageUIState';
import type { ExperimentEntity } from '../../../types';
import { createExperimentPageSearchFacetsState } from '../models/ExperimentPageSearchFacetsState';
import { isNil, omitBy } from 'lodash';
import { IntlProvider } from 'react-intl';
import { shouldUseCompressedExperimentViewSharedState } from '../../../../common/utils/FeatureUtils';
import { textCompressDeflate, textDecompressDeflate } from '../../../../common/utils/StringUtils';

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
    jest.mocked(useNavigate).mockReturnValue(navigateMock as ReturnType<typeof useNavigate>);
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

    it('does not apply or clear view state when the legacy share key has no matching tag', async () => {
      jest.spyOn(console, 'error').mockImplementation(() => {});
      jest
        .mocked(useSearchParams)
        .mockReturnValue([new URLSearchParams(`viewStateShareKey=${testSerializedStateHash}`), jest.fn()]);

      // Bare-hash key routes to the legacy branch; the experiment has no matching tag.
      const experimentWithoutTag = { experimentId: 'experiment_1', tags: [] } as unknown as ExperimentEntity;

      const { result } = renderHookWithIntl(() =>
        useSharedExperimentViewState(uiStateSetterMock, experimentWithoutTag),
      );

      await waitFor(() => {
        expect(updateSearchFacetsMock).not.toHaveBeenCalled();
        expect(uiStateSetterMock).not.toHaveBeenCalled();
        expect(result.current.sharedStateError).toMatch(/does not exist/);
      });
      // eslint-disable-next-line no-console -- TODO(FEINF-3587)
      jest.mocked(console.error).mockRestore();
    });
  });

  describe('url-embedded shared links (view state carried in the viewStateShareKey param)', () => {
    it.each([true, false])(
      'applies search facets and ui state from a %s-compressed URL blob without a tag lookup',
      async (isCompressed) => {
        jest.mocked(shouldUseCompressedExperimentViewSharedState).mockImplementation(() => true);

        const blob = isCompressed
          ? await textCompressDeflate(testSerializedShareViewState)
          : testSerializedShareViewState;
        jest.mocked(useSearchParams).mockReturnValue([new URLSearchParams({ viewStateShareKey: blob }), jest.fn()]);

        // No experiment is passed: a url-embedded link must not need a tag lookup
        const { result } = renderHookWithIntl(() => useSharedExperimentViewState(uiStateSetterMock));

        const expectedFacetsState = omitBy(testFacetsState, isNil);
        const expectedUiState = omitBy(testUIState, isNil);

        await waitFor(() => {
          expect(result.current.isViewStateShared).toBe(true);
          expect(updateSearchFacetsMock).toHaveBeenCalledWith(expect.objectContaining(expectedFacetsState), {
            replace: true,
          });
          expect(uiStateSetterMock).toHaveBeenCalledWith(expect.objectContaining(expectedUiState));
          expect(result.current.sharedStateError).toBeNull();
        });
      },
    );

    it('applies the embedded blob only once when the experiment reference changes', async () => {
      jest.mocked(shouldUseCompressedExperimentViewSharedState).mockImplementation(() => true);

      const blob = await textCompressDeflate(testSerializedShareViewState);
      jest.mocked(useSearchParams).mockReturnValue([new URLSearchParams({ viewStateShareKey: blob }), jest.fn()]);

      // A url-embedded link doesn't need an experiment, but `experiment` (passed as
      // first(experiments)) loads async and later mutates on tag edits. The hook must
      // not re-apply the URL blob over the user's edits each time that reference changes.
      const { rerender } = renderHook(
        ({ experiment }: { experiment?: ExperimentEntity }) =>
          useSharedExperimentViewState(uiStateSetterMock, experiment),
        {
          initialProps: { experiment: undefined as ExperimentEntity | undefined },
          wrapper: ({ children }) => <IntlProvider locale="en">{children}</IntlProvider>,
        },
      );

      await waitFor(() => {
        expect(updateSearchFacetsMock).toHaveBeenCalledTimes(1);
        expect(uiStateSetterMock).toHaveBeenCalledTimes(1);
      });

      // experiment resolves (undefined -> defined), then a tag edit refetches it (new ref)
      rerender({ experiment: { experimentId: 'experiment_1', tags: [] } as unknown as ExperimentEntity });
      rerender({ experiment: { experimentId: 'experiment_1', tags: [] } as unknown as ExperimentEntity });

      // Let any (buggy) re-application settle: the apply path awaits the same decompress,
      // and act() flushes the resulting state updates + downstream effects. Without the
      // guard this is where updateSearchFacets/uiStateSetter would fire a second time.
      await act(async () => {
        await textDecompressDeflate(blob);
        await Promise.resolve();
      });

      expect(updateSearchFacetsMock).toHaveBeenCalledTimes(1);
      expect(uiStateSetterMock).toHaveBeenCalledTimes(1);
    });

    it('reports an error when the embedded blob is malformed', async () => {
      jest.spyOn(console, 'error').mockImplementation(() => {});
      const malformedBlob = `${'deflate;'}not-valid-base64-deflate`;
      jest
        .mocked(useSearchParams)
        .mockReturnValue([new URLSearchParams({ viewStateShareKey: malformedBlob }), jest.fn()]);

      const { result } = renderHookWithIntl(() => useSharedExperimentViewState(uiStateSetterMock));

      await waitFor(() => {
        expect(updateSearchFacetsMock).not.toHaveBeenCalled();
        expect(uiStateSetterMock).not.toHaveBeenCalled();
        expect(result.current.sharedStateError).toMatch(/Error loading shared view state: share key is invalid/);
      });
      // eslint-disable-next-line no-console -- TODO(FEINF-3587)
      jest.mocked(console.error).mockRestore();
    });

    it('does not apply or clear view state when the embedded blob is valid JSON but not an object', async () => {
      jest.spyOn(console, 'error').mockImplementation(() => {});
      jest.mocked(shouldUseCompressedExperimentViewSharedState).mockImplementation(() => true);

      // Deflate-wrapped so it is detected as url-embedded, but decodes to a non-object (42),
      // which must be rejected without touching the user's existing view.
      const blob = await textCompressDeflate('42');
      jest.mocked(useSearchParams).mockReturnValue([new URLSearchParams({ viewStateShareKey: blob }), jest.fn()]);

      const { result } = renderHookWithIntl(() => useSharedExperimentViewState(uiStateSetterMock));

      await waitFor(() => {
        expect(updateSearchFacetsMock).not.toHaveBeenCalled();
        expect(uiStateSetterMock).not.toHaveBeenCalled();
        expect(result.current.sharedStateError).toMatch(/Error loading shared view state: share key is invalid/);
      });
      // eslint-disable-next-line no-console -- TODO(FEINF-3587)
      jest.mocked(console.error).mockRestore();
    });

    it('does not apply or clear view state when the embedded blob is a JSON array', async () => {
      jest.spyOn(console, 'error').mockImplementation(() => {});
      jest.mocked(shouldUseCompressedExperimentViewSharedState).mockImplementation(() => true);

      // Arrays are typeof 'object'; deflate-wrapped so it is detected as url-embedded. It must
      // be rejected (not pick()ed into {}), otherwise the recipient's view silently resets to defaults.
      const blob = await textCompressDeflate('[1,2,3]');
      jest.mocked(useSearchParams).mockReturnValue([new URLSearchParams({ viewStateShareKey: blob }), jest.fn()]);

      const { result } = renderHookWithIntl(() => useSharedExperimentViewState(uiStateSetterMock));

      await waitFor(() => {
        expect(updateSearchFacetsMock).not.toHaveBeenCalled();
        expect(uiStateSetterMock).not.toHaveBeenCalled();
        expect(result.current.sharedStateError).toMatch(/Error loading shared view state: share key is invalid/);
      });
      // eslint-disable-next-line no-console -- TODO(FEINF-3587)
      jest.mocked(console.error).mockRestore();
    });

    it('drops smuggled non-shareable per-run fields and resets them to defaults', async () => {
      jest.mocked(shouldUseCompressedExperimentViewSharedState).mockImplementation(() => true);

      // A hand-crafted blob carrying per-run state keyed by run UUIDs that don't exist for the
      // recipient, plus a legitimate shareable field. The reader must apply the legit field while
      // dropping the smuggled ones back to defaults (write side already omits them).
      const blob = JSON.stringify({
        viewMaximized: true,
        runsHidden: ['someone-elses-run'],
        runsPinned: ['another-run'],
        runsVisibilityMap: { 'a-run': true },
        runsExpanded: { 'a-run': true },
        autoRefreshEnabled: false,
      });
      jest.mocked(useSearchParams).mockReturnValue([new URLSearchParams({ viewStateShareKey: blob }), jest.fn()]);

      const defaults = createExperimentPageUIState();
      const { result } = renderHookWithIntl(() => useSharedExperimentViewState(uiStateSetterMock));

      await waitFor(() => {
        expect(uiStateSetterMock).toHaveBeenCalledWith(
          expect.objectContaining({
            viewMaximized: true,
            runsHidden: defaults.runsHidden,
            runsPinned: defaults.runsPinned,
            runsVisibilityMap: defaults.runsVisibilityMap,
            runsExpanded: defaults.runsExpanded,
            autoRefreshEnabled: defaults.autoRefreshEnabled,
          }),
        );
        expect(result.current.sharedStateError).toBeNull();
      });
    });

    it('marks the shared view active after a successful apply and clears it on exitSharedView', async () => {
      jest.mocked(shouldUseCompressedExperimentViewSharedState).mockImplementation(() => true);
      const blob = await textCompressDeflate(testSerializedShareViewState);
      jest.mocked(useSearchParams).mockReturnValue([new URLSearchParams({ viewStateShareKey: blob }), jest.fn()]);

      const { result } = renderHookWithIntl(() => useSharedExperimentViewState(uiStateSetterMock));

      await waitFor(() => expect(result.current.sharedViewActive).toBe(true));

      act(() => result.current.exitSharedView());
      expect(result.current.sharedViewActive).toBe(false);
    });

    it('keeps the shared view active after the share key leaves the URL (sticky read-only)', async () => {
      jest.mocked(shouldUseCompressedExperimentViewSharedState).mockImplementation(() => true);
      const blob = await textCompressDeflate(testSerializedShareViewState);
      jest.mocked(useSearchParams).mockReturnValue([new URLSearchParams({ viewStateShareKey: blob }), jest.fn()]);

      const { result, rerender } = renderHookWithIntl(() => useSharedExperimentViewState(uiStateSetterMock));
      await waitFor(() => expect(result.current.sharedViewActive).toBe(true));

      // Simulate navigating to a keyless route (e.g. the Runs tab): the key is gone, but the session
      // must stay read-only so the shared view isn't auto-persisted over the user's own saved view.
      jest.mocked(useSearchParams).mockReturnValue([new URLSearchParams(), jest.fn()]);
      rerender();

      expect(result.current.isViewStateShared).toBe(false);
      expect(result.current.sharedViewActive).toBe(true);
    });

    it('clears the shared view when navigating to a different experiment', async () => {
      jest.mocked(shouldUseCompressedExperimentViewSharedState).mockImplementation(() => true);
      const blob = await textCompressDeflate(testSerializedShareViewState);
      jest.mocked(useSearchParams).mockReturnValue([new URLSearchParams({ viewStateShareKey: blob }), jest.fn()]);

      const { result, rerender } = renderHook(
        ({ experiment }: { experiment?: ExperimentEntity }) =>
          useSharedExperimentViewState(uiStateSetterMock, experiment),
        {
          initialProps: { experiment: { experimentId: 'experiment_1', tags: [] } as unknown as ExperimentEntity },
          wrapper: ({ children }) => <IntlProvider locale="en">{children}</IntlProvider>,
        },
      );

      await waitFor(() => expect(result.current.sharedViewActive).toBe(true));

      // The view isn't remounted on an experiment switch, so the latch must reset itself — otherwise
      // it would keep local-storage persistence disabled for the next experiment too.
      rerender({ experiment: { experimentId: 'experiment_2', tags: [] } as unknown as ExperimentEntity });

      await waitFor(() => expect(result.current.sharedViewActive).toBe(false));
    });

    it('does not clear the shared view when the experiment resolves from undefined for the same link', async () => {
      jest.mocked(shouldUseCompressedExperimentViewSharedState).mockImplementation(() => true);
      const blob = await textCompressDeflate(testSerializedShareViewState);
      jest.mocked(useSearchParams).mockReturnValue([new URLSearchParams({ viewStateShareKey: blob }), jest.fn()]);

      // A url-embedded link latches the view active before `experiment` has loaded; the subsequent
      // undefined -> resolved transition must NOT be mistaken for an experiment switch.
      const { result, rerender } = renderHook(
        ({ experiment }: { experiment?: ExperimentEntity }) =>
          useSharedExperimentViewState(uiStateSetterMock, experiment),
        {
          initialProps: { experiment: undefined as ExperimentEntity | undefined },
          wrapper: ({ children }) => <IntlProvider locale="en">{children}</IntlProvider>,
        },
      );

      await waitFor(() => expect(result.current.sharedViewActive).toBe(true));

      rerender({ experiment: { experimentId: 'experiment_1', tags: [] } as unknown as ExperimentEntity });

      expect(result.current.sharedViewActive).toBe(true);
    });

    it('does not activate the shared view when the embedded blob is invalid', async () => {
      jest.spyOn(console, 'error').mockImplementation(() => {});
      jest.mocked(shouldUseCompressedExperimentViewSharedState).mockImplementation(() => true);
      const malformedBlob = `${'deflate;'}not-valid-base64-deflate`;
      jest
        .mocked(useSearchParams)
        .mockReturnValue([new URLSearchParams({ viewStateShareKey: malformedBlob }), jest.fn()]);

      const { result } = renderHookWithIntl(() => useSharedExperimentViewState(uiStateSetterMock));

      await waitFor(() => expect(result.current.sharedStateError).toMatch(/share key is invalid/));
      expect(result.current.sharedViewActive).toBe(false);
      // eslint-disable-next-line no-console -- TODO(FEINF-3587)
      jest.mocked(console.error).mockRestore();
    });

    it('re-applies a shared link after navigating away and back instead of hanging', async () => {
      jest.mocked(shouldUseCompressedExperimentViewSharedState).mockImplementation(() => true);
      const blob = await textCompressDeflate(testSerializedShareViewState);
      jest.mocked(useSearchParams).mockReturnValue([new URLSearchParams({ viewStateShareKey: blob }), jest.fn()]);

      const { rerender } = renderHookWithIntl(() => useSharedExperimentViewState(uiStateSetterMock));
      await waitFor(() => expect(updateSearchFacetsMock).toHaveBeenCalledTimes(1));

      // Navigate to a keyless route (client-side, no remount): the apply-once guard must reset.
      jest.mocked(useSearchParams).mockReturnValue([new URLSearchParams(), jest.fn()]);
      rerender();

      // Navigate back to the same shared link: it must re-apply rather than hang on a stale guard
      // (otherwise searchFacets never repopulate and the view stays on the loading skeleton).
      jest.mocked(useSearchParams).mockReturnValue([new URLSearchParams({ viewStateShareKey: blob }), jest.fn()]);
      rerender();
      await waitFor(() => expect(updateSearchFacetsMock).toHaveBeenCalledTimes(2));
    });

    it('re-applies when the facet params are wiped while the share key stays (re-pasting the bare link)', async () => {
      jest.mocked(shouldUseCompressedExperimentViewSharedState).mockImplementation(() => true);
      const blob = await textCompressDeflate(testSerializedShareViewState);

      // Initial arrival: key present, no facet params yet → applies (which writes the facets).
      jest.mocked(useSearchParams).mockReturnValue([new URLSearchParams({ viewStateShareKey: blob }), jest.fn()]);
      const { rerender } = renderHookWithIntl(() => useSharedExperimentViewState(uiStateSetterMock));
      await waitFor(() => expect(updateSearchFacetsMock).toHaveBeenCalledTimes(1));

      // Facets now present in the URL (apply landed) → must NOT re-apply (no stomp, no re-fire spam).
      jest
        .mocked(useSearchParams)
        .mockReturnValue([new URLSearchParams({ viewStateShareKey: blob, orderByKey: 'metrics.m1' }), jest.fn()]);
      rerender();
      expect(updateSearchFacetsMock).toHaveBeenCalledTimes(1);

      // Re-paste the bare link: facet params wiped, key unchanged → must re-apply, not hang on the
      // skeleton (the bug). The previous guard skipped this because the key hadn't changed.
      jest.mocked(useSearchParams).mockReturnValue([new URLSearchParams({ viewStateShareKey: blob }), jest.fn()]);
      rerender();
      await waitFor(() => expect(updateSearchFacetsMock).toHaveBeenCalledTimes(2));
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
