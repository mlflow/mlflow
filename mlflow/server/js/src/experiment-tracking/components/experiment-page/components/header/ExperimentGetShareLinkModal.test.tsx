import type { ExperimentPageSearchFacetsState } from '../../models/ExperimentPageSearchFacetsState';
import { createExperimentPageSearchFacetsState } from '../../models/ExperimentPageSearchFacetsState';
import type { ExperimentPageUIState } from '../../models/ExperimentPageUIState';
import { createExperimentPageUIState } from '../../models/ExperimentPageUIState';
import { ExperimentGetShareLinkModal } from './ExperimentGetShareLinkModal';
import { MockedReduxStoreProvider } from '../../../../../common/utils/TestUtils';
import { render, screen, waitFor } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';
import { setExperimentTagApi } from '../../../../actions';
import { shouldUseCompressedExperimentViewSharedState } from '../../../../../common/utils/FeatureUtils';
import { textDecompressDeflate } from '../../../../../common/utils/StringUtils';
import { IntlProvider } from 'react-intl';
import { setupTestRouter, testRoute, TestRouter } from '../../../../../common/utils/RoutingTestUtils';

jest.mock('../../../../../common/utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('../../../../../common/utils/FeatureUtils')>(
    '../../../../../common/utils/FeatureUtils',
  ),
  shouldUseCompressedExperimentViewSharedState: jest.fn(),
}));

jest.mock('../../../../../common/utils/StringUtils', () => {
  const windowCryptoSupported = Boolean(global.crypto?.subtle);
  // If window.crypto is not supported, provide a simple hex hashing function instead of SHA256
  if (!windowCryptoSupported) {
    return {
      ...jest.requireActual<typeof import('../../../../../common/utils/StringUtils')>(
        '../../../../../common/utils/StringUtils',
      ),
      getStringSHA256: (val: string) =>
        val.split('').reduce((hex, c) => hex + c.charCodeAt(0).toString(16).padStart(2, '0'), ''),
    };
  }
  return jest.requireActual<typeof import('../../../../../common/utils/StringUtils')>(
    '../../../../../common/utils/StringUtils',
  );
});

jest.mock('../../../../actions', () => ({
  ...jest.requireActual<typeof import('../../../../actions')>('../../../../actions'),
  setExperimentTagApi: jest.fn(() => ({ type: 'SET_EXPERIMENT_TAG_API', payload: Promise.resolve() })),
}));

const experimentIds = ['experiment-1'];

describe('ExperimentGetShareLinkModal', () => {
  const onCancel = jest.fn();
  const { history } = setupTestRouter();

  let navigatorClipboard: Clipboard;

  beforeAll(() => {
    navigatorClipboard = navigator.clipboard;
    // @ts-expect-error: navigator is overridable in tests
    navigator.clipboard = { writeText: jest.fn() };
  });

  afterAll(() => {
    jest.restoreAllMocks();
    // @ts-expect-error: navigator is overridable in tests
    navigator.clipboard = navigatorClipboard;
  });

  const renderExperimentGetShareLinkModal = (
    searchFacetsState = createExperimentPageSearchFacetsState(),
    uiState = createExperimentPageUIState(),
    initialUrl = '/',
  ) => {
    const Component = ({
      searchFacetsState,
      uiState,
    }: {
      searchFacetsState: ExperimentPageSearchFacetsState;
      uiState: ExperimentPageUIState;
    }) => {
      const [visible, setVisible] = useState(false);
      return (
        <IntlProvider locale="en">
          <MockedReduxStoreProvider>
            <button onClick={() => setVisible(true)}>get link</button>
            <ExperimentGetShareLinkModal
              experimentIds={experimentIds}
              onCancel={onCancel}
              searchFacetsState={searchFacetsState}
              uiState={uiState}
              visible={visible}
            />
          </MockedReduxStoreProvider>
        </IntlProvider>
      );
    };
    const { rerender } = render(<Component searchFacetsState={searchFacetsState} uiState={uiState} />, {
      wrapper: ({ children }) => (
        <TestRouter routes={[testRoute(<>{children}</>, '/')]} history={history} initialEntries={[initialUrl]} />
      ),
    });
    return {
      rerender: (
        searchFacetsState = createExperimentPageSearchFacetsState(),
        uiState = createExperimentPageUIState(),
      ) => rerender(<Component searchFacetsState={searchFacetsState} uiState={uiState} />),
    };
  };

  test.each([true, false])(
    'copies the shareable URL and expects to reuse the same tag for identical view state when compression enabled: %s',
    async (isCompressionEnabled) => {
      jest.mocked(shouldUseCompressedExperimentViewSharedState).mockImplementation(() => isCompressionEnabled);

      // Initial facets and UI state
      const initialSearchState = { ...createExperimentPageSearchFacetsState(), searchFilter: 'metrics.m1 = 2' };
      const initialUIState = { ...createExperimentPageUIState(), viewMaximized: true };

      // Render the modal and open it
      renderExperimentGetShareLinkModal(initialSearchState, initialUIState);
      await waitFor(() => expect(screen.getByText('get link')).toBeInTheDocument());
      await userEvent.click(screen.getByText('get link'));

      // Wait for the link and tag to be processed and copy button to be visible
      await waitFor(() => {
        expect(screen.getByTestId('share-link-copy-button')).toBeInTheDocument();
      });

      // Click the copy button and assert that the URL was copied to the clipboard
      await userEvent.click(screen.getByTestId('share-link-copy-button'));
      expect(navigator.clipboard.writeText).toHaveBeenCalledWith(
        expect.stringMatching(/\/experiments\/experiment-1\?viewStateShareKey=([0-9a-f]+)/i),
      );

      // Assert that the tag was created with the correct name and serialized state
      expect(setExperimentTagApi).toHaveBeenCalledWith(
        'experiment-1',
        expect.stringMatching(/mlflow\.sharedViewState\.([0-9a-f]+)/),
        // Assert serialized state in the next step
        expect.anything(),
      );
      const serializedTagValue = jest.mocked(setExperimentTagApi).mock.lastCall?.[2];

      const serializedState = isCompressionEnabled
        ? JSON.parse(await textDecompressDeflate(serializedTagValue))
        : JSON.parse(serializedTagValue);

      expect(serializedState).toEqual({
        ...initialSearchState,
        ...initialUIState,
      });
    },
  );

  test('propagate experiment page view mode', async () => {
    const initialSearchState = { ...createExperimentPageSearchFacetsState(), searchFilter: 'metrics.m1 = 2' };
    const initialUIState = { ...createExperimentPageUIState(), viewMaximized: true };

    renderExperimentGetShareLinkModal(initialSearchState, initialUIState, '/?compareRunsMode=CHART');
    await waitFor(() => expect(screen.getByText('get link')).toBeInTheDocument());

    await userEvent.click(screen.getByText('get link'));

    // Expect shareable URL to contain the view mode query param
    await waitFor(() => expect(screen.getByRole<HTMLInputElement>('textbox').value).toContain('compareRunsMode=CHART'));
  });

  test('reuse the same tag for identical view state', async () => {
    // Initial facets and UI state
    const initialSearchState = { ...createExperimentPageSearchFacetsState(), searchFilter: 'metrics.m1 = 2' };
    const initialUIState = { ...createExperimentPageUIState(), viewMaximized: true };

    // Render the modal and open it
    const { rerender } = renderExperimentGetShareLinkModal(initialSearchState, initialUIState);

    await waitFor(() => expect(screen.getByText('get link')).toBeInTheDocument());

    await userEvent.click(screen.getByText('get link'));

    // Wait for the link and tag to be processed and copy button to be visible
    await waitFor(() => {
      expect(screen.getByTestId('share-link-copy-button')).toBeInTheDocument();
    });
    await userEvent.click(screen.getByTestId('share-link-copy-button'));

    // Save the first persisted tag name (containing serialized state hash)
    const firstSavedTagName = jest.mocked(setExperimentTagApi).mock.lastCall?.[1];

    // Update the search state and UI state, rerender the modal
    const updatedSearchState = { ...initialSearchState, searchFilter: 'metrics.m1 = 5' };
    const updatedUIState = { ...initialUIState, viewMaximized: false };
    rerender(updatedSearchState, updatedUIState);

    // Click the copy button
    await waitFor(() => {
      expect(screen.getByTestId('share-link-copy-button')).toBeInTheDocument();
    });
    await userEvent.click(screen.getByTestId('share-link-copy-button'));

    // Save the second persisted tag name (containing serialized state hash), should be different from the first one
    const secondSavedTagName = jest.mocked(setExperimentTagApi).mock.lastCall?.[1];
    expect(firstSavedTagName).not.toEqual(secondSavedTagName);

    // Change the search state and UI state back to the initial values (but with new object references)
    const revertedSearchState = { ...updatedSearchState, searchFilter: 'metrics.m1 = 2' };
    const revertedUIState = { ...updatedUIState, viewMaximized: true };
    rerender(revertedSearchState, revertedUIState);

    await waitFor(() => {
      expect(screen.getByTestId('share-link-copy-button')).toBeInTheDocument();
    });

    await userEvent.click(screen.getByTestId('share-link-copy-button'));

    // Assert the third persisted tag name, should be the same as the first one
    const lastSavedTagName = jest.mocked(setExperimentTagApi).mock.lastCall?.[1];
    expect(lastSavedTagName).toEqual(firstSavedTagName);
  });
});
