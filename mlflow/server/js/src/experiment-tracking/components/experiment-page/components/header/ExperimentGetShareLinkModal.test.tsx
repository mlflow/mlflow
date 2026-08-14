import { jest, describe, beforeAll, afterAll, beforeEach, test, expect } from '@jest/globals';
import type { ExperimentPageSearchFacetsState } from '../../models/ExperimentPageSearchFacetsState';
import { createExperimentPageSearchFacetsState } from '../../models/ExperimentPageSearchFacetsState';
import type { ExperimentPageUIState } from '../../models/ExperimentPageUIState';
import { createExperimentPageUIState } from '../../models/ExperimentPageUIState';
import { ExperimentGetShareLinkModal, viewNameExists } from './ExperimentGetShareLinkModal';
import { MockedReduxStoreProvider } from '../../../../../common/utils/TestUtils';
import { render, screen, waitFor } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';
import { shouldUseCompressedExperimentViewSharedState } from '../../../../../common/utils/FeatureUtils';
import { IntlProvider } from 'react-intl';
import { setupTestRouter, testRoute, TestRouter } from '../../../../../common/utils/RoutingTestUtils';
import { DesignSystemProvider } from '@databricks/design-system';
import { MlflowService } from '../../../../sdk/MlflowService';
import {
  encodeSavedViewEnvelope,
  decodeSavedViewEnvelope,
  getSavedViewIdFromTagKey,
  getSavedViewTagKey,
} from '../../utils/savedViewEnvelope';
import { ExperimentTag } from '../../../../sdk/MlflowMessages';
import Utils from '../../../../../common/utils/Utils';

jest.mock('../../../../../common/utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('../../../../../common/utils/FeatureUtils')>(
    '../../../../../common/utils/FeatureUtils',
  ),
  shouldUseCompressedExperimentViewSharedState: jest.fn(),
}));

jest.mock('../../../../sdk/MlflowService', () => ({
  MlflowService: {
    setExperimentTag: jest.fn(() => Promise.resolve({})),
  },
}));

const experimentId = 'experiment-1';

describe('viewNameExists', () => {
  const views = [
    { id: 'a', name: 'Prod runs', createdAt: 1 },
    { id: 'b', name: 'Best F1', createdAt: 2 },
  ];

  test('matches case-insensitively and trims surrounding whitespace', () => {
    expect(viewNameExists(views, '  prod RUNS ')).toBe(true);
    expect(viewNameExists(views, 'Best F1')).toBe(true);
  });

  test('returns false for a name that is not present', () => {
    expect(viewNameExists(views, 'Prod runs 2')).toBe(false);
    expect(viewNameExists([], 'anything')).toBe(false);
  });
});

describe('ExperimentGetShareLinkModal', () => {
  const onCancel = jest.fn();
  const { history } = setupTestRouter();

  let navigatorClipboard: Clipboard;
  let copiedText = '';

  beforeAll(() => {
    navigatorClipboard = navigator.clipboard;
    // @ts-expect-error: navigator is overridable in tests
    navigator.clipboard = {
      writeText: jest.fn((text: string) => {
        copiedText = text;
        return Promise.resolve();
      }),
    };
  });

  afterAll(() => {
    jest.restoreAllMocks();
    // @ts-expect-error: navigator is overridable in tests
    navigator.clipboard = navigatorClipboard;
  });

  beforeEach(() => {
    jest.clearAllMocks();
    jest.mocked(shouldUseCompressedExperimentViewSharedState).mockImplementation(() => true);
    jest.mocked(MlflowService.setExperimentTag).mockImplementation(() => Promise.resolve({}) as any);
  });

  // Seed existing saved views into the redux slice the modal reads for uniqueness checks, in the
  // same Immutable-record shape the reducer stores.
  const makeStateWithViews = (views: { id: string; name: string }[]) => {
    const tagObj: Record<string, unknown> = {};
    views.forEach(({ id, name }) => {
      const key = getSavedViewTagKey(id);
      tagObj[key] = (ExperimentTag as any).fromJs({ key, value: encodeSavedViewEnvelope(name, 'deflate;xxx', 1000) });
    });
    return { entities: { experimentTagsByExperimentId: { [experimentId]: tagObj } } } as any;
  };

  const renderModal = ({
    searchFacetsState = createExperimentPageSearchFacetsState(),
    uiState = createExperimentPageUIState(),
    initialUrl = '/',
    existingViews = [],
  }: {
    searchFacetsState?: ExperimentPageSearchFacetsState;
    uiState?: ExperimentPageUIState;
    initialUrl?: string;
    existingViews?: { id: string; name: string }[];
  } = {}) => {
    const Component = () => {
      const [visible, setVisible] = useState(false);
      return (
        <IntlProvider locale="en">
          <DesignSystemProvider>
            <MockedReduxStoreProvider state={makeStateWithViews(existingViews)}>
              <button onClick={() => setVisible(true)}>open modal</button>
              <ExperimentGetShareLinkModal
                experimentId={experimentId}
                onCancel={onCancel}
                searchFacetsState={searchFacetsState}
                uiState={uiState}
                visible={visible}
              />
            </MockedReduxStoreProvider>
          </DesignSystemProvider>
        </IntlProvider>
      );
    };
    render(<Component />, {
      wrapper: ({ children }) => (
        <TestRouter routes={[testRoute(<>{children}</>, '/')]} history={history} initialEntries={[initialUrl]} />
      ),
    });
  };

  test('shows a name-entry field on open and does not write a tag yet (single experiment)', async () => {
    renderModal();
    await userEvent.click(screen.getByText('open modal'));

    // Name input is shown; no tag write happened just by opening.
    expect(screen.getByTestId('save-view-name-input')).toBeInTheDocument();
    expect(MlflowService.setExperimentTag).not.toHaveBeenCalled();
  });

  test('saving writes an envelope-encoded saved-view tag and then shows a copyable id-based link', async () => {
    const initialSearchState = { ...createExperimentPageSearchFacetsState(), searchFilter: 'metrics.m1 = 2' };
    const initialUIState = { ...createExperimentPageUIState(), viewMaximized: true };
    renderModal({ searchFacetsState: initialSearchState, uiState: initialUIState });

    await userEvent.click(screen.getByText('open modal'));
    await userEvent.type(screen.getByTestId('save-view-name-input'), 'My named view');
    await userEvent.click(screen.getByTestId('save-view-save-button'));

    // The tag write carries an id-keyed saved-view tag with an envelope-shaped value.
    await waitFor(() => expect(MlflowService.setExperimentTag).toHaveBeenCalledTimes(1));
    const call = jest.mocked(MlflowService.setExperimentTag).mock.calls[0][0] as {
      experiment_id: string;
      key: string;
      value: string;
    };
    expect(call.experiment_id).toBe('experiment-1');
    const id = getSavedViewIdFromTagKey(call.key);
    expect(id).toBeTruthy();
    const envelope = decodeSavedViewEnvelope(call.value);
    expect(envelope.name).toBe('My named view');
    expect(typeof envelope.createdAt).toBe('number');

    // Phase 2: the copyable link references the saved view by its bare id (not an embedded blob).
    await waitFor(() => expect(screen.getByTestId('share-link-copy-button')).toBeInTheDocument());
    await userEvent.click(screen.getByTestId('share-link-copy-button'));
    expect(copiedText).toMatch(new RegExp(`viewStateShareKey=${id}(?:$|&)`));
    expect(copiedText).toMatch(/\/experiments\/experiment-1\/runs/);
    // The link must NOT embed the serialized state blob.
    expect(copiedText).not.toContain('deflate;');
  });

  test('propagates the experiment page view mode into the saved-view link', async () => {
    renderModal({ initialUrl: '/?compareRunsMode=CHART' });
    await userEvent.click(screen.getByText('open modal'));
    await userEvent.type(screen.getByTestId('save-view-name-input'), 'Charts view');
    await userEvent.click(screen.getByTestId('save-view-save-button'));

    await waitFor(() => expect(screen.getByTestId('share-link-copy-button')).toBeInTheDocument());
    const input = screen.getByTestId('share-link-input') as HTMLInputElement;
    expect(input.value).toContain('compareRunsMode=CHART');
    expect(input.value).toContain('viewStateShareKey=');
  });

  test('surfaces a clear error and writes no tag when the view exceeds the tag size limit', async () => {
    // Turn compression off so the serialized state stays large, and give it a payload that blows
    // past the 5000-char experiment-tag ceiling.
    jest.mocked(shouldUseCompressedExperimentViewSharedState).mockImplementation(() => false);
    const errorSpy = jest.spyOn(Utils, 'displayGlobalErrorNotification').mockImplementation(() => {});
    const hugeSearchState = { ...createExperimentPageSearchFacetsState(), searchFilter: 'x'.repeat(6000) };
    renderModal({ searchFacetsState: hugeSearchState });

    await userEvent.click(screen.getByText('open modal'));
    await userEvent.type(screen.getByTestId('save-view-name-input'), 'Too big');
    await userEvent.click(screen.getByTestId('save-view-save-button'));

    await waitFor(() => expect(errorSpy).toHaveBeenCalled());
    expect(String(errorSpy.mock.calls[0][0])).toMatch(/too large to save/i);
    // The oversized write must be blocked before hitting the backend, and the modal stays on the
    // name-entry phase so the user can trim and retry.
    expect(MlflowService.setExperimentTag).not.toHaveBeenCalled();
    expect(screen.getByTestId('save-view-name-input')).toBeInTheDocument();
  });

  test('does not disable the Save button while a name is present but keeps the modal open on save failure', async () => {
    jest.spyOn(console, 'error').mockImplementation(() => {});
    jest
      .mocked(MlflowService.setExperimentTag)
      .mockImplementation(() => Promise.reject(new Error('write failed')) as any);

    renderModal();
    await userEvent.click(screen.getByText('open modal'));
    await userEvent.type(screen.getByTestId('save-view-name-input'), 'Will fail');
    await userEvent.click(screen.getByTestId('save-view-save-button'));

    // The name-entry phase stays visible (no link produced) so the user can retry.
    await waitFor(() => expect(MlflowService.setExperimentTag).toHaveBeenCalled());
    expect(screen.getByTestId('save-view-name-input')).toBeInTheDocument();
    expect(screen.queryByTestId('share-link-copy-button')).not.toBeInTheDocument();
    // eslint-disable-next-line no-console -- TODO(FEINF-3587)
    jest.mocked(console.error).mockRestore();
  });

  test('blocks a duplicate name (case-insensitive) and writes no tag', async () => {
    const errorSpy = jest.spyOn(Utils, 'displayGlobalErrorNotification').mockImplementation(() => {});
    renderModal({ existingViews: [{ id: 'existing', name: 'My view' }] });

    await userEvent.click(screen.getByText('open modal'));
    // Same name, different case — must still be rejected.
    await userEvent.type(screen.getByTestId('save-view-name-input'), '  my VIEW  ');
    await userEvent.click(screen.getByTestId('save-view-save-button'));

    await waitFor(() => expect(errorSpy).toHaveBeenCalled());
    expect(String(errorSpy.mock.calls[0][0])).toMatch(/already exists/i);
    expect(MlflowService.setExperimentTag).not.toHaveBeenCalled();
    // Stays on the name-entry phase so the user can rename and retry.
    expect(screen.getByTestId('save-view-name-input')).toBeInTheDocument();
  });

  test('allows a unique name even when other views exist', async () => {
    renderModal({ existingViews: [{ id: 'existing', name: 'Some other view' }] });

    await userEvent.click(screen.getByText('open modal'));
    await userEvent.type(screen.getByTestId('save-view-name-input'), 'A brand new name');
    await userEvent.click(screen.getByTestId('save-view-save-button'));

    await waitFor(() => expect(MlflowService.setExperimentTag).toHaveBeenCalledTimes(1));
  });

  test('disables Save, shows an at-cap message, and writes no tag when the view cap is reached', async () => {
    const atCap = Array.from({ length: 40 }, (_, i) => ({ id: `v${i}`, name: `View ${i}` }));
    renderModal({ existingViews: atCap });

    await userEvent.click(screen.getByText('open modal'));
    // A brand-new (unique) name is still blocked because the experiment is at the view cap.
    await userEvent.type(screen.getByTestId('save-view-name-input'), 'One too many');

    const saveButton = screen.getByTestId('save-view-save-button');
    expect(saveButton).toBeDisabled();
    expect(screen.getByText(/maximum of 40 saved views/i)).toBeInTheDocument();

    await userEvent.click(saveButton);
    expect(MlflowService.setExperimentTag).not.toHaveBeenCalled();
  });

  test('still allows saving when just below the view cap', async () => {
    const belowCap = Array.from({ length: 39 }, (_, i) => ({ id: `v${i}`, name: `View ${i}` }));
    renderModal({ existingViews: belowCap });

    await userEvent.click(screen.getByText('open modal'));
    await userEvent.type(screen.getByTestId('save-view-name-input'), 'The fortieth');
    await userEvent.click(screen.getByTestId('save-view-save-button'));

    await waitFor(() => expect(MlflowService.setExperimentTag).toHaveBeenCalledTimes(1));
  });
});
