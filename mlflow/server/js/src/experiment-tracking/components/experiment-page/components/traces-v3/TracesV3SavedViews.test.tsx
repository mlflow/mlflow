import { jest, describe, beforeEach, test, expect } from '@jest/globals';
import { render, screen, waitFor, fireEvent } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import userEvent from '@testing-library/user-event';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';

import { TracesV3SavedViewsButton, TracesV3ShareButton, TraceLiveViewStateProvider } from './TracesV3SavedViews';
import { MockedReduxStoreProvider } from '../../../../../common/utils/TestUtils';
import { setupTestRouter, testRoute, TestRouter } from '../../../../../common/utils/RoutingTestUtils';
import { useGetExperimentQuery } from '@mlflow/mlflow/src/experiment-tracking/hooks/useExperimentQuery';
import { encodeSavedViewEnvelope } from '../../utils/savedViewEnvelope';
import { textDecompressDeflate } from '@mlflow/mlflow/src/common/utils/StringUtils';
import Utils from '@mlflow/mlflow/src/common/utils/Utils';

jest.mock('@mlflow/mlflow/src/experiment-tracking/hooks/useExperimentQuery', () => ({
  useGetExperimentQuery: jest.fn(),
}));

// `mock`-prefixed so jest's mock-factory hoisting allows referencing them.
const mockSetExperimentTagApi = jest.fn((..._args: string[]) => ({
  type: 'SET_EXPERIMENT_TAG',
  payload: Promise.resolve({}),
}));
const mockDeleteExperimentTagApi = jest.fn((..._args: string[]) => ({
  type: 'DELETE_EXPERIMENT_TAG',
  payload: Promise.resolve({}),
}));
jest.mock('@mlflow/mlflow/src/experiment-tracking/actions', () => ({
  setExperimentTagApi: (...args: string[]) => mockSetExperimentTagApi(...args),
  deleteExperimentTagApi: (...args: string[]) => mockDeleteExperimentTagApi(...args),
}));

const mockCopyToClipboard = jest.fn(async (_text: string) => true);
jest.mock('@mlflow/mlflow/src/common/utils/copyToClipboard', () => ({
  copyToClipboard: (text: string) => mockCopyToClipboard(text),
}));

// Two saved-view tags on the experiment, plus an unrelated tag that must be ignored.
const savedViewTags = [
  { key: 'mlflow.traceViewState.v1', value: encodeSavedViewEnvelope('Latency triage', 'x', 1000) },
  { key: 'mlflow.traceViewState.v2', value: encodeSavedViewEnvelope('Error traces', 'x', 2000) },
  { key: 'mlflow.note', value: 'not a view' },
];

// Return a STABLE object reference so re-renders don't produce a new `experiment.tags` array each
// pass (which would re-run the views useMemo and thrash userEvent typing into a timeout).
const stableRefetch = jest.fn(() => Promise.resolve({}));
const mockExperiment = (tags: { key: string; value: string }[]) => {
  const value = { data: { tags }, refetch: stableRefetch } as never;
  jest.mocked(useGetExperimentQuery).mockReturnValue(value);
};

const { history } = setupTestRouter();

const renderButton = () =>
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <MockedReduxStoreProvider>
          <TracesV3SavedViewsButton experimentId="exp-1" />
        </MockedReduxStoreProvider>
      </DesignSystemProvider>
    </IntlProvider>,
    {
      wrapper: ({ children }) => (
        <TestRouter routes={[testRoute(<>{children}</>, '/')]} history={history} initialEntries={['/']} />
      ),
    },
  );

const openDropdown = async () => {
  await userEvent.click(screen.getByTestId('trace-saved-views-trigger'));
};

describe('TracesV3SavedViewsButton', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockExperiment(savedViewTags);
  });

  test('lists the saved views parsed from experiment tags (newest first, non-view tags ignored)', async () => {
    renderButton();
    await openDropdown();

    expect(screen.getByText('Latency triage')).toBeInTheDocument();
    expect(screen.getByText('Error traces')).toBeInTheDocument();
    expect(screen.queryByText('not a view')).not.toBeInTheDocument();
  });

  test('filters the list by search text', async () => {
    renderButton();
    await openDropdown();

    fireEvent.change(screen.getByTestId('trace-saved-views-search'), { target: { value: 'error' } });
    await waitFor(() => {
      expect(screen.queryByText('Latency triage')).not.toBeInTheDocument();
      expect(screen.getByText('Error traces')).toBeInTheDocument();
    });
  });

  test('deleting a view requires confirmation then dispatches delete for the right tag', async () => {
    renderButton();
    await openDropdown();

    await userEvent.click(screen.getByTestId('trace-saved-views-delete-v1'));
    expect(mockDeleteExperimentTagApi).not.toHaveBeenCalled();
    const confirm = await screen.findByText('Delete');
    await userEvent.click(confirm, { pointerEventsCheck: 0 });
    expect(mockDeleteExperimentTagApi).toHaveBeenCalledWith('exp-1', 'mlflow.traceViewState.v1');
  });

  test('opening the save modal shows the name input', async () => {
    renderButton();
    await openDropdown();

    await userEvent.click(screen.getByTestId('trace-saved-views-save-current'));
    expect(await screen.findByTestId('save-trace-view-name-input')).toBeInTheDocument();
  });

  test('rejects a duplicate view name (case-insensitive, trimmed) without writing a tag', async () => {
    const errorSpy = jest.spyOn(Utils, 'displayGlobalErrorNotification').mockImplementation(() => {});
    renderButton();
    await openDropdown();

    await userEvent.click(screen.getByTestId('trace-saved-views-save-current'));
    // "Latency triage" already exists; a differently-cased, padded variant must still be rejected.
    await userEvent.type(await screen.findByTestId('save-trace-view-name-input'), '  latency TRIAGE  ');
    await userEvent.click(screen.getByTestId('save-trace-view-save-button'));

    await waitFor(() => expect(errorSpy).toHaveBeenCalled());
    expect(String(errorSpy.mock.calls[0][0])).toMatch(/already exists/i);
    expect(mockSetExperimentTagApi).not.toHaveBeenCalled();
    errorSpy.mockRestore();
  });

  test('saving captures the live column/sort selection from context (not the empty URL)', async () => {
    render(
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <MockedReduxStoreProvider>
            <TraceLiveViewStateProvider
              value={{
                selectedColumnIds: ['request', 'response', 'tokens'],
                tableSort: { key: 'tokens', type: 'trace-info' as any, asc: true },
              }}
            >
              <TracesV3SavedViewsButton experimentId="exp-1" />
            </TraceLiveViewStateProvider>
          </MockedReduxStoreProvider>
        </DesignSystemProvider>
      </IntlProvider>,
      {
        wrapper: ({ children }) => (
          <TestRouter routes={[testRoute(<>{children}</>, '/')]} history={history} initialEntries={['/']} />
        ),
      },
    );
    await openDropdown();
    await userEvent.click(screen.getByTestId('trace-saved-views-save-current'));
    await userEvent.type(await screen.findByTestId('save-trace-view-name-input'), 'Token view');
    await userEvent.click(screen.getByTestId('save-trace-view-save-button'));

    await waitFor(() => expect(mockSetExperimentTagApi).toHaveBeenCalled());
    const [, tagKey, tagValue] = mockSetExperimentTagApi.mock.calls[0];
    expect(tagKey).toMatch(/^mlflow\.traceViewState\./);
    const envelope = JSON.parse(tagValue);
    const decodedState = JSON.parse(await textDecompressDeflate(envelope.state));
    // The live columns/sort are captured in the URL wire format the preview decoder reads.
    expect(decodedState.single.selectedColumns).toEqual('request,response,tokens');
    expect(decodedState.single.sort).toEqual('tokens::trace-info::true');
  });

  test('shows an empty state when the experiment has no saved-view tags', async () => {
    mockExperiment([{ key: 'mlflow.note', value: 'x' }]);
    renderButton();
    await openDropdown();

    expect(screen.getByText(/No saved views yet/)).toBeInTheDocument();
  });
});

describe('TracesV3ShareButton', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockCopyToClipboard.mockResolvedValue(true);
  });

  test('opens the Save & share view modal (name-first) rather than copying an anonymous link', async () => {
    render(
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <MockedReduxStoreProvider>
            <TracesV3ShareButton experimentId="exp-1" />
          </MockedReduxStoreProvider>
        </DesignSystemProvider>
      </IntlProvider>,
      {
        wrapper: ({ children }) => (
          <TestRouter routes={[testRoute(<>{children}</>, '/')]} history={history} initialEntries={['/']} />
        ),
      },
    );

    // No modal and no anonymous link until the button is clicked.
    expect(screen.queryByTestId('save-trace-view-name-input')).not.toBeInTheDocument();

    await userEvent.click(screen.getByTestId('trace-share-button'));

    // Sharing routes through the named-view flow: the modal prompts for a view name before saving,
    // and nothing is copied to the clipboard just by opening it.
    expect(await screen.findByTestId('save-trace-view-name-input')).toBeInTheDocument();
    expect(mockCopyToClipboard).not.toHaveBeenCalled();
  });
});
