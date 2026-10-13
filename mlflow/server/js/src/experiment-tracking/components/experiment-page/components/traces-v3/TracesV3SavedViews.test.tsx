import { jest, describe, beforeEach, test, expect } from '@jest/globals';
import { render, screen, waitFor, fireEvent } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import userEvent from '@testing-library/user-event';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';

import {
  TracesV3SavedViewsButton,
  TraceLiveViewStateProvider,
  TracePreviewActionsProvider,
  useTraceSavedViews,
} from './TracesV3SavedViews';
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

// The buttons take the useTraceSavedViews result as a prop (hoisted once in TracesV3View so both
// buttons share one Apollo subscription); these harnesses call the hook the same way for the tests.
const SavedViewsButtonHarness = ({ experimentId }: { experimentId: string }) => {
  const savedViews = useTraceSavedViews({ experimentId });
  return <TracesV3SavedViewsButton experimentId={experimentId} savedViews={savedViews} />;
};

// Mirrors the PRODUCTION component layering: useTraceSavedViews is hoisted ABOVE the
// TraceLiveViewStateProvider (which TracesV3Logs mounts), and only the button subtree is inside the
// provider. The hook therefore cannot read the live column/sort context itself — the modal, which
// renders inside the provider, must. A harness that wraps the hook in the provider (as the simpler
// tests below do) hides this because the hook then sees the provider; this one reproduces the tree
// that shipped a view with no columns/sort.
const HoistedLiveStateHarness = ({
  experimentId,
  live,
}: {
  experimentId: string;
  live: { selectedColumnIds: string[]; tableSort: { key: string; type: string; asc: boolean } };
}) => {
  const savedViews = useTraceSavedViews({ experimentId });
  return (
    <TraceLiveViewStateProvider value={live as any}>
      <TracesV3SavedViewsButton experimentId={experimentId} savedViews={savedViews} />
    </TraceLiveViewStateProvider>
  );
};

const renderButton = () =>
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <MockedReduxStoreProvider>
          <SavedViewsButtonHarness experimentId="exp-1" />
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

  test('at the saved-view cap, shows the at-cap message and disables Save without writing', async () => {
    // Seed 40 views (MAX_SAVED_VIEWS) so the experiment is exactly at the cap.
    const cappedTags = Array.from({ length: 40 }, (_, i) => ({
      key: `mlflow.traceViewState.cap${i}`,
      value: encodeSavedViewEnvelope(`View ${i}`, 'x', i),
    }));
    mockExperiment(cappedTags);
    renderButton();
    await openDropdown();

    await userEvent.click(screen.getByTestId('trace-saved-views-save-current'));
    expect(await screen.findByTestId('save-trace-view-at-cap-message')).toBeInTheDocument();
    // Even with a valid name typed, Save stays disabled and never dispatches a write.
    await userEvent.type(await screen.findByTestId('save-trace-view-name-input'), 'One too many');
    expect(screen.getByTestId('save-trace-view-save-button')).toBeDisabled();
    expect(mockSetExperimentTagApi).not.toHaveBeenCalled();
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
              <SavedViewsButtonHarness experimentId="exp-1" />
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

  test('captures live column/sort even when the hook is hoisted above the live-state provider', async () => {
    // Regression: the hook lives above TraceLiveViewStateProvider in production, so it reads null
    // from context; the save path must pull the live state at call time from the in-provider modal.
    // Before the fix this stored a view with no selectedColumns/sort (empty preview → "could not be
    // applied" on open).
    render(
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <MockedReduxStoreProvider>
            <HoistedLiveStateHarness
              experimentId="exp-1"
              live={{
                selectedColumnIds: ['request', 'response', 'tokens'],
                tableSort: { key: 'tokens', type: 'trace-info', asc: true },
              }}
            />
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
    const [, , tagValue] = mockSetExperimentTagApi.mock.calls[0];
    const envelope = JSON.parse(tagValue);
    const decodedState = JSON.parse(await textDecompressDeflate(envelope.state));
    expect(decodedState.single.selectedColumns).toEqual('request,response,tokens');
    expect(decodedState.single.sort).toEqual('tokens::trace-info::true');
  });

  test('shows an empty state when the experiment has no saved-view tags', async () => {
    mockExperiment([{ key: 'mlflow.note', value: 'x' }]);
    renderButton();
    await openDropdown();

    expect(screen.getByText(/No saved views yet/)).toBeInTheDocument();
  });

  test('the "Save & share current view" entry opens the save-and-share modal (name-first)', async () => {
    renderButton();
    await openDropdown();

    // No modal until the entry is clicked.
    expect(screen.queryByTestId('save-trace-view-name-input')).not.toBeInTheDocument();

    await userEvent.click(screen.getByTestId('trace-saved-views-save-current'));

    // Sharing routes through the named-view flow: the modal prompts for a view name before saving.
    expect(await screen.findByTestId('save-trace-view-name-input')).toBeInTheDocument();
  });

  test('surfaces Override/Discard in the menu (wired to preview actions) only when a shared view is active', async () => {
    const override = jest.fn();
    const discard = jest.fn();
    const renderWithPreview = (active: boolean) =>
      render(
        <IntlProvider locale="en">
          <DesignSystemProvider>
            <MockedReduxStoreProvider>
              <TracePreviewActionsProvider value={{ active, override, discard }}>
                <SavedViewsButtonHarness experimentId="exp-1" />
              </TracePreviewActionsProvider>
            </MockedReduxStoreProvider>
          </DesignSystemProvider>
        </IntlProvider>,
        {
          wrapper: ({ children }) => (
            <TestRouter routes={[testRoute(<>{children}</>, '/')]} history={history} initialEntries={['/']} />
          ),
        },
      );

    // Not previewing → no Override/Discard entries.
    const { unmount } = renderWithPreview(false);
    await openDropdown();
    expect(screen.queryByTestId('trace-saved-views-override-active')).not.toBeInTheDocument();
    expect(screen.queryByTestId('trace-saved-views-discard-active')).not.toBeInTheDocument();
    unmount();

    // Previewing → entries appear and invoke the bridged preview actions.
    renderWithPreview(true);
    await openDropdown();
    await userEvent.click(screen.getByTestId('trace-saved-views-override-active'));
    expect(override).toHaveBeenCalledTimes(1);
    await userEvent.click(screen.getByTestId('trace-saved-views-trigger'));
    await userEvent.click(screen.getByTestId('trace-saved-views-discard-active'));
    expect(discard).toHaveBeenCalledTimes(1);
  });
});

describe('useTraceSavedViews stale-tag refetch on active share key', () => {
  // Renders just the hook at a URL carrying the given share key, so we can assert its refetch
  // behavior without the button UI. `useGetExperimentQuery`/`refetch` are mocked at module scope.
  const HookProbe = ({ experimentId }: { experimentId: string }) => {
    useTraceSavedViews({ experimentId });
    return null;
  };

  const renderHookAt = (shareKey?: string) =>
    render(
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <MockedReduxStoreProvider>
            <HookProbe experimentId="exp-1" />
          </MockedReduxStoreProvider>
        </DesignSystemProvider>
      </IntlProvider>,
      {
        wrapper: ({ children }) => (
          <TestRouter
            routes={[testRoute(<>{children}</>, '/')]}
            history={history}
            initialEntries={[shareKey ? `/?traceViewShareKey=${shareKey}` : '/']}
          />
        ),
      },
    );

  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('refetches once when the active share key is not in the cached views (stale tag cache)', async () => {
    // The cached experiment has v1/v2 but the pasted link references a just-saved view absent here.
    mockExperiment(savedViewTags);

    renderHookAt('freshly-saved-id');

    await waitFor(() => expect(stableRefetch).toHaveBeenCalledTimes(1));
  });

  test('does not refetch again while the key stays missing after a tags update (anti-loop guard)', async () => {
    mockExperiment(savedViewTags);

    const { rerender } = renderHookAt('freshly-saved-id');
    await waitFor(() => expect(stableRefetch).toHaveBeenCalledTimes(1));

    // Simulate the refetch resolving with a NEW tags array that still lacks the key (e.g. a genuinely
    // deleted view): views changes identity → the effect re-runs → the per-key guard must hold and
    // NOT fire a second refetch, or a missing key would loop forever.
    mockExperiment([...savedViewTags]);
    rerender(
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <MockedReduxStoreProvider>
            <HookProbe experimentId="exp-1" />
          </MockedReduxStoreProvider>
        </DesignSystemProvider>
      </IntlProvider>,
    );

    // Assert via waitFor so a second refetch firing on a later tick would push the count past 1 and
    // fail this (a synchronous expect could pass before an async re-fire lands).
    await waitFor(() => expect(stableRefetch).toHaveBeenCalledTimes(1));
  });

  test('does not refetch when the active share key is already in the cached views', async () => {
    mockExperiment(savedViewTags);

    // render() flushes the mount effect under act, and the effect calls refetch() synchronously, so a
    // stray refetch would already be recorded here — no timing anchor needed.
    renderHookAt('v1');

    expect(stableRefetch).not.toHaveBeenCalled();
  });

  test('does not refetch when there is no active share key', async () => {
    mockExperiment(savedViewTags);

    renderHookAt();

    expect(stableRefetch).not.toHaveBeenCalled();
  });

  test('the trigger label catches up to the view name once the refetch reveals the tag', async () => {
    // Stale cache: the active share key's view isn't in the tags yet, so the trigger shows the
    // generic "Views" and a refetch fires.
    mockExperiment(savedViewTags);

    const { rerender } = render(
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <MockedReduxStoreProvider>
            <SavedViewsButtonHarness experimentId="exp-1" />
          </MockedReduxStoreProvider>
        </DesignSystemProvider>
      </IntlProvider>,
      {
        wrapper: ({ children }) => (
          <TestRouter
            routes={[testRoute(<>{children}</>, '/')]}
            history={history}
            initialEntries={['/?traceViewShareKey=v3']}
          />
        ),
      },
    );

    expect(screen.getByTestId('trace-saved-views-trigger')).toHaveTextContent('Views');
    await waitFor(() => expect(stableRefetch).toHaveBeenCalledTimes(1));

    // Refetch resolves with the freshly-saved view now present → the trigger shows its name.
    mockExperiment([
      ...savedViewTags,
      { key: 'mlflow.traceViewState.v3', value: encodeSavedViewEnvelope('P95 spikes', 'x', 3000) },
    ]);
    rerender(
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <MockedReduxStoreProvider>
            <SavedViewsButtonHarness experimentId="exp-1" />
          </MockedReduxStoreProvider>
        </DesignSystemProvider>
      </IntlProvider>,
    );

    await waitFor(() => expect(screen.getByTestId('trace-saved-views-trigger')).toHaveTextContent('P95 spikes'));
  });
});
