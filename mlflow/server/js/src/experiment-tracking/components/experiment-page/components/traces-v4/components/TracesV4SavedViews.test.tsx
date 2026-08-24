import { jest, describe, beforeEach, test, expect } from '@jest/globals';
import { render, screen, waitFor, fireEvent, act } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import userEvent from '@testing-library/user-event';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';

import { TracesV4SavedViewsButton, TracesV4SharedViewBanner, useTracesV4SavedViews } from './TracesV4SavedViews';
import { MockedReduxStoreProvider } from '@mlflow/mlflow/src/common/utils/TestUtils';
import { setupTestRouter, testRoute, TestRouter } from '@mlflow/mlflow/src/common/utils/RoutingTestUtils';
import { useSearchParams } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import { useGetExperimentQuery } from '@mlflow/mlflow/src/experiment-tracking/hooks/useExperimentQuery';
import { encodeSavedViewEnvelope } from '../../../utils/savedViewEnvelope';
import { textCompressDeflate, textDecompressDeflate } from '@mlflow/mlflow/src/common/utils/StringUtils';
import Utils from '@mlflow/mlflow/src/common/utils/Utils';
import { TRACE_V4_SHARE_URL_PARAM_KEY, buildV4ViewQuery, captureV4ViewState } from '../utils/tracesV4SavedViewState';
import type { TraceColumnId } from '@databricks/web-shared/traces-table';

jest.mock('@mlflow/mlflow/src/experiment-tracking/hooks/useExperimentQuery', () => ({
  useGetExperimentQuery: jest.fn(),
}));

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

// Build a V4 saved-view tag whose state encodes the given URL query + columns.
const makeV4ViewTag = async (
  id: string,
  name: string,
  createdAt: number,
  query = 'q=x',
  cols: TraceColumnId[] = ['start_time'],
) => {
  const state = captureV4ViewState(new URLSearchParams(query), cols);
  const compressed = await textCompressDeflate(JSON.stringify(state));
  return { key: `mlflow.tracesV4ViewState.${id}`, value: encodeSavedViewEnvelope(name, compressed, createdAt) };
};

const stableRefetch = jest.fn(() => Promise.resolve({}));
const mockExperiment = (tags: { key: string; value: string }[]) => {
  jest.mocked(useGetExperimentQuery).mockReturnValue({ data: { tags }, refetch: stableRefetch } as never);
};

const { history } = setupTestRouter();

const SavedViewsButtonHarness = ({ experimentId }: { experimentId: string }) => {
  const savedViews = useTracesV4SavedViews({
    experimentId,
    visibleColumns: ['start_time', 'input'],
    setColumns: jest.fn(),
  });
  return <TracesV4SavedViewsButton experimentId={experimentId} savedViews={savedViews} />;
};

const renderButtonAt = (entry = '/') =>
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
        <TestRouter routes={[testRoute(<>{children}</>, '/')]} history={history} initialEntries={[entry]} />
      ),
    },
  );

const openDropdown = async () => {
  await userEvent.click(screen.getByTestId('trace-v4-saved-views-trigger'));
};

describe('TracesV4SavedViewsButton', () => {
  let viewTags: { key: string; value: string }[];

  beforeEach(async () => {
    jest.clearAllMocks();
    viewTags = [
      await makeV4ViewTag('v1', 'Latency triage', 1000),
      await makeV4ViewTag('v2', 'Error traces', 2000),
      { key: 'mlflow.note', value: 'not a view' },
    ];
    mockExperiment(viewTags);
  });

  test('lists the V4 saved views parsed from experiment tags (newest first, non-view tags ignored)', async () => {
    renderButtonAt();
    await openDropdown();
    expect(screen.getByText('Error traces')).toBeInTheDocument();
    expect(screen.getByText('Latency triage')).toBeInTheDocument();
    expect(screen.queryByText('not a view')).not.toBeInTheDocument();
  });

  test('filters the list by search text', async () => {
    renderButtonAt();
    await openDropdown();
    fireEvent.change(screen.getByTestId('trace-v4-saved-views-search'), { target: { value: 'error' } });
    await waitFor(() => {
      expect(screen.queryByText('Latency triage')).not.toBeInTheDocument();
      expect(screen.getByText('Error traces')).toBeInTheDocument();
    });
  });

  test('deleting a view requires confirmation then dispatches delete for the right tag', async () => {
    renderButtonAt();
    await openDropdown();
    await userEvent.click(screen.getByTestId('trace-v4-saved-views-delete-v1'));
    expect(mockDeleteExperimentTagApi).not.toHaveBeenCalled();
    await userEvent.click(await screen.findByText('Delete'), { pointerEventsCheck: 0 });
    expect(mockDeleteExperimentTagApi).toHaveBeenCalledWith('exp-1', 'mlflow.tracesV4ViewState.v1');
  });

  test('saving captures the current URL + live columns into a compressed envelope tag', async () => {
    renderButtonAt('/?q=refund&sort=duration&dir=asc');
    await openDropdown();
    await userEvent.click(screen.getByTestId('trace-v4-saved-views-save-current'));
    await userEvent.type(await screen.findByTestId('save-trace-v4-view-name-input'), 'My view');
    await userEvent.click(screen.getByTestId('save-trace-v4-view-save-button'));

    await waitFor(() => expect(mockSetExperimentTagApi).toHaveBeenCalled());
    const [expId, tagKey, tagValue] = mockSetExperimentTagApi.mock.calls[0];
    expect(expId).toBe('exp-1');
    expect(tagKey).toMatch(/^mlflow\.tracesV4ViewState\./);
    const decoded = JSON.parse(await textDecompressDeflate(JSON.parse(tagValue).state));
    const out = new URLSearchParams(buildV4ViewQuery(decoded, 'x'));
    expect(out.get('q')).toBe('refund');
    expect(out.get('sort')).toBe('duration');
    expect(out.get('cols')).toBe('start_time,input'); // the harness's live columns
  });

  test('saving a new view activates it: the URL gains the new view share key', async () => {
    let currentSearch = '';
    const SearchReporter = () => {
      const [params] = useSearchParams();
      currentSearch = params.toString();
      return null;
    };
    render(
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <MockedReduxStoreProvider>
            <SearchReporter />
            <SavedViewsButtonHarness experimentId="exp-1" />
          </MockedReduxStoreProvider>
        </DesignSystemProvider>
      </IntlProvider>,
      {
        wrapper: ({ children }) => (
          <TestRouter routes={[testRoute(<>{children}</>, '/')]} history={history} initialEntries={['/?q=hello']} />
        ),
      },
    );
    await openDropdown();
    await userEvent.click(screen.getByTestId('trace-v4-saved-views-save-current'));
    await userEvent.type(await screen.findByTestId('save-trace-v4-view-name-input'), 'My new view');
    await userEvent.click(screen.getByTestId('save-trace-v4-view-save-button'));

    await waitFor(() => expect(mockSetExperimentTagApi).toHaveBeenCalled());
    // Saving activates the view directly from the captured state (no decode round-trip), so its share
    // key lands in the URL — a copied link then points at the just-saved view.
    const savedId = mockSetExperimentTagApi.mock.calls[0][1].replace('mlflow.tracesV4ViewState.', '');
    await waitFor(() => expect(new URLSearchParams(currentSearch).get(TRACE_V4_SHARE_URL_PARAM_KEY)).toBe(savedId));
    // Save + deflate compression + URL propagation is slow under parallel jsdom load — bump off 5s.
  }, 20000);

  test('rejects a duplicate view name (case-insensitive, trimmed) without writing a tag', async () => {
    const errorSpy = jest.spyOn(Utils, 'displayGlobalErrorNotification').mockImplementation(() => {});
    renderButtonAt();
    await openDropdown();
    await userEvent.click(screen.getByTestId('trace-v4-saved-views-save-current'));
    await userEvent.type(await screen.findByTestId('save-trace-v4-view-name-input'), '  latency TRIAGE  ');
    await userEvent.click(screen.getByTestId('save-trace-v4-view-save-button'));

    await waitFor(() => expect(errorSpy).toHaveBeenCalled());
    expect(String(errorSpy.mock.calls[0][0])).toMatch(/already exists/i);
    expect(mockSetExperimentTagApi).not.toHaveBeenCalled();
    errorSpy.mockRestore();
  });

  test('at the saved-view cap, shows the at-cap message and disables Save without writing', async () => {
    const capped = await Promise.all(Array.from({ length: 40 }, (_, i) => makeV4ViewTag(`cap${i}`, `View ${i}`, i)));
    mockExperiment(capped);
    renderButtonAt();
    await openDropdown();
    await userEvent.click(screen.getByTestId('trace-v4-saved-views-save-current'));
    expect(await screen.findByTestId('save-trace-v4-view-at-cap-message')).toBeInTheDocument();
    await userEvent.type(await screen.findByTestId('save-trace-v4-view-name-input'), 'One too many');
    expect(screen.getByTestId('save-trace-v4-view-save-button')).toBeDisabled();
    expect(mockSetExperimentTagApi).not.toHaveBeenCalled();
  });

  test('trigger shows the active view name when a known share key is in the URL', async () => {
    renderButtonAt(`/?${TRACE_V4_SHARE_URL_PARAM_KEY}=v1`);
    expect(screen.getByTestId('trace-v4-saved-views-trigger')).toHaveTextContent('Latency triage');
  });

  test('opening a view rewrites the URL to the view stored query + share key', async () => {
    // v1 was stored with q=x and cols=[start_time] (the harness default).
    let currentSearch = '';
    const SearchReporter = () => {
      const [params] = useSearchParams();
      currentSearch = params.toString();
      return null;
    };
    render(
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <MockedReduxStoreProvider>
            <SearchReporter />
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
    await openDropdown();
    await userEvent.click(screen.getByTestId('trace-v4-saved-views-item-v1'));
    await waitFor(() => {
      const out = new URLSearchParams(currentSearch);
      expect(out.get(TRACE_V4_SHARE_URL_PARAM_KEY)).toBe('v1');
      expect(out.get('q')).toBe('x');
      expect(out.get('cols')).toBe('start_time');
    });
  });

  test('opening a corrupt view shows an error toast and does not navigate', async () => {
    const errorSpy = jest.spyOn(Utils, 'displayGlobalErrorNotification').mockImplementation(() => {});
    // A view tag whose envelope is valid but whose state blob is not decodable JSON.
    mockExperiment([
      { key: 'mlflow.tracesV4ViewState.bad', value: encodeSavedViewEnvelope('Broken', 'not-base64', 5) },
    ]);
    renderButtonAt();
    await openDropdown();
    await userEvent.click(screen.getByTestId('trace-v4-saved-views-item-bad'));
    await waitFor(() => expect(errorSpy).toHaveBeenCalled());
    expect(String(errorSpy.mock.calls[0][0])).toMatch(/could not be opened/i);
    errorSpy.mockRestore();
  });

  test('copy-link copies the view own stored share URL and shows a success toast', async () => {
    const infoSpy = jest.spyOn(Utils, 'displayGlobalInfoNotification').mockImplementation(() => {});
    renderButtonAt();
    await openDropdown();
    await userEvent.click(screen.getByTestId('trace-v4-saved-views-copy-link-v1'));
    await waitFor(() => expect(mockCopyToClipboard).toHaveBeenCalled());
    const copied = mockCopyToClipboard.mock.calls[0][0];
    // The link carries the view's OWN stored state (q=x, cols=start_time) + its share key.
    const copiedQuery = new URLSearchParams(copied.split('?')[1]);
    expect(copiedQuery.get(TRACE_V4_SHARE_URL_PARAM_KEY)).toBe('v1');
    expect(copiedQuery.get('q')).toBe('x');
    await waitFor(() => expect(infoSpy).toHaveBeenCalled());
    infoSpy.mockRestore();
  });

  test('save→open round-trips the full captured state (all whitelisted params + cols)', async () => {
    // Save from a rich URL, then decode the written tag and confirm every whitelisted param survives
    // the compress→decompress round trip (not just q/sort).
    renderButtonAt(
      '/?q=refund&sort=duration&dir=asc&pageSize=50&tag=env%3Dprod&tag=team%3Dml&startTimeLabel=LAST_7_DAYS',
    );
    await openDropdown();
    await userEvent.click(screen.getByTestId('trace-v4-saved-views-save-current'));
    await userEvent.type(await screen.findByTestId('save-trace-v4-view-name-input'), 'Round trip');
    await userEvent.click(screen.getByTestId('save-trace-v4-view-save-button'));

    await waitFor(() => expect(mockSetExperimentTagApi).toHaveBeenCalled());
    const [, , tagValue] = mockSetExperimentTagApi.mock.calls[0];
    const decoded = JSON.parse(await textDecompressDeflate(JSON.parse(tagValue).state));
    const out = new URLSearchParams(buildV4ViewQuery(decoded, 'x'));
    expect(out.get('q')).toBe('refund');
    expect(out.get('sort')).toBe('duration');
    expect(out.get('dir')).toBe('asc');
    expect(out.get('pageSize')).toBe('50');
    expect(out.get('startTimeLabel')).toBe('LAST_7_DAYS');
    expect(out.getAll('tag')).toEqual(['env=prod', 'team=ml']);
    expect(out.get('cols')).toBe('start_time,input'); // the harness's live columns
  });
});

describe('useTracesV4SavedViews preview / override / discard', () => {
  const setColumns = jest.fn();
  // Report BOTH the hook API and the live URL search each render, so assertions read the in-memory
  // router's state (TestRouter never touches window.location.hash) after a param rewrite.
  const PreviewProbe = ({ onRender }: { onRender: (s: any, search: string) => void }) => {
    const savedViews = useTracesV4SavedViews({
      experimentId: 'exp-1',
      visibleColumns: ['start_time', 'input'],
      setColumns,
    });
    const [params] = useSearchParams();
    onRender(savedViews, params.toString());
    return null;
  };

  const renderProbeAt = (entry: string, onRender: (s: any, search: string) => void) =>
    render(
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <MockedReduxStoreProvider>
            <PreviewProbe onRender={onRender} />
          </MockedReduxStoreProvider>
        </DesignSystemProvider>
      </IntlProvider>,
      {
        wrapper: ({ children }) => (
          <TestRouter routes={[testRoute(<>{children}</>, '/')]} history={history} initialEntries={[entry]} />
        ),
      },
    );

  beforeEach(() => {
    jest.clearAllMocks();
    mockExperiment([]);
  });

  test('sharedViewActive is true only when a share key is present in the URL', () => {
    let state: any;
    renderProbeAt('/?q=x', (s) => (state = s));
    expect(state.sharedViewActive).toBe(false);

    renderProbeAt(`/?${TRACE_V4_SHARE_URL_PARAM_KEY}=v1&cols=tokens,cost`, (s) => (state = s));
    expect(state.sharedViewActive).toBe(true);
  });

  test('previewColumns decode from the cols URL param while a shared view is active', () => {
    let state: any;
    renderProbeAt(`/?${TRACE_V4_SHARE_URL_PARAM_KEY}=v1&cols=tokens,cost`, (s) => (state = s));
    expect(state.previewColumns).toEqual(['tokens', 'cost']);
  });

  test('override adopts the previewed columns into the user store and strips the preview params', async () => {
    let state: any;
    let search = '';
    renderProbeAt(`/?q=x&${TRACE_V4_SHARE_URL_PARAM_KEY}=v1&cols=tokens,cost`, (s, sp) => {
      state = s;
      search = sp;
    });
    await act(async () => {
      state.override();
    });
    // The previewed columns are persisted to the user's own column store...
    expect(setColumns).toHaveBeenCalledWith(['tokens', 'cost']);
    // ...and the preview params are cleared while the rest of the view (q) stays.
    await waitFor(() => {
      const url = new URLSearchParams(search);
      expect(url.get(TRACE_V4_SHARE_URL_PARAM_KEY)).toBeNull();
      expect(url.get('cols')).toBeNull();
      expect(url.get('q')).toBe('x');
    });
  });

  test('discard strips the preview params WITHOUT writing to the user column store', async () => {
    let state: any;
    let search = '';
    renderProbeAt(`/?q=x&${TRACE_V4_SHARE_URL_PARAM_KEY}=v1&cols=tokens,cost`, (s, sp) => {
      state = s;
      search = sp;
    });
    await act(async () => {
      state.discard();
    });
    expect(setColumns).not.toHaveBeenCalled();
    await waitFor(() => {
      const url = new URLSearchParams(search);
      expect(url.get(TRACE_V4_SHARE_URL_PARAM_KEY)).toBeNull();
      expect(url.get('cols')).toBeNull();
      expect(url.get('q')).toBe('x');
    });
  });

  test('previewColumns is undefined when the cols param resolves to no known columns', () => {
    let state: any;
    renderProbeAt(`/?${TRACE_V4_SHARE_URL_PARAM_KEY}=v1&cols=bogus1,bogus2`, (s) => (state = s));
    // A view saved against an older/foreign column set: nothing resolves, so the caller falls back
    // to the user's own columns rather than hiding every column.
    expect(state.previewColumns).toBeUndefined();
  });

  test('override on a filter-only view (no cols) exits preview WITHOUT touching the column store', async () => {
    let state: any;
    let search = '';
    renderProbeAt(`/?q=x&${TRACE_V4_SHARE_URL_PARAM_KEY}=v1`, (s, sp) => {
      state = s;
      search = sp;
    });
    await act(async () => {
      state.override();
    });
    // No cols to adopt → the user's own columns are left untouched, and preview still exits.
    expect(setColumns).not.toHaveBeenCalled();
    await waitFor(() => {
      const url = new URLSearchParams(search);
      expect(url.get(TRACE_V4_SHARE_URL_PARAM_KEY)).toBeNull();
      expect(url.get('q')).toBe('x');
    });
  });

  test('setPreviewColumns edits the cols param WITHOUT writing the user column store', async () => {
    let state: any;
    let search = '';
    renderProbeAt(`/?q=x&${TRACE_V4_SHARE_URL_PARAM_KEY}=v1&cols=tokens,cost`, (s, sp) => {
      state = s;
      search = sp;
    });
    await act(async () => {
      state.setPreviewColumns(['tokens', 'cost', 'duration']);
    });
    // The preview (cols param) changes; localStorage is untouched (that only happens on Override).
    expect(setColumns).not.toHaveBeenCalled();
    await waitFor(() => {
      const url = new URLSearchParams(search);
      expect(url.get('cols')).toBe('tokens,cost,duration');
      expect(url.get(TRACE_V4_SHARE_URL_PARAM_KEY)).toBe('v1'); // still previewing
    });
  });
});

describe('useTracesV4SavedViews stale-tag refetch on active share key', () => {
  const HookProbe = ({ experimentId }: { experimentId: string }) => {
    useTracesV4SavedViews({ experimentId, visibleColumns: ['start_time'], setColumns: jest.fn() });
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
            initialEntries={[shareKey ? `/?${TRACE_V4_SHARE_URL_PARAM_KEY}=${shareKey}` : '/']}
          />
        ),
      },
    );

  beforeEach(async () => {
    jest.clearAllMocks();
    mockExperiment([await makeV4ViewTag('v1', 'Known', 1000)]);
  });

  test('refetches once when the active share key is not in the cached views', async () => {
    renderHookAt('freshly-saved-id');
    await waitFor(() => expect(stableRefetch).toHaveBeenCalledTimes(1));
  });

  test('does not refetch when the active share key is already cached', async () => {
    renderHookAt('v1');
    expect(stableRefetch).not.toHaveBeenCalled();
  });

  test('does not refetch when there is no active share key', async () => {
    renderHookAt();
    expect(stableRefetch).not.toHaveBeenCalled();
  });

  test('does not refetch again while the key stays missing after a tags update (anti-loop guard)', async () => {
    const { rerender } = renderHookAt('freshly-saved-id');
    await waitFor(() => expect(stableRefetch).toHaveBeenCalledTimes(1));

    // The refetch resolves with a NEW tags array that STILL lacks the key (e.g. a genuinely deleted
    // view): `views` changes identity → the effect re-runs → the per-key guard must hold and NOT fire
    // a second refetch, or a missing key would loop forever.
    mockExperiment([await makeV4ViewTag('v1', 'Known', 1000), await makeV4ViewTag('v2', 'Another', 2000)]);
    rerender(
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <MockedReduxStoreProvider>
            <HookProbe experimentId="exp-1" />
          </MockedReduxStoreProvider>
        </DesignSystemProvider>
      </IntlProvider>,
    );
    await waitFor(() => expect(stableRefetch).toHaveBeenCalledTimes(1));
  });
});

describe('TracesV4SharedViewBanner', () => {
  const setColumns = jest.fn();
  const BannerHarness = ({ entry }: { entry: string }) => {
    const savedViews = useTracesV4SavedViews({ experimentId: 'exp-1', visibleColumns: ['start_time'], setColumns });
    return <TracesV4SharedViewBanner savedViews={savedViews} />;
  };
  const renderBannerAt = (entry: string) =>
    render(
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <MockedReduxStoreProvider>
            <BannerHarness entry={entry} />
          </MockedReduxStoreProvider>
        </DesignSystemProvider>
      </IntlProvider>,
      {
        wrapper: ({ children }) => (
          <TestRouter routes={[testRoute(<>{children}</>, '/')]} history={history} initialEntries={[entry]} />
        ),
      },
    );

  beforeEach(async () => {
    jest.clearAllMocks();
    mockExperiment([await makeV4ViewTag('v1', 'Shared latency view', 1000)]);
  });

  test('renders nothing when no shared view is active', () => {
    const { container } = renderBannerAt('/?q=x');
    expect(container).toBeEmptyDOMElement();
  });

  test('announces the active shared view and offers Override / Discard', () => {
    renderBannerAt(`/?${TRACE_V4_SHARE_URL_PARAM_KEY}=v1&cols=tokens`);
    expect(screen.getByTestId('trace-v4-shared-view-banner')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Override my view/ })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Discard/ })).toBeInTheDocument();
  });
});
