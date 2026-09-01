import { jest, describe, beforeEach, test, expect } from '@jest/globals';
import { render, screen, waitFor, fireEvent, act } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import userEvent from '@testing-library/user-event';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';

import { TracesV4SavedViewsButton, useTracesV4SavedViews } from './TracesV4SavedViews';
import { MockedReduxStoreProvider } from '@mlflow/mlflow/src/common/utils/TestUtils';
import { setupTestRouter, testRoute, TestRouter } from '@mlflow/mlflow/src/common/utils/RoutingTestUtils';
import { useSearchParams } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import { useGetExperimentQuery } from '@mlflow/mlflow/src/experiment-tracking/hooks/useExperimentQuery';
import { encodeSavedViewEnvelope } from '../../../utils/savedViewEnvelope';
import { textCompressDeflate, textDecompressDeflate } from '@mlflow/mlflow/src/common/utils/StringUtils';
import Utils from '@mlflow/mlflow/src/common/utils/Utils';
import { TRACE_V4_SHARE_URL_PARAM_KEY, buildV4ViewQuery, captureV4ViewState } from '../utils/tracesV4SavedViewState';
import { FilterOp, type TraceColumnId, type TraceFilterModel } from '@databricks/web-shared/traces-table';

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
  filters: TraceFilterModel = [],
  assessments: Record<string, boolean> = {},
  custom: Record<string, boolean> = {},
) => {
  const state = captureV4ViewState(new URLSearchParams(query), cols, filters, assessments, custom);
  const compressed = await textCompressDeflate(JSON.stringify(state));
  return { key: `mlflow.tracesV4ViewState.${id}`, value: encodeSavedViewEnvelope(name, compressed, createdAt) };
};

// Build a LEGACY V3 saved-view tag (prefix `mlflow.traceViewState.`) whose compressed state is the
// frozen V3 shape, in V3's OWN vocabulary: `single.selectedColumns` is a comma-joined list of V3
// column ids (`request_time`, `request`, `execution_duration`, …), `single.sort` is `key::type::asc`
// with a V3 column key, time range, and `multi.filter` holds `column::operator::value::key` entries.
// Same envelope codec as V4; only the inner state differs.
const makeV3ViewTag = async (
  id: string,
  name: string,
  createdAt: number,
  single: Record<string, string> = { selectedColumns: 'request_time', sort: 'request_time::date::false' },
  filter: string[] = [],
) => {
  const state = { single, multi: filter.length > 0 ? { filter } : {} };
  const compressed = await textCompressDeflate(JSON.stringify(state));
  return { key: `mlflow.traceViewState.${id}`, value: encodeSavedViewEnvelope(name, compressed, createdAt) };
};

const stableRefetch = jest.fn(() => Promise.resolve({}));
const mockExperiment = (tags: { key: string; value: string }[]) => {
  jest.mocked(useGetExperimentQuery).mockReturnValue({ data: { tags }, refetch: stableRefetch } as never);
};

const { history } = setupTestRouter();

// Shared across the button harness so tests can assert column restore on open / reset.
const buttonSetColumns = jest.fn();
const buttonSetCustomVisibility = jest.fn();
const SavedViewsButtonHarness = ({ experimentId }: { experimentId: string }) => {
  const savedViews = useTracesV4SavedViews({
    experimentId,
    visibleColumns: ['start_time', 'input'],
    filterModel: [],
    setColumns: buttonSetColumns,
    resetColumns: jest.fn(),
    setFilterModel: jest.fn(),
    customVisibility: {},
    setCustomVisibility: buttonSetCustomVisibility,
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
    expect(decoded.single.cols).toBe('start_time,input'); // the harness's live columns, stored not URL'd
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

  test('opening a view rewrites the URL to the stored query + share key and restores its columns', async () => {
    // v1 was stored with q=x and cols=[start_time] (the harness default). Columns are no longer
    // carried in the URL — they are restored into the user's column store via setColumns.
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
      // Columns ride in the store, not the URL.
      expect(out.get('cols')).toBeNull();
    });
    expect(buttonSetColumns).toHaveBeenCalledWith(['start_time']);
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

  test('landing on a direct link to a corrupt view degrades gracefully (no crash, view not marked active)', async () => {
    // A share link opened cold: the view tag exists but its state blob is undecodable. The URL params
    // still drive the table, so the cold-load hydration just no-ops — no crash, and the button shows
    // the view name from the (readable) envelope without hydrating columns or flagging a dirty view.
    mockExperiment([
      { key: 'mlflow.tracesV4ViewState.bad', value: encodeSavedViewEnvelope('Broken', 'not-base64', 5) },
    ]);
    renderButtonAt(`/?q=x&${TRACE_V4_SHARE_URL_PARAM_KEY}=bad`);
    await waitFor(() => expect(screen.getByTestId('trace-v4-saved-views-trigger')).toHaveTextContent('Broken'));
    // Undecodable state → no columns restored and no dirty dot (stays clean rather than half-applied).
    expect(buttonSetColumns).not.toHaveBeenCalled();
    expect(screen.queryByTestId('trace-v4-saved-views-dirty-dot')).not.toBeInTheDocument();
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
    expect(decoded.single.cols).toBe('start_time,input'); // the harness's live columns, stored not URL'd
  });
});

describe('useTracesV4SavedViews dirty / overwrite / reset', () => {
  const setColumns = jest.fn();
  // Report BOTH the hook API and the live URL search each render, so assertions read the in-memory
  // router's state (TestRouter never touches window.location.hash) after a param rewrite.
  const setFilterModel = jest.fn();
  const setAssessmentVisibility = jest.fn();
  const DirtyProbe = ({
    onRender,
    filterModel = [],
    assessmentVisibility = {},
  }: {
    onRender: (s: any, search: string) => void;
    filterModel?: TraceFilterModel;
    assessmentVisibility?: Record<string, boolean>;
  }) => {
    const savedViews = useTracesV4SavedViews({
      experimentId: 'exp-1',
      visibleColumns: ['start_time'],
      filterModel,
      setColumns,
      resetColumns: jest.fn(),
      setFilterModel,
      assessmentVisibility,
      setAssessmentVisibility,
    });
    const [params] = useSearchParams();
    onRender(savedViews, params.toString());
    return null;
  };

  const renderProbeAt = (
    entry: string,
    onRender: (s: any, search: string) => void,
    filterModel: TraceFilterModel = [],
    assessmentVisibility: Record<string, boolean> = {},
  ) =>
    render(
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <MockedReduxStoreProvider>
            <DirtyProbe onRender={onRender} filterModel={filterModel} assessmentVisibility={assessmentVisibility} />
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
    // v1 is stored with q=x and cols=[start_time], matching the probe's live state below.
    mockExperiment([await makeV4ViewTag('v1', 'Known', 1000, 'q=x', ['start_time'])]);
  });

  test('activeViewId is set only when the share key resolves to a known view', () => {
    let state: any;
    renderProbeAt('/?q=x', (s) => (state = s));
    expect(state.activeViewId).toBeNull();

    renderProbeAt(`/?q=x&${TRACE_V4_SHARE_URL_PARAM_KEY}=nope`, (s) => (state = s));
    // A share key that isn't in the list must not drive overwrite/reset/dirty.
    expect(state.activeViewId).toBeNull();

    renderProbeAt(`/?q=x&${TRACE_V4_SHARE_URL_PARAM_KEY}=v1`, (s) => (state = s));
    expect(state.activeViewId).toBe('v1');
  });

  test('dirtyStatus is clean when the live state matches the active view', async () => {
    let state: any;
    // Live URL (q=x) + live columns ([start_time]) equal v1's stored state.
    renderProbeAt(`/?q=x&${TRACE_V4_SHARE_URL_PARAM_KEY}=v1`, (s) => (state = s));
    await waitFor(() => expect(state.dirtyStatus).toBe('clean'));
  });

  test('dirtyStatus becomes dirty when the live URL diverges from the active view', async () => {
    let state: any;
    // Live URL (q=changed) differs from v1's stored q=x.
    renderProbeAt(`/?q=changed&${TRACE_V4_SHARE_URL_PARAM_KEY}=v1`, (s) => (state = s));
    await waitFor(() => expect(state.dirtyStatus).toBe('dirty'));
  });

  test('overwriteView rewrites the active view tag in place, keeping id + createdAt and bumping updatedAt', async () => {
    let state: any;
    renderProbeAt(`/?q=changed&${TRACE_V4_SHARE_URL_PARAM_KEY}=v1`, (s) => (state = s));
    await waitFor(() => expect(state.dirtyStatus).toBe('dirty'));
    await act(async () => {
      await state.overwriteView('v1');
    });
    expect(mockSetExperimentTagApi).toHaveBeenCalledTimes(1);
    const [expId, tagKey, tagValue] = mockSetExperimentTagApi.mock.calls[0];
    expect(expId).toBe('exp-1');
    expect(tagKey).toBe('mlflow.tracesV4ViewState.v1'); // same tag, in place
    const envelope = JSON.parse(tagValue);
    expect(envelope.name).toBe('Known'); // name preserved
    expect(envelope.createdAt).toBe(1000); // creation time preserved
    expect(envelope.updatedAt).toBeGreaterThan(1000); // edit time bumped
    // The rewritten state captures the edited URL.
    const decoded = JSON.parse(await textDecompressDeflate(envelope.state));
    expect(decoded.single.q).toBe('changed');
  });

  test('overwriteView refuses to resurrect a phantom (unknown) id', async () => {
    let state: any;
    renderProbeAt(`/?q=x&${TRACE_V4_SHARE_URL_PARAM_KEY}=v1`, (s) => (state = s));
    await act(async () => {
      await state.overwriteView('does-not-exist');
    });
    expect(mockSetExperimentTagApi).not.toHaveBeenCalled();
  });

  test('resetActiveView re-applies the stored view, restoring its columns and clearing the edit', async () => {
    let state: any;
    let search = '';
    renderProbeAt(`/?q=changed&${TRACE_V4_SHARE_URL_PARAM_KEY}=v1`, (s, sp) => {
      state = s;
      search = sp;
    });
    await waitFor(() => expect(state.dirtyStatus).toBe('dirty'));
    await act(async () => {
      state.resetActiveView();
    });
    // The stored query (q=x) is re-applied and the stored columns restored.
    await waitFor(() => {
      const url = new URLSearchParams(search);
      expect(url.get('q')).toBe('x');
      expect(url.get(TRACE_V4_SHARE_URL_PARAM_KEY)).toBe('v1');
    });
    expect(setColumns).toHaveBeenCalledWith(['start_time']);
  });

  test('opening a view restores its stored popover filter model', async () => {
    // v1 stored with a state filter; opening it must push that clause into the live filter model.
    mockExperiment([
      await makeV4ViewTag(
        'v1',
        'Known',
        1000,
        'q=x',
        ['start_time'],
        [{ field: 'state', operator: FilterOp.EQUALS, value: 'ERROR' }],
      ),
    ]);
    let state: any;
    renderProbeAt('/', (s) => (state = s));
    await act(async () => {
      await state.openView('v1');
    });
    expect(setFilterModel).toHaveBeenCalledWith([{ field: 'state', operator: FilterOp.EQUALS, value: 'ERROR' }]);
  });

  test('a filter-model divergence from the stored view reads as dirty', async () => {
    // v1 stored with NO filters; the probe's live filter model carries a clause → dirty.
    let state: any;
    renderProbeAt(`/?q=x&${TRACE_V4_SHARE_URL_PARAM_KEY}=v1`, (s) => (state = s), [
      { field: 'state', operator: FilterOp.EQUALS, value: 'ERROR' },
    ]);
    await waitFor(() => expect(state.dirtyStatus).toBe('dirty'));
  });

  test('overwriteView captures the live filter model into the rewritten tag', async () => {
    let state: any;
    renderProbeAt(`/?q=x&${TRACE_V4_SHARE_URL_PARAM_KEY}=v1`, (s) => (state = s), [
      { field: 'state', operator: FilterOp.EQUALS, value: 'ERROR' },
    ]);
    await waitFor(() => expect(state.dirtyStatus).toBe('dirty'));
    await act(async () => {
      await state.overwriteView('v1');
    });
    const [, , tagValue] = mockSetExperimentTagApi.mock.calls[0];
    const decoded = JSON.parse(await textDecompressDeflate(JSON.parse(tagValue).state));
    expect(decoded.filters).toEqual([{ field: 'state', operator: FilterOp.EQUALS, value: 'ERROR' }]);
  });

  test('an unsupported stored filter clause is dropped on restore, not stranding the view dirty', async () => {
    // A clause whose operator the `state` field never offers (CONTAINS) is invalid; opening must
    // drop it (restoring an empty model), and the dirty diff normalizes the baseline the same way so
    // the view still reads clean against a live empty filter model.
    mockExperiment([
      await makeV4ViewTag(
        'v1',
        'Known',
        1000,
        'q=x',
        ['start_time'],
        [{ field: 'state', operator: FilterOp.CONTAINS, value: 'ERROR' }],
      ),
    ]);
    let state: any;
    renderProbeAt(`/?q=x&${TRACE_V4_SHARE_URL_PARAM_KEY}=v1`, (s) => (state = s), []);
    await act(async () => {
      await state.openView('v1');
    });
    expect(setFilterModel).toHaveBeenLastCalledWith([]);
    await waitFor(() => expect(state.dirtyStatus).toBe('clean'));
  });

  test('opening a view restores its stored assessment-column visibility', async () => {
    // v1 stored with one assessment hidden; opening it must push the full map into the live store.
    mockExperiment([
      await makeV4ViewTag('v1', 'Known', 1000, 'q=x', ['start_time'], [], { correctness: true, relevance: false }),
    ]);
    let state: any;
    renderProbeAt('/', (s) => (state = s));
    await act(async () => {
      await state.openView('v1');
    });
    expect(setAssessmentVisibility).toHaveBeenCalledWith({ correctness: true, relevance: false });
  });

  test('an assessment-visibility divergence from the stored view reads as dirty', async () => {
    // v1 stored with correctness visible; the probe's live map hides it → dirty.
    mockExperiment([await makeV4ViewTag('v1', 'Known', 1000, 'q=x', ['start_time'], [], { correctness: true })]);
    let state: any;
    renderProbeAt(`/?q=x&${TRACE_V4_SHARE_URL_PARAM_KEY}=v1`, (s) => (state = s), [], { correctness: false });
    await waitFor(() => expect(state.dirtyStatus).toBe('dirty'));
  });
});

describe('useTracesV4SavedViews legacy V3 view compatibility', () => {
  const setColumns = jest.fn();
  const setFilterModel = jest.fn();
  const V3Probe = ({ onRender }: { onRender: (s: any, search: string) => void }) => {
    const savedViews = useTracesV4SavedViews({
      experimentId: 'exp-1',
      visibleColumns: ['start_time'],
      filterModel: [],
      setColumns,
      resetColumns: jest.fn(),
      setFilterModel,
    });
    const [params] = useSearchParams();
    onRender(savedViews, params.toString());
    return null;
  };
  const renderV3ProbeAt = (entry: string, onRender: (s: any, search: string) => void) =>
    render(
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <MockedReduxStoreProvider>
            <V3Probe onRender={onRender} />
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
  });

  test('lists a legacy V3 view alongside V4 views', async () => {
    mockExperiment([await makeV4ViewTag('v4a', 'Native V4', 2000), await makeV3ViewTag('v3a', 'Legacy V3', 1000)]);
    let state: any;
    renderV3ProbeAt('/', (s) => (state = s));
    await waitFor(() => expect(state.views).toHaveLength(2));
    expect(state.views.find((v: any) => v.id === 'v3a')).toMatchObject({ name: 'Legacy V3', origin: 'v3' });
    expect(state.views.find((v: any) => v.id === 'v4a')).toMatchObject({ name: 'Native V4', origin: 'v4' });
  });

  test('opening a V3 view translates its state onto the URL + restores columns/filters', async () => {
    mockExperiment([
      await makeV3ViewTag(
        'v3a',
        'Legacy V3',
        1000,
        // V3 vocabulary: request_time→start_time, request→input columns; execution_duration sort key;
        // a `column::operator::value::key` filter that maps onto V4's `state` popover field.
        {
          selectedColumns: 'request_time,request',
          sort: 'execution_duration::number::false',
          startTimeLabel: 'LAST_7_DAYS',
        },
        ['state::=::ERROR::'],
      ),
    ]);
    let state: any;
    let search = '';
    renderV3ProbeAt('/', (s, sp) => {
      state = s;
      search = sp;
    });
    await act(async () => {
      await state.openView('v3a');
    });
    await waitFor(() => {
      const url = new URLSearchParams(search);
      expect(url.get('sort')).toBe('duration'); // execution_duration → duration, `type` segment dropped
      expect(url.get('dir')).toBe('desc');
      expect(url.get('startTimeLabel')).toBe('LAST_7_DAYS');
      expect(url.get(TRACE_V4_SHARE_URL_PARAM_KEY)).toBe('v3a');
    });
    // Columns restored into the user's store (V3 ids → V4 ids: request_time→start_time, request→input).
    expect(setColumns).toHaveBeenCalledWith(['start_time', 'input']);
    // V3 filter[] → V4's in-memory popover model (not the URL-backed tag[]).
    expect(setFilterModel).toHaveBeenCalledWith([{ field: 'state', operator: '=', value: 'ERROR' }]);
  });

  test('overwriting a V3 view migrates it: writes a V4 tag (same id) and deletes the V3 tag', async () => {
    mockExperiment([await makeV3ViewTag('v3a', 'Legacy V3', 1000, { selectedColumns: 'start_time' })]);
    let state: any;
    renderV3ProbeAt(`/?q=edited&${TRACE_V4_SHARE_URL_PARAM_KEY}=v3a`, (s) => (state = s));
    await waitFor(() => expect(state.activeViewId).toBe('v3a'));
    await act(async () => {
      await state.overwriteView('v3a');
    });
    // V4 tag written under the same id, name + createdAt preserved, with the edited live state.
    expect(mockSetExperimentTagApi).toHaveBeenCalledTimes(1);
    const [, setKey, setValue] = mockSetExperimentTagApi.mock.calls[0];
    expect(setKey).toBe('mlflow.tracesV4ViewState.v3a');
    const envelope = JSON.parse(setValue);
    expect(envelope.name).toBe('Legacy V3');
    expect(envelope.createdAt).toBe(1000);
    const decoded = JSON.parse(await textDecompressDeflate(envelope.state));
    expect(decoded.single.q).toBe('edited');
    // Legacy V3 tag deleted so the view is native V4 afterwards.
    expect(mockDeleteExperimentTagApi).toHaveBeenCalledWith('exp-1', 'mlflow.traceViewState.v3a');
  });

  test('deleting a V3 view targets the V3 tag prefix', async () => {
    mockExperiment([await makeV3ViewTag('v3a', 'Legacy V3', 1000)]);
    let state: any;
    renderV3ProbeAt('/', (s) => (state = s));
    await waitFor(() => expect(state.views).toHaveLength(1));
    await act(async () => {
      await state.deleteView('v3a');
    });
    expect(mockDeleteExperimentTagApi).toHaveBeenCalledTimes(1);
    expect(mockDeleteExperimentTagApi).toHaveBeenCalledWith('exp-1', 'mlflow.traceViewState.v3a');
  });

  test('deleting a half-migrated view removes BOTH the V4 and leftover V3 tags (no resurrection)', async () => {
    // A migration whose V3-delete never landed leaves both prefixes for the id; the list de-dupes to
    // one V4 entry, so deleting only the V4 tag would let the V3 twin come back on the next refetch.
    mockExperiment([
      await makeV3ViewTag('dup', 'Legacy name', 1000),
      await makeV4ViewTag('dup', 'Migrated name', 3000),
    ]);
    let state: any;
    renderV3ProbeAt('/', (s) => (state = s));
    await waitFor(() => expect(state.views).toHaveLength(1));
    await act(async () => {
      await state.deleteView('dup');
    });
    expect(mockDeleteExperimentTagApi).toHaveBeenCalledWith('exp-1', 'mlflow.tracesV4ViewState.dup');
    expect(mockDeleteExperimentTagApi).toHaveBeenCalledWith('exp-1', 'mlflow.traceViewState.dup');
    expect(mockDeleteExperimentTagApi).toHaveBeenCalledTimes(2);
  });

  test('a migrated V4 tag shadows a leftover V3 tag of the same id (V4 wins, listed once)', async () => {
    // Both prefixes hold the id (e.g. a refetch race mid-migration); the list must show one V4 entry.
    mockExperiment([
      await makeV3ViewTag('dup', 'Legacy name', 1000),
      await makeV4ViewTag('dup', 'Migrated name', 3000),
    ]);
    let state: any;
    renderV3ProbeAt('/', (s) => (state = s));
    await waitFor(() => expect(state.views).toHaveLength(1));
    expect(state.views[0]).toMatchObject({ id: 'dup', name: 'Migrated name', origin: 'v4' });
  });
});

describe('useTracesV4SavedViews stale-tag refetch on active share key', () => {
  const HookProbe = ({ experimentId }: { experimentId: string }) => {
    useTracesV4SavedViews({
      experimentId,
      visibleColumns: ['start_time'],
      filterModel: [],
      setColumns: jest.fn(),
      resetColumns: jest.fn(),
      setFilterModel: jest.fn(),
    });
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

describe('TracesV4SavedViewsButton dirty affordances', () => {
  beforeEach(async () => {
    jest.clearAllMocks();
    mockExperiment([await makeV4ViewTag('v1', 'Latency triage', 1000, 'q=x', ['start_time'])]);
  });

  test('no dirty dot and no Overwrite/Reset actions when the active view is clean', async () => {
    // Live state (q=x, cols=[start_time,input]) — cols differ from the stored [start_time], but the
    // trigger only shows the dot when dirty; here we land on the clean case via matching state.
    renderButtonAt(`/?q=x&${TRACE_V4_SHARE_URL_PARAM_KEY}=v1`);
    // The harness's live columns are [start_time, input] while v1 stored [start_time] — that IS a
    // divergence, so this view is dirty. Assert the dirty affordances appear instead.
    await waitFor(() => expect(screen.getByTestId('trace-v4-saved-views-dirty-dot')).toBeInTheDocument());
    await openDropdown();
    expect(screen.getByTestId('trace-v4-saved-views-overwrite-active')).toBeInTheDocument();
    expect(screen.getByTestId('trace-v4-saved-views-reset-active')).toBeInTheDocument();
  });

  test('a clean active view shows neither the dot nor the dirty actions', async () => {
    // Store v1 with cols matching the harness's live columns so it is genuinely clean.
    mockExperiment([await makeV4ViewTag('v1', 'Latency triage', 1000, 'q=x', ['start_time', 'input'])]);
    renderButtonAt(`/?q=x&${TRACE_V4_SHARE_URL_PARAM_KEY}=v1`);
    // Give the stored-state effect a chance to run, then confirm it stays clean.
    await waitFor(() => expect(screen.getByTestId('trace-v4-saved-views-trigger')).toHaveTextContent('Latency triage'));
    expect(screen.queryByTestId('trace-v4-saved-views-dirty-dot')).not.toBeInTheDocument();
    await openDropdown();
    expect(screen.queryByTestId('trace-v4-saved-views-overwrite-active')).not.toBeInTheDocument();
    expect(screen.queryByTestId('trace-v4-saved-views-reset-active')).not.toBeInTheDocument();
  });

  test('the Overwrite action names the active view and dispatches an in-place tag write', async () => {
    renderButtonAt(`/?q=changed&${TRACE_V4_SHARE_URL_PARAM_KEY}=v1`);
    await waitFor(() => expect(screen.getByTestId('trace-v4-saved-views-dirty-dot')).toBeInTheDocument());
    await openDropdown();
    const overwrite = screen.getByTestId('trace-v4-saved-views-overwrite-active');
    expect(overwrite).toHaveTextContent('Overwrite "Latency triage"');
    await userEvent.click(overwrite, { pointerEventsCheck: 0 });
    await waitFor(() => expect(mockSetExperimentTagApi).toHaveBeenCalled());
    expect(mockSetExperimentTagApi.mock.calls[0][1]).toBe('mlflow.tracesV4ViewState.v1');
  });

  test('the dirty dot renders outside the collapsible label so it survives icon-only collapse', async () => {
    // ToolbarCollapsibleLabel sets `display:none` under a container query when the toolbar is narrow.
    // jsdom can't evaluate the query, so assert structurally: the dot must not be a descendant of the
    // label span (its `maxWidth:200` name span identifies it), i.e. it collapses independently.
    renderButtonAt(`/?q=changed&${TRACE_V4_SHARE_URL_PARAM_KEY}=v1`);
    const dot = await screen.findByTestId('trace-v4-saved-views-dirty-dot');
    const label = screen.getByText('Latency triage');
    expect(label).not.toContainElement(dot);
  });
});
