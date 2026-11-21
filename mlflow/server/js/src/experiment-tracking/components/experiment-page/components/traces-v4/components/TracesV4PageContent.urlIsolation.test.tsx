import { afterEach, describe, expect, jest, test } from '@jest/globals';
import { fireEvent, screen, waitFor, within } from '@testing-library/react';
import { rest } from 'msw';
import { useNavigate, useSearchParams } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import type { ModelTraceSpan } from '@databricks/web-shared/model-trace-explorer';
import { TracesV4PageContent } from './TracesV4PageContent';
import * as controllerModule from '../hooks/useTracesV4Controller';
import * as savedViewsModule from './TracesV4SavedViews';
import * as tableModule from './TracesV4TableSurface';
import { renderTracesV4Page } from '../test-utils/renderTracesV4Page';
import { EXPERIMENT_ID, URL, history, findTraceRow, state, server } from '../test-utils/tracesV4PageContentTestBed';
import { makeTaggedTrace } from '../test-utils/mockTraces';

const Navigation = () => {
  const [params, setParams] = useSearchParams();
  const navigate = useNavigate();
  return (
    <>
      <output data-testid="selected-span">{params.get('spanId')}</output>
      <output data-testid="query">{params.toString()}</output>
      <button onClick={() => navigate(-1)}>Back</button>
      <button onClick={() => navigate(1)}>Forward</button>
      <button
        onClick={() =>
          setParams((current) => {
            current.set('traceId', 'trace:/cat.sch/tr-000');
            current.set('spanId', 'child');
            return current;
          })
        }
      >
        Open linked child
      </button>
      <button
        onClick={() =>
          setParams(
            (current) => {
              current.set('latest', 'preserved');
              return current;
            },
            { replace: true },
          )
        }
      >
        Update latest parameter
      </button>
      <button
        onClick={() =>
          setParams((current) => {
            current.set('sort', 'duration');
            return current;
          })
        }
      >
        Change table sort
      </button>
      <button
        onClick={() =>
          setParams(
            (params) => {
              params.set('spanId', 'child');
              return params;
            },
            { replace: true },
          )
        }
      >
        Select child URL
      </button>
    </>
  );
};

const mountPage = (initialUrl = URL) =>
  renderTracesV4Page({
    initialUrl,
    routes: [
      {
        path: '*',
        element: (
          <>
            <Navigation />
            <TracesV4PageContent experimentId={EXPERIMENT_ID} />
          </>
        ),
      },
    ],
    history,
    dataRouter: true,
  });

afterEach(() => {
  jest.restoreAllMocks();
  server.resetHandlers();
});

describe('trace URL subscriptions', () => {
  test('span-only navigation does not rerender the page or saved views', async () => {
    const controller = jest.spyOn(controllerModule, 'useTracesV4Controller');
    const savedViews = jest.spyOn(savedViewsModule, 'useTracesV4SavedViews');
    const table = jest.spyOn(tableModule, 'TracesV4TableSurface');
    mountPage();
    await findTraceRow('tr-000');
    await waitFor(() => expect(state.searchCalls.length).toBeGreaterThan(0));
    const pageRenders = controller.mock.calls.length;
    const savedViewRenders = savedViews.mock.calls.length;
    const tableRenders = table.mock.calls.length;
    fireEvent.click(screen.getByText('Select child URL'));
    expect(screen.getByTestId('selected-span')).toHaveTextContent('child');
    expect({
      page: controller.mock.calls.length - pageRenders,
      savedViews: savedViews.mock.calls.length - savedViewRenders,
      table: table.mock.calls.length - tableRenders,
    }).toEqual({ page: 0, savedViews: 0, table: 0 });
    fireEvent.click(screen.getByText('Change table sort'));
    await waitFor(() => {
      expect(controller.mock.calls.length).toBeGreaterThan(pageRenders);
      expect(savedViews.mock.calls.length).toBeGreaterThan(savedViewRenders);
      expect(table.mock.calls.length).toBeGreaterThan(tableRenders);
    });
  });

  test('experiment drawer span clicks replace the URL and preserve table parameters through history', async () => {
    const trace = makeTaggedTrace('tr-000', { env: 'prod' });
    state.pages = { '': { traces: [trace] } };
    const spans: ModelTraceSpan[] = ['root', 'child'].map((name, index) => ({
      name,
      context: { span_id: name, trace_id: trace.trace_id },
      parent_id: index ? 'root' : null,
      start_time: index * 1_000_000,
      end_time: 10_000_000,
      status: { status_code: 1, description: 'OK' },
      attributes: {
        'mlflow.spanType': JSON.stringify('CHAIN'),
        'mlflow.spanOutputs': JSON.stringify(`${name}-only-output`),
      },
      events: [],
    }));
    server.use(
      rest.get('/ajax-api/3.0/mlflow/traces/:traceId', (_req, res, ctx) =>
        res(ctx.json({ trace: { trace_info: trace, spans } })),
      ),
    );
    mountPage(`${URL}?startTimeLabel=ALL&tag=env%3Dprod&filters=%5B%5D`);
    await findTraceRow('tr-000');
    fireEvent.click(screen.getByText('Open linked child'));
    const drawer = await screen.findByRole('dialog', {}, { timeout: 10000 });
    await within(drawer).findByText('child-only-output', { exact: false }, { timeout: 10000 });
    fireEvent.click(screen.getByText('Update latest parameter'));
    fireEvent.click(within(drawer).getByTestId('timeline-tree-node-root'));
    await waitFor(() => expect(screen.getByTestId('selected-span')).toHaveTextContent('root'));
    const params = new URLSearchParams(screen.getByTestId('query').textContent ?? '');
    expect(params.get('startTimeLabel')).toBe('ALL');
    expect(params.getAll('tag')).toEqual(['env=prod']);
    expect(params.get('filters')).toBe('[]');
    expect(params.get('latest')).toBe('preserved');
    fireEvent.click(screen.getByText('Back', { selector: 'button' }));
    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
    fireEvent.click(screen.getByText('Forward', { selector: 'button' }));
    await waitFor(() => expect(screen.getByTestId('selected-span')).toHaveTextContent('root'));
    expect(await screen.findByText('root-only-output', { exact: false }, { timeout: 10000 })).toBeVisible();
  }, 15000);
});
