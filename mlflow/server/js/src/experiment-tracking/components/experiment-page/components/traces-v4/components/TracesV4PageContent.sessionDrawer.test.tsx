import { beforeAll, beforeEach, describe, expect, jest, test } from '@jest/globals';
import { fireEvent, screen, waitFor, within } from '@testing-library/react';
import { shouldEnableNewEvalRunsUi } from '@mlflow/mlflow/src/common/utils/FeatureUtils';
import { makeSessionTrace } from '../test-utils/mockTraces';
import { renderPage, findTraceRow, state, env, URL as PAGE_URL } from '../test-utils/tracesV4PageContentTestBed';

jest.mock('@mlflow/mlflow/src/common/utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('@mlflow/mlflow/src/common/utils/FeatureUtils')>(
    '@mlflow/mlflow/src/common/utils/FeatureUtils',
  ),
  shouldEnableNewEvalRunsUi: jest.fn(() => false),
}));

// Saved views read experiment tags through Apollo; stub it as the other page tests do.
jest.mock('@mlflow/mlflow/src/experiment-tracking/hooks/useExperimentQuery', () => ({
  useGetExperimentQuery: () => ({ data: { tags: [] }, refetch: () => Promise.resolve({}) }),
}));

const GROUPED_URL = `${PAGE_URL}?groupBy=session&q=hello`;
const params = () => new URLSearchParams(env.lastSearch);
// A grouped session's header row, found through its session id cell.
const findSessionRow = async (sessionId: string) =>
  (await screen.findByText(sessionId, {}, { timeout: 15000 })).closest('[role="row"]') as HTMLElement;

describe('TracesV4PageContent sessions in the trace drawer', () => {
  beforeAll(() => {
    // jsdom has no element scrolling; the session drawer scrolls its selected turn into view.
    Element.prototype.scrollTo = jest.fn() as unknown as Element['scrollTo'];
  });

  beforeEach(() => {
    state.pages = {
      '': {
        traces: [makeSessionTrace('tr-000', 'sess-1'), makeSessionTrace('tr-001', 'sess-1')],
        next_page_token: undefined,
      },
    };
  });

  test('with the new eval runs UI on, a session row opens the session in the drawer and keeps the table state', async () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
    renderPage({ initialUrl: GROUPED_URL });

    fireEvent.click(await findSessionRow('sess-1'));

    await waitFor(() => expect(params().get('traceDrawerView')).toBe('session'));
    expect(params().get('traceDrawerSessionId')).toBe('sess-1');
    expect(params().get('traceId')).toBe('trace:/cat.sch/tr-000');
    expect(params().get('groupBy')).toBe('session');
    expect(params().get('q')).toBe('hello');
    // The table stays on the page behind the drawer.
    expect(screen.getAllByText('sess-1').length).toBeGreaterThan(0);
    const drawer = await screen.findByRole('dialog', {}, { timeout: 10000 });
    expect(within(drawer).getAllByText('Session').length).toBeGreaterThan(0);
  }, 30000);

  test('with the new eval runs UI on, the session link carries the table state to the session route', async () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
    state.pages = { '': { traces: [makeSessionTrace('tr-000', 'sess-1')], next_page_token: undefined } };
    renderPage({ initialUrl: `${PAGE_URL}?q=hello` });
    await findTraceRow('tr-000');

    const href = new URL(screen.getByRole('link', { name: 'sess-1' }).getAttribute('href') ?? '', 'http://localhost');
    expect(href.pathname).toContain('/experiments/exp-1/chat-sessions/sess-1');
    expect(href.searchParams.get('q')).toBe('hello');
    expect(href.searchParams.get('selectedTraceId')).toBe('tr-000');
  }, 30000);

  test('with the new eval runs UI off, a session row still leaves for the chat session page', async () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(false);
    renderPage({ initialUrl: GROUPED_URL });

    fireEvent.click(await findSessionRow('sess-1'));

    // The Traces route unmounts as the page moves to the chat session route, without drawer state.
    await waitFor(() => expect(screen.queryByText('sess-1')).not.toBeInTheDocument());
    expect(params().get('traceDrawerView')).toBeNull();
  }, 30000);
});
