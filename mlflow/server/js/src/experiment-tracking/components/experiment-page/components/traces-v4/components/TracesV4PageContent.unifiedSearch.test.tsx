import { beforeEach, describe, expect, jest, test } from '@jest/globals';
import { fireEvent, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { shouldEnableNewEvalRunsUi } from '@mlflow/mlflow/src/common/utils/FeatureUtils';
import { makeFeedbackAssessment, makeTrace } from '../test-utils/mockTraces';
import { renderPage, findTraceRow, state, env, URL } from '../test-utils/tracesV4PageContentTestBed';

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

const SEARCH_PLACEHOLDER = 'Search traces by id, input, or output';
const searchBox = () => screen.getByRole('combobox', { name: 'Type something to use full text search' });
// Typing key by key re-renders the whole table per key, so set the query in one change.
const typeQuery = (value: string) => fireEvent.change(searchBox(), { target: { value } });
// rc-select reads the legacy keyCode for Enter.
const pressEnter = () => fireEvent.keyDown(searchBox(), { key: 'Enter', code: 'Enter', keyCode: 13, which: 13 });
const lastFilter = () => state.searchCalls[state.searchCalls.length - 1]?.filter ?? '';

describe('TracesV4PageContent unified search', () => {
  beforeEach(() => {
    state.pages = {
      '': {
        traces: [makeTrace('tr-000', { assessments: [makeFeedbackAssessment('quality', 'pass')] })],
        next_page_token: undefined,
      },
    };
  });

  test('with the new eval runs UI off, keeps the plain search box and the Filters button', async () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(false);
    renderPage();

    await findTraceRow('tr-000');
    expect(screen.getByPlaceholderText(SEARCH_PLACEHOLDER)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Filters' })).toBeInTheDocument();
    expect(screen.queryByPlaceholderText('Add a filter')).not.toBeInTheDocument();
  }, 20000);

  test('with the new eval runs UI on, turns a typed query into a filter pill', async () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
    const user = userEvent.setup();
    renderPage();

    await findTraceRow('tr-000');
    expect(screen.queryByPlaceholderText(SEARCH_PLACEHOLDER)).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Filters' })).not.toBeInTheDocument();
    expect(searchBox()).toHaveAttribute('placeholder', 'Add a filter');

    typeQuery("feedback.quality = 'pass'");
    pressEnter();

    await waitFor(() => expect(lastFilter()).toContain("feedback.`quality` = 'pass'"));
    expect(searchBox()).toHaveValue('');
    const remove = await screen.findByRole('button', { name: "Remove filter feedback.quality = 'pass'" });

    // The unfiltered page is served from the query cache, so check the pills rather than a new request.
    await user.click(remove);
    await waitFor(() => expect(screen.queryByRole('button', { name: /Remove filter/ })).not.toBeInTheDocument());
  }, 20000);

  test('shows click-to-filter tags as pills and keeps them in the URL until their own pill is removed', async () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
    const user = userEvent.setup();
    renderPage({ initialUrl: `${URL}?tag=${encodeURIComponent('env=prod')}` });

    await findTraceRow('tr-000');
    expect(screen.getByRole('button', { name: "Edit filter tag.env = 'prod'" })).toBeInTheDocument();

    typeQuery("tag.team = 'ml'");
    pressEnter();
    await waitFor(() => expect(lastFilter()).toContain("tags.team = 'ml'"));
    expect(lastFilter()).toContain("tags.env = 'prod'");

    // Removing another pill leaves the URL tag in the URL, so a reload or a shared link keeps it.
    await user.click(screen.getByRole('button', { name: "Remove filter tag.team = 'ml'" }));
    await waitFor(() =>
      expect(screen.queryByRole('button', { name: "Edit filter tag.team = 'ml'" })).not.toBeInTheDocument(),
    );
    expect(new URLSearchParams(env.lastSearch).get('tag')).toBe('env=prod');
    expect(screen.getByRole('button', { name: "Edit filter tag.env = 'prod'" })).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: "Remove filter tag.env = 'prod'" }));
    await waitFor(() => expect(new URLSearchParams(env.lastSearch).get('tag')).toBeNull());
    expect(screen.queryByRole('button', { name: "Edit filter tag.env = 'prod'" })).not.toBeInTheDocument();
  }, 40000);

  test('with the new eval runs UI on, plain text still runs a text search', async () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
    renderPage();

    await findTraceRow('tr-000');
    typeQuery('hello');
    pressEnter();

    await waitFor(() => expect(new URLSearchParams(env.lastSearch).get('q')).toBe('hello'));
  }, 20000);
});
