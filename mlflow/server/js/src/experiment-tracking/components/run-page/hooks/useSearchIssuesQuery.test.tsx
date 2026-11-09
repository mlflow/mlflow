import { jest, describe, beforeEach, expect, test } from '@jest/globals';

import { fetchAPI } from '@mlflow/mlflow/src/common/utils/FetchUtils';

import { searchIssueIdsByStatus } from './useSearchIssuesQuery';

jest.mock('@mlflow/mlflow/src/common/utils/FetchUtils', () => ({
  fetchAPI: jest.fn(),
  getAjaxUrl: (url: string) => url,
}));

const mockFetchAPI = jest.mocked(fetchAPI);

/** One page of `size` issues, named after the page so the assertions can tell them apart. */
const page = (name: string, size: number, nextPageToken: string) => ({
  issues: Array.from({ length: size }, (_, index) => ({ issue_id: `${name}-${index}` })),
  next_page_token: nextPageToken,
});

/** Serves the given pages in order, as the search endpoint would. */
const givenPages = (pages: ReturnType<typeof page>[]) => {
  let next = 0;
  mockFetchAPI.mockImplementation(() => Promise.resolve(pages[next++]) as ReturnType<typeof fetchAPI>);
};

describe('searchIssueIdsByStatus', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('asks for one status and only the fields it reads', async () => {
    givenPages([page('only', 2, '')]);

    expect(await searchIssueIdsByStatus('experiment-1', 'rejected')).toEqual(['only-0', 'only-1']);
    expect(mockFetchAPI).toHaveBeenCalledTimes(1);
    expect(mockFetchAPI.mock.calls[0][1]).toMatchObject({
      body: {
        experiment_id: 'experiment-1',
        statuses: ['rejected'],
        include_trace_count: false,
        view: 'basic',
        // The Databricks issues backend rejects anything larger.
        max_results: 100,
      },
    });
    // No token to send on the first page.
    expect(mockFetchAPI.mock.calls[0][1]).not.toHaveProperty('body.page_token');
  });

  test('drains every page, past the five an earlier revision stopped at', async () => {
    // An ID left out of the dismissed set reads as not dismissed, so a short read silently puts
    // dismissed issues back on screen -- the bug this lookup exists to fix.
    const sixPages = [1, 2, 3, 4, 5, 6].map((number) =>
      page(`page${number}`, 1, number === 6 ? '' : `token-${number}`),
    );
    givenPages(sixPages);

    expect(await searchIssueIdsByStatus('experiment-1', 'rejected')).toEqual([
      'page1-0',
      'page2-0',
      'page3-0',
      'page4-0',
      'page5-0',
      'page6-0',
    ]);
    expect(mockFetchAPI).toHaveBeenCalledTimes(6);
    // Each request carries the token the page before it handed back.
    expect(mockFetchAPI.mock.calls.map(([, options]) => (options as any)?.body?.page_token)).toEqual([
      undefined,
      'token-1',
      'token-2',
      'token-3',
      'token-4',
      'token-5',
    ]);
  });

  test('rejects rather than looping when the server repeats a page token', async () => {
    mockFetchAPI.mockImplementation(
      () => Promise.resolve(page('stuck', 1, 'same-token')) as ReturnType<typeof fetchAPI>,
    );

    await expect(searchIssueIdsByStatus('experiment-1', 'rejected')).rejects.toThrow('repeated page token');
    // Caught on the second answer, rather than paging forever.
    expect(mockFetchAPI).toHaveBeenCalledTimes(2);
  });

  test('rejects rather than reading forever when the pages never end', async () => {
    let served = 0;
    mockFetchAPI.mockImplementation(
      () => Promise.resolve(page('endless', 1, `token-${served++}`)) as ReturnType<typeof fetchAPI>,
    );

    await expect(searchIssueIdsByStatus('experiment-1', 'rejected')).rejects.toThrow('within 1000 pages');
    expect(mockFetchAPI).toHaveBeenCalledTimes(1000);
  });

  test("passes the caller's abort signal to every page request", async () => {
    const controller = new AbortController();
    givenPages([page('page1', 1, 'token-1'), page('page2', 1, '')]);

    await searchIssueIdsByStatus('experiment-1', 'rejected', controller.signal);

    expect(mockFetchAPI).toHaveBeenCalledTimes(2);
    for (const [, options] of mockFetchAPI.mock.calls) {
      expect((options as any)?.signal).toBe(controller.signal);
    }
  });

  test('treats a page carrying no issues as the end when the token is empty', async () => {
    givenPages([{ next_page_token: '' }] as ReturnType<typeof page>[]);

    expect(await searchIssueIdsByStatus('experiment-1', 'rejected')).toEqual([]);
  });
});
