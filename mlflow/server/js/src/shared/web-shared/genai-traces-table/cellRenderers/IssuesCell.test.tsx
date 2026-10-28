import { beforeEach, describe, expect, it, jest } from '@jest/globals';
import { render, screen, waitFor } from '@testing-library/react';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';
import { getIssue } from '@mlflow/mlflow/src/experiment-tracking/components/run-page/hooks/useGetIssueQuery';

import { MemoryRouter } from 'react-router-dom';

import { IssuesCell } from './IssuesCell';
import { QueryClient, QueryClientProvider } from '../../query-client/queryClient';

jest.mock('@mlflow/mlflow/src/experiment-tracking/components/run-page/hooks/useGetIssueQuery', () => ({
  ...jest.requireActual<
    typeof import('@mlflow/mlflow/src/experiment-tracking/components/run-page/hooks/useGetIssueQuery')
  >('@mlflow/mlflow/src/experiment-tracking/components/run-page/hooks/useGetIssueQuery'),
  getIssue: jest.fn(),
}));

const mockGetIssue = jest.mocked(getIssue);

/** Answers the status read for each issue ID; an ID left out resolves as still open. */
const givenIssueStatuses = (statuses: Record<string, string>) =>
  mockGetIssue.mockImplementation((issueId: string) =>
    Promise.resolve({ issue_id: issueId, status: statuses[issueId] ?? 'pending' } as Awaited<
      ReturnType<typeof getIssue>
    >),
  );

const renderCell = (issues: { id: string; name: string }[]) =>
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
          <MemoryRouter>
            <IssuesCell issues={issues} otherIssues={undefined} isComparing={false} />
          </MemoryRouter>
        </QueryClientProvider>
      </DesignSystemProvider>
    </IntlProvider>,
  );

describe('IssuesCell', () => {
  beforeEach(() => {
    mockGetIssue.mockReset();
  });

  it('drops an issue dismissed as not an issue and keeps the rest', async () => {
    givenIssueStatuses({ 'iss-dismissed': 'rejected', 'iss-open': 'pending', 'iss-resolved': 'resolved' });
    renderCell([
      { id: 'iss-dismissed', name: 'Not really a bug' },
      { id: 'iss-open', name: 'Genuinely slow' },
      { id: 'iss-resolved', name: 'Fixed since' },
    ]);

    // Shown until the statuses arrive, so the dismissed one has to be waited out.
    await waitFor(() => expect(screen.queryByText('Not really a bug')).not.toBeInTheDocument());
    // A resolved issue did happen on this trace, so the column still reports it.
    expect(screen.getByText('Genuinely slow')).toBeInTheDocument();
  });

  it('falls back to the empty cell when every issue was dismissed', async () => {
    givenIssueStatuses({ 'iss-dismissed': 'rejected' });
    renderCell([{ id: 'iss-dismissed', name: 'Not really a bug' }]);

    await waitFor(() => expect(screen.queryByText('Not really a bug')).not.toBeInTheDocument());
  });

  it('keeps an issue whose status cannot be read', async () => {
    mockGetIssue.mockRejectedValue(new Error('issues API unavailable'));
    renderCell([{ id: 'iss-open', name: 'Genuinely slow' }]);

    await waitFor(() => expect(mockGetIssue).toHaveBeenCalled());
    expect(screen.getByText('Genuinely slow')).toBeInTheDocument();
  });

  it('reads each issue once however many rows name it', async () => {
    givenIssueStatuses({ 'iss-open': 'pending' });
    renderCell([{ id: 'iss-open', name: 'Genuinely slow' }]);
    renderCell([{ id: 'iss-open', name: 'Genuinely slow' }]);

    await waitFor(() => expect(mockGetIssue).toHaveBeenCalled());
    // Both cells share the query cache, so the second render adds no request.
    expect(new Set(mockGetIssue.mock.calls.map(([issueId]) => issueId))).toEqual(new Set(['iss-open']));
  });
});
