import { beforeEach, describe, expect, it, jest } from '@jest/globals';
import { render, screen, waitFor } from '@testing-library/react';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';
import { searchIssueIdsByStatus } from '@mlflow/mlflow/src/experiment-tracking/components/run-page/hooks/useSearchIssuesQuery';

import { MemoryRouter, Route, Routes } from 'react-router-dom';

import { IssuesCell } from './IssuesCell';
import { QueryClient, QueryClientProvider } from '../../query-client/queryClient';

jest.mock('@mlflow/mlflow/src/experiment-tracking/components/run-page/hooks/useSearchIssuesQuery', () => ({
  ...jest.requireActual<
    typeof import('@mlflow/mlflow/src/experiment-tracking/components/run-page/hooks/useSearchIssuesQuery')
  >('@mlflow/mlflow/src/experiment-tracking/components/run-page/hooks/useSearchIssuesQuery'),
  searchIssueIdsByStatus: jest.fn(),
}));

const mockSearchIssueIdsByStatus = jest.mocked(searchIssueIdsByStatus);

/** Answers the experiment's dismissed-issue lookup. */
const givenDismissedIssueIds = (issueIds: string[]) => mockSearchIssueIdsByStatus.mockResolvedValue(issueIds);

type CellIssue = { id: string; name: string };

/** The table is mounted under an experiment route, which the dismissed-issue lookup reads. */
const renderCells = (cells: React.ReactNode) =>
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
          <MemoryRouter initialEntries={['/experiments/experiment-1/traces']}>
            <Routes>
              <Route path="/experiments/:experimentId/*" element={cells} />
            </Routes>
          </MemoryRouter>
        </QueryClientProvider>
      </DesignSystemProvider>
    </IntlProvider>,
  );

const renderRows = (rows: CellIssue[][]) =>
  renderCells(
    <>
      {rows.map((issues, index) => (
        <IssuesCell key={index} issues={issues} otherIssues={undefined} isComparing={false} />
      ))}
    </>,
  );

const renderCell = (issues: CellIssue[]) => renderRows([issues]);

describe('IssuesCell', () => {
  beforeEach(() => {
    mockSearchIssueIdsByStatus.mockReset();
  });

  it('drops an issue dismissed as not an issue and keeps the rest', async () => {
    givenDismissedIssueIds(['iss-dismissed']);
    renderCell([
      { id: 'iss-dismissed', name: 'Not really a bug' },
      { id: 'iss-open', name: 'Genuinely slow' },
      { id: 'iss-resolved', name: 'Fixed since' },
    ]);

    // Shown until the lookup answers, so the dismissed one is waited out.
    await waitFor(() => expect(screen.queryByText('Not really a bug')).not.toBeInTheDocument());
    expect(screen.getByText('Genuinely slow')).toBeInTheDocument();
    // Only the first survivor is tagged; the rest sit behind the overflow trigger, which is
    // unopened here. So the resolved issue shows up as the count rather than by name: two left.
    expect(screen.getByText('+1')).toBeInTheDocument();
  });

  it('keeps an issue that was resolved rather than dismissed', async () => {
    // Sole survivor here, so its name is in the cell rather than behind the overflow trigger.
    givenDismissedIssueIds(['iss-dismissed']);
    renderCell([
      { id: 'iss-dismissed', name: 'Not really a bug' },
      { id: 'iss-resolved', name: 'Fixed since' },
    ]);

    await waitFor(() => expect(screen.queryByText('Not really a bug')).not.toBeInTheDocument());
    expect(screen.getByText('Fixed since')).toBeInTheDocument();
    expect(mockSearchIssueIdsByStatus).toHaveBeenCalledWith('experiment-1', 'rejected');
  });

  it('falls back to the empty cell when every issue was dismissed', async () => {
    givenDismissedIssueIds(['iss-dismissed']);
    renderCell([{ id: 'iss-dismissed', name: 'Not really a bug' }]);

    await waitFor(() => expect(screen.queryByText('Not really a bug')).not.toBeInTheDocument());
  });

  it('keeps every issue when the lookup fails', async () => {
    mockSearchIssueIdsByStatus.mockRejectedValue(new Error('issues API unavailable'));
    renderCell([{ id: 'iss-open', name: 'Genuinely slow' }]);

    await waitFor(() => expect(mockSearchIssueIdsByStatus).toHaveBeenCalled());
    expect(screen.getByText('Genuinely slow')).toBeInTheDocument();
  });

  it('looks the experiment up once for a whole page of rows', async () => {
    givenDismissedIssueIds([]);
    // Four issues between three rows, under one cache: one request, not one per distinct issue.
    renderRows([
      [{ id: 'iss-open', name: 'Genuinely slow' }],
      [{ id: 'iss-other', name: 'Something else' }],
      [
        { id: 'iss-third', name: 'Third' },
        { id: 'iss-fourth', name: 'Fourth' },
      ],
    ]);

    await waitFor(() => expect(mockSearchIssueIdsByStatus).toHaveBeenCalled());
    expect(mockSearchIssueIdsByStatus).toHaveBeenCalledTimes(1);
  });

  it('asks for nothing when no row names an issue', () => {
    givenDismissedIssueIds([]);
    renderRows([[], []]);

    expect(mockSearchIssueIdsByStatus).not.toHaveBeenCalled();
  });

  it('looks the experiment up once across both compared columns', async () => {
    givenDismissedIssueIds([]);
    renderCells(
      <IssuesCell
        issues={[{ id: 'iss-open', name: 'Genuinely slow' }]}
        otherIssues={[{ id: 'iss-other', name: 'Something else' }]}
        isComparing
      />,
    );

    await waitFor(() => expect(mockSearchIssueIdsByStatus).toHaveBeenCalled());
    expect(mockSearchIssueIdsByStatus).toHaveBeenCalledTimes(1);
  });
});
