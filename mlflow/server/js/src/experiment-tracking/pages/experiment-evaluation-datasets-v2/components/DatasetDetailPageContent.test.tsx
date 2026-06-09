import { describe, expect, jest, test, beforeEach } from '@jest/globals';
import { fireEvent, screen, waitFor } from '@testing-library/react';
import { setupTestRouter } from '@mlflow/mlflow/src/common/utils/RoutingTestUtils';
import { fetchAPI } from '@mlflow/mlflow/src/common/utils/FetchUtils';
import type { Dataset } from '../hooks/useDatasetsQueries';
import { DatasetDetailPageContent } from './DatasetDetailPageContent';
import { renderDatasetsPage } from '../test-utils/renderDatasetsPage';

jest.mock('@mlflow/mlflow/src/common/utils/FetchUtils', () => ({
  getAjaxUrl: (url: string) => `/${url}`,
  fetchAPI: jest.fn(),
}));

// Stub the Monaco-backed editor so the side panel can render in jsdom without loading Monaco.
jest.mock('./LazyJsonRecordEditor', () => {
  const ReactLib = jest.requireActual<typeof import('react')>('react');
  return {
    LazyJsonRecordEditor: ({
      value,
      onChange,
      ariaLabel,
    }: {
      value: string;
      onChange: (v: string) => void;
      ariaLabel: string;
    }) =>
      ReactLib.createElement('textarea', {
        'aria-label': ariaLabel,
        value,
        onChange: (e: { target: { value: string } }) => onChange(e.target.value),
      }),
  };
});

const mockFetchAPI = jest.mocked(fetchAPI);

const DATASET: Dataset = { dataset_id: 'ds-1', name: 'test', create_time: '2026-01-01T00:00:00Z' };

const renderPage = () => {
  const { history } = setupTestRouter();
  const result = renderDatasetsPage({
    initialUrl: '/experiments/exp-1/datasets/ds-1',
    history,
    routes: [
      {
        path: '/experiments/:experimentId/datasets/:datasetId',
        element: <DatasetDetailPageContent experimentId="exp-1" datasetId="ds-1" dataset={DATASET} />,
      },
    ],
  });
  return { ...result, history };
};

describe('DatasetDetailPageContent — single-step add', () => {
  beforeEach(() => {
    mockFetchAPI.mockReset();
    mockFetchAPI.mockImplementation((...args: unknown[]) => {
      const url = String(args[0]);
      const opts = args[1] as { method?: string } | undefined;
      const method = opts?.method ?? 'GET';
      if (method === 'GET' && url.includes('/records')) {
        // Empty dataset -> empty state with the "+ Add record" button.
        return Promise.resolve({ records: JSON.stringify([]), next_page_token: undefined });
      }
      if (method === 'POST' && url.includes('/records')) {
        // Create (upsert) echoes back the new id.
        return Promise.resolve({ inserted_count: 1, updated_count: 0, record_ids: ['dr-new'] });
      }
      return Promise.resolve({});
    });
  });

  test('clicking "+ Add record" creates one record and opens it for editing', async () => {
    renderPage();

    const addButton = await screen.findByRole('button', { name: /add record/i });
    fireEvent.click(addButton);

    // Exactly one create POST to the records endpoint fires on the click (not per keystroke).
    await waitFor(() => {
      const posts = mockFetchAPI.mock.calls.filter((call) => {
        const opts = call[1] as { method?: string } | undefined;
        return opts?.method === 'POST' && String(call[0]).includes('/records');
      });
      expect(posts).toHaveLength(1);
    });

    // The new record is selected and the side panel opens in edit mode — its inputs editor
    // (Monaco stubbed to a textarea above) only renders when a record is selected by id.
    expect(await screen.findByLabelText(/dataset record inputs/i)).toBeInTheDocument();
  }, 15000);
});
