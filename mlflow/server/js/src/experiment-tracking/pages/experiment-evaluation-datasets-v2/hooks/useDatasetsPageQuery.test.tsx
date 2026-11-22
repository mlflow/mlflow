import { afterEach, beforeEach, describe, expect, jest, test } from '@jest/globals';
import { render, renderHook, screen, waitFor, within } from '@testing-library/react';
import { DesignSystemProvider } from '@databricks/design-system';
import type { ReactNode } from 'react';
import { IntlProvider } from 'react-intl';
import { MemoryRouter } from 'react-router-dom';
import { fetchAPI, getAjaxUrl } from '@mlflow/mlflow/src/common/utils/FetchUtils';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { DatasetsListTable } from '../components/DatasetsListTable';
import { DEFAULT_DATASETS_ORDER_BY } from '../utils/constants';
import { useDatasetsPageQuery } from './useDatasetsPageQuery';

jest.mock('@mlflow/mlflow/src/common/utils/FetchUtils', () => ({
  ...jest.requireActual<typeof import('@mlflow/mlflow/src/common/utils/FetchUtils')>(
    '@mlflow/mlflow/src/common/utils/FetchUtils',
  ),
  fetchAPI: jest.fn(),
}));

const params = {
  experimentId: 'exp-1',
  nameFilter: '',
  pageSize: 10,
  pageToken: 'previous-page',
};

describe('useDatasetsPageQuery', () => {
  let queryClient: QueryClient;

  const wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );

  beforeEach(() => {
    queryClient = new QueryClient();
  });

  afterEach(() => {
    queryClient.clear();
    jest.clearAllMocks();
  });

  test('normalizes OSS timestamps for the list table and preserves metadata and pagination', async () => {
    jest.mocked(fetchAPI).mockResolvedValue({
      datasets: [
        {
          dataset_id: 'ds-1',
          name: 'Customer Eval',
          created_time: 1791901590572,
          last_update_time: 1791901591938,
          created_by: 'creator',
          last_updated_by: 'updater',
          digest: 'digest',
          schema: '{"inputs": ["question"]}',
          profile: '{"num_records": 1}',
        },
      ],
      next_page_token: 'next-page',
    });

    const { result } = renderHook(() => useDatasetsPageQuery(params), { wrapper });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data).toEqual({
      datasets: [
        expect.objectContaining({
          dataset_id: 'ds-1',
          name: 'Customer Eval',
          create_time: new Date(1791901590572).toISOString(),
          last_update_time: new Date(1791901591938).toISOString(),
          created_by: 'creator',
          last_updated_by: 'updater',
          digest: 'digest',
          schema: '{"inputs": ["question"]}',
          profile: '{"num_records": 1}',
        }),
      ],
      next_page_token: 'next-page',
    });
    expect(fetchAPI).toHaveBeenCalledTimes(1);
    expect(fetchAPI).toHaveBeenCalledWith(getAjaxUrl('ajax-api/3.0/mlflow/datasets/search'), {
      method: 'POST',
      body: {
        experiment_ids: ['exp-1'],
        filter_string: undefined,
        order_by: [DEFAULT_DATASETS_ORDER_BY],
        max_results: 10,
        page_token: 'previous-page',
      },
    });

    render(
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <MemoryRouter>
            <DatasetsListTable
              experimentId="exp-1"
              datasets={result.current.data?.datasets ?? []}
              hasNextPage
              hasPreviousPage
              onNextPage={jest.fn()}
              onPreviousPage={jest.fn()}
              onDeleteDataset={jest.fn()}
              showPagination={false}
              isLoadingRows={false}
            />
          </MemoryRouter>
        </DesignSystemProvider>
      </IntlProvider>,
    );
    const cells = within(screen.getAllByRole('row')[1]).getAllByRole('cell');
    expect(cells[2]).not.toHaveTextContent('-');
    expect(cells[3]).not.toHaveTextContent('-');
    expect(cells[2].querySelector('span[title]')).toBeInTheDocument();
    expect(cells[3].querySelector('span[title]')).toBeInTheDocument();
  });

  test.each([{}, { datasets: [] }])('handles an empty search response: %j', async (response) => {
    jest.mocked(fetchAPI).mockResolvedValue({ ...response, next_page_token: 'next-page' });

    const { result } = renderHook(() => useDatasetsPageQuery(params), { wrapper });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data?.datasets).toEqual(response.datasets);
    expect(result.current.data?.next_page_token).toBe('next-page');
  });

  test('handles datasets without timestamps', async () => {
    jest.mocked(fetchAPI).mockResolvedValue({ datasets: [{ dataset_id: 'ds-1' }] });

    const { result } = renderHook(() => useDatasetsPageQuery(params), { wrapper });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data?.datasets).toEqual([
      expect.objectContaining({ dataset_id: 'ds-1', create_time: '', last_update_time: '' }),
    ]);
  });
});
