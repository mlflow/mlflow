import { waitFor, screen, waitForElementToBeRemoved } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithIntl } from '../../../common/utils/TestUtils.react18';
import { TracesView, TRACE_AUTO_REFRESH_INTERVAL } from './TracesView';
import { MlflowService } from '../../sdk/MlflowService';
import type { KeyValueEntity } from '../../../common/types';
import type { ModelTraceInfo } from '@databricks/web-shared/model-trace-explorer';
import { MemoryRouter } from '@mlflow/mlflow/src/common/utils/RoutingUtils';

// eslint-disable-next-line no-restricted-syntax -- TODO(FEINF-4392)
jest.setTimeout(90000); // increase timeout

const testExperimentId = 'some-experiment-id';
const testExperimentIds = [testExperimentId];

const pagesCount = 3;

const generateMockTrace = (uniqueId: string, timestampMs = 100, metadata: KeyValueEntity[] = []): ModelTraceInfo => ({
  request_id: `tr-${uniqueId}`,
  experiment_id: testExperimentId,
  timestamp_ms: 1712134300000 + timestampMs,
  execution_time_ms: timestampMs,
  status: 'OK',
  attributes: {},
  request_metadata: [...metadata],
  tags: [],
});

const getMockTraceResponse = (tracesToReturn: number) => {
  return (_: string[], __: string, token: string | undefined) => {
    const page = token ? JSON.parse(token).page : 1;
    const traces = new Array(tracesToReturn).fill(0).map((_, i) => generateMockTrace(`trace-page${page}-${i + 1}`, i));
    const next_page_token = page < pagesCount ? JSON.stringify({ page: page + 1 }) : undefined;
    return Promise.resolve({ traces, next_page_token });
  };
};

describe('TracesView', () => {
  it('should auto-refresh traces', async () => {
    jest.useFakeTimers();

    jest.spyOn(MlflowService, 'getExperimentTraces').mockImplementation(getMockTraceResponse(10));

    renderWithIntl(
      <MemoryRouter>
        <TracesView experimentIds={testExperimentIds} />
      </MemoryRouter>,
    );
    expect(MlflowService.getExperimentTraces).toHaveBeenCalledTimes(1);

    await waitFor(() => {
      // 10 rows + 1 header
      expect(screen.getAllByRole('row')).toHaveLength(11);
    });

    // simulate new traces from the backend
    jest.spyOn(MlflowService, 'getExperimentTraces').mockImplementation(getMockTraceResponse(20));

    jest.advanceTimersByTime(TRACE_AUTO_REFRESH_INTERVAL + 1000);

    // expect that the new traces have been fetched and show up
    expect(MlflowService.getExperimentTraces).toHaveBeenCalledTimes(2);
    await waitFor(() => {
      expect(screen.getAllByRole('row')).toHaveLength(21);
    });
  });

  it('should allow trace selection', async () => {
    // this line is necessary for userEvent to work with fake timers
    const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });

    const mockTraces = new Array(10).fill(0).map((_, i) => generateMockTrace(`trace-${i + 1}`));
    const getTracesSpy = jest.spyOn(MlflowService, 'getExperimentTraces').mockResolvedValue({
      traces: mockTraces,
      next_page_token: undefined,
    });
    const deleteTracesSpy = jest.spyOn(MlflowService, 'deleteTraces').mockResolvedValue({ traces_deleted: 1 });

    renderWithIntl(
      <MemoryRouter>
        <TracesView experimentIds={testExperimentIds} />
      </MemoryRouter>,
    );

    await waitFor(() => {
      expect(screen.getAllByRole('row')).toHaveLength(11);
    });

    const traceIds = mockTraces.map((trace) => trace.request_id);
    const headerCheckbox = screen.getByTestId('trace-table-header-checkbox');

    // clicking on the header checkbox should select all rows
    await user.click(headerCheckbox);
    traceIds.forEach((traceId) => {
      expect(screen.getByTestId(`trace-table-cell-checkbox-${traceId}`)).toBeChecked();
    });

    // when opening the delete dialog, all selected traces should be listed
    await user.click(screen.getByRole('button', { name: 'Delete' }));
    expect(await screen.queryByRole('dialog')).toBeInTheDocument();
    expect(screen.getByText('10 traces will be deleted.')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Close' }));

    // test that deselecting all rows works as well
    await user.click(headerCheckbox);
    traceIds.forEach((traceId) => {
      expect(screen.getByTestId(`trace-table-cell-checkbox-${traceId}`)).not.toBeChecked();
    });

    // test that single-row selections works
    const rowCheckbox = screen.getByTestId(`trace-table-cell-checkbox-${traceIds[0]}`);
    await user.click(rowCheckbox);
    await user.click(screen.getByRole('button', { name: 'Delete' }));

    // test that the description correctly handles plural (10 traces -> 1 trace)
    expect(await screen.queryByRole('dialog')).toBeInTheDocument();
    expect(screen.getByText('1 trace will be deleted.')).toBeInTheDocument();

    // auto-refresh might cause a hardcoded value to flake, so we
    // check that after clicking the button, it has incremented by 1
    const mockCalls = getTracesSpy.mock.calls.length;
    await user.click(screen.getByRole('button', { name: 'Delete 1 trace' }));

    // check that delete and refresh have been called
    expect(deleteTracesSpy).toHaveBeenCalledTimes(1);
    await waitForElementToBeRemoved((): any => screen.queryByRole('dialog'));
    expect(getTracesSpy).toHaveBeenCalledTimes(mockCalls + 1);
  });
});
