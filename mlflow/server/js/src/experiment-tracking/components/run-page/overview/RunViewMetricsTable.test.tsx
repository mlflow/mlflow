import { keyBy } from 'lodash';
import { renderWithIntl, fastFillInput, act, screen, within } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import type { LoggedModelProto, MetricEntitiesByName, RunInfoEntity } from '../../../types';
import { RunViewMetricsTable } from './RunViewMetricsTable';
import { MemoryRouter } from '../../../../common/utils/RoutingUtils';

// Larger timeout for integration testing (table rendering)
// eslint-disable-next-line no-restricted-syntax -- TODO(FEINF-4392)
jest.setTimeout(60000);

const testRunUuid = 'test-run-uuid';
const testRunName = 'Test run name';
const testExperimentId = '12345';

const testRunInfo = {
  experimentId: testExperimentId,
  lifecycleStage: 'active',
  runName: testRunName,
  runUuid: testRunUuid,
} as RunInfoEntity;

// Generates array of metric_a1, metric_a2, ..., metric_b2, ..., metric_c3 metric keys with values from 1.0 to 9.0
const sampleLatestMetrics = keyBy(
  ['a', 'b', 'c'].flatMap((letter, letterIndex) =>
    [1, 2, 3].map((digit, digitIndex) => ({
      key: `metric_${letter}${digit}`,
      value: (letterIndex * 3 + digitIndex + 1).toFixed(1),
    })),
  ),
  'key',
) as any;

describe('RunViewMetricsTable', () => {
  const renderComponent = (
    latestMetrics: MetricEntitiesByName = sampleLatestMetrics,
    loggedModels?: LoggedModelProto[],
  ) => {
    return renderWithIntl(
      <MemoryRouter>
        <RunViewMetricsTable runInfo={testRunInfo} latestMetrics={latestMetrics} loggedModels={loggedModels} />
      </MemoryRouter>,
    );
  };

  test('Renders the table with no metrics recorded', () => {
    renderComponent({});
    expect(screen.getByText('No metrics recorded')).toBeInTheDocument();
  });
  test('Renders the table with values and filters values', async () => {
    renderComponent();
    expect(screen.getByRole('heading', { name: 'Metrics (9)' })).toBeInTheDocument();
    expect(screen.getByRole('row', { name: 'metric_a1 1.0' })).toBeInTheDocument();
    expect(screen.getByRole('row', { name: 'metric_c3 9.0' })).toBeInTheDocument();

    // Expect 10 rows for 9 metrics and 1 table header
    expect(screen.getAllByRole('row')).toHaveLength(9 + 1);

    // Change the filter query
    await fastFillInput(screen.getByRole('textbox'), 'metric_a');

    // Expect 4 rows for 3 filtered metrics and 1 table header
    expect(screen.getAllByRole('row')).toHaveLength(3 + 1);

    // Change the filter query
    await fastFillInput(screen.getByRole('textbox'), 'metric_xyz');

    // Expect no result rows, a header row and a message
    expect(screen.queryAllByRole('row')).toHaveLength(0 + 1);
    expect(screen.getByText('No metrics match the search filter')).toBeInTheDocument();
  });
  test('Renders the table with system and model metrics', async () => {
    renderComponent({
      'system/system_metric_abc': { key: 'system/system_metric_abc', value: 'system_value_1' },
      'system/system_metric_xyz': { key: 'system/system_metric_xyz', value: 'system_value_2' },
      model_metric_1: { key: 'model_metric_abc', value: 'model_value_abc' },
      model_metric_2: { key: 'model_metric_xyz', value: 'model_value_xyz' },
    } as any);
    expect(screen.getByRole('heading', { name: 'Metrics (4)' })).toBeInTheDocument();

    // Expect 7 rows: 4 rows for metrics, 2 header rows for sections (system and model) and 1 header row for table
    expect(screen.getAllByRole('row')).toHaveLength(4 + 2 + 1);
    expect(screen.getByRole('cell', { name: /^System metrics/ })).toBeInTheDocument();
    expect(screen.getByRole('cell', { name: /^Model metrics/ })).toBeInTheDocument();

    // Change the filter query
    await fastFillInput(screen.getByRole('textbox'), 'system_');

    // Expect 4 rows: 2 rows for metrics, 1 header row for sections (system) and 1 header row for table
    expect(screen.getAllByRole('row')).toHaveLength(2 + 1 + 1);
    expect(screen.queryByRole('cell', { name: /^System metrics/ })).toBeInTheDocument();
    expect(screen.queryByRole('cell', { name: /^Model metrics/ })).not.toBeInTheDocument();

    // Change the filter query
    await fastFillInput(screen.getByRole('textbox'), 'foo-bar-abc-xyz');

    // Expect no result rows, a header row and a message
    expect(screen.queryAllByRole('row')).toHaveLength(0 + 1);
    expect(screen.getByText('No metrics match the search filter')).toBeInTheDocument();
  });
  test('Renders the table with logged models', () => {
    renderComponent(sampleLatestMetrics, [
      {
        data: { metrics: [sampleLatestMetrics['metric_a2']] },
        info: { name: 'model_a2', model_id: 'm-a2' },
      },
      {
        data: { metrics: [sampleLatestMetrics['metric_c1']] },
        info: { name: 'model_c1', model_id: 'm-c1' },
      },
    ]);
    const rowWithMetricA2 = screen.getByRole('row', { name: /metric_a2/ });
    const rowWithMetricC1 = screen.getByRole('row', { name: /metric_c1/ });

    expect(within(rowWithMetricA2).getByRole('link', { name: 'model_a2' })).toHaveAttribute(
      'href',
      expect.stringMatching(/models\/m-a2/),
    );

    expect(within(rowWithMetricC1).getByRole('link', { name: 'model_c1' })).toHaveAttribute(
      'href',
      expect.stringMatching(/models\/m-c1/),
    );
  });
});
