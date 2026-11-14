import { jest, describe, test, expect } from '@jest/globals';
import { RunsMetricsBarPlot } from './RunsMetricsBarPlot';
import { renderWithIntl } from '../../../../common/utils/TestUtils.react18';
import { createChartImageDownloadHandler } from '../hooks/useChartImageDownloadHandler';

jest.mock('../../LazyPlot', () => ({
  LazyPlot: jest.fn(() => null),
}));

jest.mock('../hooks/useChartImageDownloadHandler', () => ({
  createChartImageDownloadHandler: jest.fn(() => jest.fn()),
}));

describe('RunsMetricsBarPlot', () => {
  test('exported image keeps separate bars for runs sharing a name', () => {
    const runsData = ['run-1', 'run-2', 'run-3'].map((uuid, idx) => ({
      uuid,
      displayName: 'baseline',
      metrics: { accuracy: { key: 'accuracy', value: idx, step: 0, timestamp: 0 } },
    }));

    renderWithIntl(
      <RunsMetricsBarPlot
        metricKey="accuracy"
        runsData={runsData}
        displayRunNames={false}
        onSetDownloadHandler={jest.fn()}
      />,
    );

    const [data, layout] = jest.mocked(createChartImageDownloadHandler).mock.lastCall ?? [];
    expect(data?.[0]).toEqual(expect.objectContaining({ y: ['run-1', 'run-2', 'run-3'] }));
    expect(layout?.yaxis).toEqual(
      expect.objectContaining({
        showticklabels: true,
        tickmode: 'array',
        tickvals: ['run-1', 'run-2', 'run-3'],
        ticktext: ['baseline', 'baseline', 'baseline'],
      }),
    );
  });
});
