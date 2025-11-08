import { renderHook, waitFor } from '@testing-library/react';
import { useExperimentLoggedModelsChartsUIState } from './useExperimentLoggedModelsChartsUIState';
import type { LoggedModelProto } from '../../../types';
import { useExperimentLoggedModelAllMetricsByDataset } from './useExperimentLoggedModelAllMetricsByDataset';
import { getMetricByDatasetChartDataKey } from './useExperimentLoggedModelsChartsData';

const getLoggedModelList = (n = 5) => {
  return Array.from(
    { length: n },
    (_, index): LoggedModelProto => ({
      info: { name: 'model-' + (index + 1) },
      data: {
        metrics: [
          { key: 'dataset-metric-' + (index + 1), dataset_name: 'dataset-' + (index + 1), value: (index + 1) * 10 },
          { key: 'independent-metric', value: (index + 1) * 10 },
        ],
      },
    }),
  );
};

const modelSetWithOneMetric = getLoggedModelList(1);
const modelSetWithTwoMetrics = getLoggedModelList(2);

describe('useExperimentLoggedModelsChartsUIState, useExperimentLoggedModelAllMetricsByDataset', () => {
  const renderTestHook = (data: LoggedModelProto[]) =>
    renderHook(
      (props) => {
        const metrics = useExperimentLoggedModelAllMetricsByDataset(props.data);
        return useExperimentLoggedModelsChartsUIState(metrics, 'test-experiment-id');
      },
      {
        initialProps: { data },
      },
    );
  test('it should generate the correct initial state and expand it when new metrics arrive', async () => {
    const { result, rerender } = renderTestHook(modelSetWithOneMetric);

    await waitFor(() => {
      expect(result.current.chartUIState.compareRunCharts).toEqual([
        expect.objectContaining({
          metricSectionId: 'autogen-dataset-1',
          metricKey: 'dataset-metric-1',
          datasetName: 'dataset-1',
          dataAccessKey: getMetricByDatasetChartDataKey('dataset-metric-1', 'dataset-1'),
        }),
        expect.objectContaining({ metricSectionId: 'default', metricKey: 'independent-metric' }),
      ]);

      expect(result.current.chartUIState.compareRunSections).toEqual([
        expect.objectContaining({ name: 'dataset-1', uuid: 'autogen-dataset-1' }),
        expect.objectContaining({ name: 'Metrics', uuid: 'default' }),
      ]);
    });

    rerender({ data: modelSetWithTwoMetrics });

    await waitFor(() => {
      expect(result.current.chartUIState.compareRunCharts).toEqual([
        expect.objectContaining({
          metricSectionId: 'autogen-dataset-1',
          metricKey: 'dataset-metric-1',
          datasetName: 'dataset-1',
          dataAccessKey: getMetricByDatasetChartDataKey('dataset-metric-1', 'dataset-1'),
        }),
        expect.objectContaining({
          metricSectionId: 'autogen-dataset-2',
          metricKey: 'dataset-metric-2',
          datasetName: 'dataset-2',
          dataAccessKey: getMetricByDatasetChartDataKey('dataset-metric-2', 'dataset-2'),
        }),
        expect.objectContaining({ metricSectionId: 'default', metricKey: 'independent-metric' }),
      ]);

      expect(result.current.chartUIState.compareRunSections).toEqual([
        expect.objectContaining({ name: 'dataset-1', uuid: 'autogen-dataset-1' }),
        expect.objectContaining({ name: 'dataset-2', uuid: 'autogen-dataset-2' }),
        expect.objectContaining({ name: 'Metrics', uuid: 'default' }),
      ]);
    });
  });
});
