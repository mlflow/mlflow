import { orderBy } from 'lodash';
import type { LoggedModelProto } from '../../../types';
import { useMemo } from 'react';
import type { RunsChartsMetricByDatasetEntry } from '../../runs-charts/runs-charts.types';
import { getMetricByDatasetChartDataKey } from './useExperimentLoggedModelsChartsData';

export const useExperimentLoggedModelAllMetricsByDataset = (loggedModels: LoggedModelProto[]) =>
  useMemo(() => {
    const metricsByDataset: RunsChartsMetricByDatasetEntry[] = [];
    loggedModels.forEach((model) => {
      model.data?.metrics?.forEach(({ key: metricKey, dataset_name: datasetName }) => {
        if (metricKey && !metricsByDataset.find((e) => e.metricKey === metricKey && e.datasetName === datasetName)) {
          const dataAccessKey = getMetricByDatasetChartDataKey(metricKey, datasetName);
          metricsByDataset.push({ metricKey, datasetName, dataAccessKey });
        }
      });
    });
    return orderBy(metricsByDataset, ({ datasetName }) => !datasetName);
  }, [loggedModels]);
