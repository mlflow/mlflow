import { compact, keyBy } from 'lodash';
import { useMemo } from 'react';
import type { LoggedModelProto } from '../../../types';
import { getStableColorForRun } from '../../../utils/RunNameUtils';
import type { RunsChartsRunData } from '../../runs-charts/components/RunsCharts.common';
import { useExperimentLoggedModelListPageRowVisibilityContext } from './useExperimentLoggedModelListPageRowVisibility';

export const getMetricByDatasetChartDataKey = (metricKey?: string, datasetName?: string) =>
  datasetName ? JSON.stringify([datasetName, metricKey]) : metricKey ?? '';

/**
 * Creates chart-consumable data based on logged models, including metrics and parameters.
 * TODO: optimize, add unit tests
 */
export const useExperimentLoggedModelsChartsData = (loggedModels: LoggedModelProto[]) => {
  const { isRowHidden } = useExperimentLoggedModelListPageRowVisibilityContext();
  return useMemo<RunsChartsRunData[]>(
    () =>
      compact(
        loggedModels.map<RunsChartsRunData | null>((model, index) =>
          model.info?.model_id
            ? {
                displayName: model.info?.name ?? model.info?.model_id ?? 'Unknown',
                images: {},

                metrics: keyBy(
                  model.data?.metrics?.map(({ dataset_name, key, value, timestamp, step }) => ({
                    // Instead of using plain metric key, we will use specific data access key generated based on metric key and dataset
                    dataKey: getMetricByDatasetChartDataKey(key, dataset_name),
                    key: key ?? '',
                    value: value ?? 0,
                    timestamp: timestamp ?? 0,
                    step: step ?? 0,
                  })),
                  'dataKey',
                ),
                params: keyBy(
                  model.data?.params
                    ?.map(({ key, value }) => ({ key: key ?? '', value: value ?? '' }))
                    .filter(({ key }) => key) ?? [],
                  'key',
                ),
                tags: {},
                uuid: model.info.model_id,
                hidden: isRowHidden(model.info.model_id, index),
                color: getStableColorForRun(model.info.model_id),
              }
            : null,
        ),
      ),
    [loggedModels, isRowHidden],
  );
};
