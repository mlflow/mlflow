import { MLFLOW_SYSTEM_METRIC_PREFIX } from '@mlflow/mlflow/src/experiment-tracking/constants';
import { KeyValueEntity, MetricEntitiesByName } from '@mlflow/mlflow/src/experiment-tracking/types';
import { useCallback, useMemo } from 'react';
import { RunsChartsRunData } from '../components/RunsCharts.common';
import { DifferenceCardAttributes, RunsChartsDifferenceCardConfig } from '../runs-charts.types';
import Utils from '@mlflow/mlflow/src/common/utils/Utils';
import type { RunsGroupByConfig } from '../../experiment-page/utils/experimentPage.group-row-utils';

export const DIFFERENCE_CHART_DEFAULT_EMPTY_VALUE = '-';
const DIFFERENCE_EPSILON = 1e-6;
export const getDifferenceChartDisplayedValue = (val: any, places = 2) => {
  if (typeof val === 'number') {
    return val.toFixed(places);
  } else if (typeof val === 'string') {
    return val;
  }
  try {
    return JSON.stringify(val);
  } catch {
    return DIFFERENCE_CHART_DEFAULT_EMPTY_VALUE;
  }
};

export enum DifferenceChartCellDirection {
  POSITIVE,
  NEGATIVE,
  SAME,
}
export const differenceView = (a: any, b: any) => {
  if (typeof a !== 'number' || typeof b !== 'number') {
    return undefined;
  } else {
    const diff = a - b;
    if (diff === 0) {
      return { label: getDifferenceChartDisplayedValue(diff).toString(), direction: DifferenceChartCellDirection.SAME };
    } else if (diff > 0) {
      return { label: `+${getDifferenceChartDisplayedValue(diff)}`, direction: DifferenceChartCellDirection.POSITIVE };
    } else {
      return {
        label: getDifferenceChartDisplayedValue(diff).toString(),
        direction: DifferenceChartCellDirection.NEGATIVE,
      };
    }
  }
};

export const isDifferent = (a: any, b: any) => {
  if (a === DIFFERENCE_CHART_DEFAULT_EMPTY_VALUE || b === DIFFERENCE_CHART_DEFAULT_EMPTY_VALUE) {
    return false;
  }
  // Check if type a and b are the same
  if (typeof a !== typeof b) {
    return true;
  } else if (typeof a === 'number' && typeof b === 'number') {
    return Math.abs(a - b) > DIFFERENCE_EPSILON;
  } else if (typeof a === 'string' && typeof b === 'string') {
    return a !== b;
  }
  return false;
};

export const getDifferenceViewDataGroups = (
  previewData: RunsChartsRunData[],
  cardConfig: RunsChartsDifferenceCardConfig,
  headingColumnId: string,
  groupBy: RunsGroupByConfig | null,
) => {
  const getMetrics = (
    filterCondition: (metric: string) => boolean,
    runDataKeys: (data: RunsChartsRunData) => string[],
    runDataAttribute: (
      data: RunsChartsRunData,
    ) =>
      | MetricEntitiesByName
      | Record<string, KeyValueEntity>
      | Record<string, { key: string; value: string | number }>,
  ) => {
    // Get array of sorted keys
    const keys = Array.from(new Set(previewData.flatMap((runData) => runDataKeys(runData))))
      .filter((key) => filterCondition(key))
      .sort();
    const values = keys.flatMap((key: string) => {
      const data: Record<string, string | number> = {};
      let hasDifference = false;

      previewData.forEach((runData, index) => {
        // Set the key as runData.uuid and the value as the metric's value or DEFAULT_EMPTY_VALUE
        data[runData.uuid] = runDataAttribute(runData)[key]
          ? runDataAttribute(runData)[key].value
          : DIFFERENCE_CHART_DEFAULT_EMPTY_VALUE;
        if (index > 0) {
          const prev = previewData[index - 1];
          if (isDifferent(data[prev.uuid], data[runData.uuid])) {
            hasDifference = true;
          }
        }
      });
      if (cardConfig.showDifferencesOnly && !hasDifference) {
        return [];
      }
      return [
        {
          [headingColumnId]: key,
          ...data,
        },
      ];
    });
    return values;
  };

  const modelMetrics = getMetrics(
    (metric: string) => !metric.startsWith(MLFLOW_SYSTEM_METRIC_PREFIX),
    (runData: RunsChartsRunData) => Object.keys(runData.metrics),
    (runData: RunsChartsRunData) => runData.metrics,
  );

  const systemMetrics = getMetrics(
    (metric: string) => metric.startsWith(MLFLOW_SYSTEM_METRIC_PREFIX),
    (runData: RunsChartsRunData) => Object.keys(runData.metrics),
    (runData: RunsChartsRunData) => runData.metrics,
  );

  if (groupBy) {
    return { modelMetrics, systemMetrics, parameters: [], tags: [], attributes: [] };
  }

  const parameters = getMetrics(
    () => true,
    (runData: RunsChartsRunData) => Object.keys(runData.params),
    (runData: RunsChartsRunData) => runData.params,
  );

  const tags = getMetrics(
    () => true,
    (runData: RunsChartsRunData) => Utils.getVisibleTagValues(runData.tags).map(([key]) => key),
    (runData: RunsChartsRunData) => runData.tags,
  );

  // Get attributes
  const attributeGroups = [
    DifferenceCardAttributes.USER,
    DifferenceCardAttributes.SOURCE,
    DifferenceCardAttributes.VERSION,
    DifferenceCardAttributes.MODELS,
  ];
  const attributes = attributeGroups.flatMap((attribute) => {
    const attributeData: Record<string, string | number> = {};
    let hasDifference = false;
    previewData.forEach((runData, index) => {
      if (attribute === DifferenceCardAttributes.USER) {
        const user = Utils.getUser(runData.runInfo, runData.tags);
        attributeData[runData.uuid] = user;
      } else if (attribute === DifferenceCardAttributes.SOURCE) {
        const source = Utils.getSourceName(runData.tags);
        attributeData[runData.uuid] = source;
      } else if (attribute === DifferenceCardAttributes.VERSION) {
        const version = Utils.getSourceVersion(runData.tags);
        attributeData[runData.uuid] = version;
      } else {
        const models = Utils.getLoggedModelsFromTags(runData.tags);
        attributeData[runData.uuid] = models.join(',');
      }
      if (index > 0) {
        const prev = previewData[index - 1];
        if (isDifferent(attributeData[prev.uuid], attributeData[runData.uuid])) {
          hasDifference = true;
        }
      }
    });
    if (cardConfig.showDifferencesOnly && !hasDifference) {
      return [];
    }
    return [
      {
        [headingColumnId]: attribute,
        ...attributeData,
      },
    ];
  });
  return { modelMetrics, systemMetrics, parameters, tags, attributes };
};
