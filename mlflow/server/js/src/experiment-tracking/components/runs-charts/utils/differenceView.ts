import { MLFLOW_SYSTEM_METRIC_PREFIX } from '@mlflow/mlflow/src/experiment-tracking/constants';
import type { MetricEntitiesByName } from '../../../types';
import type { KeyValueEntity } from '../../../../common/types';
import type { RunsChartsRunData } from '../components/RunsCharts.common';
import type { RunsChartsDifferenceCardConfig } from '../runs-charts.types';
import { DifferenceCardAttributes } from '../runs-charts.types';
import Utils from '@mlflow/mlflow/src/common/utils/Utils';
import type { RunsGroupByConfig } from '../../experiment-page/utils/experimentPage.group-row-utils';

const DIFFERENCE_CHART_DEFAULT_EMPTY_VALUE = '-';
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
        const user = Utils.getUser(runData.runInfo ?? {}, runData.tags);
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

export const DIFFERENCE_PLOT_EXPAND_COLUMN_ID = 'expand';
export const DIFFERENCE_PLOT_HEADING_COLUMN_ID = 'headingColumn';

/**
 * Transforms an array of objects into a format suitable for rendering in a table.
 * Each object in the array represents a row in the table.
 * If all values in a row are JSON objects with the same keys, the row is transformed into a parent row with child rows.
 * Each child row represents a key-value pair from the JSON objects.
 * If a value in a row is not a JSON object or the JSON objects don't have the same keys, the row is not transformed.
 *
 * @param data - An array of objects. Each object represents a row in the table.
 * @returns An array of objects. Each object represents a row or a parent row with child rows in the table.
 */
export const getDifferencePlotJSONRows = (data: { [key: string]: string | number }[]) => {
  const validateParseJSON = (value: string) => {
    try {
      const parsed = JSON.parse(value);
      if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed) || Object.keys(parsed).length === 0) {
        return null;
      }
      return parsed;
    } catch (e) {
      return null;
    }
  };

  const extractMaximumCommonSchema = (schema1: Record<any, any> | undefined, schema2: Record<any, any> | undefined) => {
    if (schema1 !== undefined && Object.keys(schema1).length === 0) {
      // This may not be a suitable object, return null
      return null;
    } else if (schema2 !== undefined && Object.keys(schema2).length === 0) {
      return null;
    }

    const schema: Record<string, unknown> = {};

    const collectKeys = (target: Record<any, any>, source: Record<any, any>) => {
      for (const key in source) {
        if (!target.hasOwnProperty(key) || target[key]) {
          if (typeof source[key] === 'object' && source[key] !== null && !Array.isArray(source[key])) {
            target[key] = target[key] || {};
            collectKeys(target[key], source[key]);
          } else if (source[key] === DIFFERENCE_CHART_DEFAULT_EMPTY_VALUE) {
            target[key] = true;
          } else {
            target[key] = false;
          }
        }
      }
    };

    schema1 !== undefined && collectKeys(schema, schema1);
    schema2 !== undefined && collectKeys(schema, schema2);

    return schema;
  };

  const getChildren = (
    parsedRowWithoutHeadingCol: { [key: string]: Record<any, any> | undefined },
    schema: Record<any, any>,
  ): Record<string, any>[] => {
    return Object.keys(schema).map((key) => {
      if (typeof schema[key] === 'boolean') {
        let result = {
          key: key,
          [DIFFERENCE_PLOT_HEADING_COLUMN_ID]: key,
        };
        Object.keys(parsedRowWithoutHeadingCol).forEach((runUuid) => {
          const value = parsedRowWithoutHeadingCol[runUuid]?.[key];
          result = {
            ...result,
            [runUuid]: value === undefined ? DIFFERENCE_CHART_DEFAULT_EMPTY_VALUE : value,
          };
        });
        return result;
      }
      // Recurse
      const newParsedRow: { [key: string]: Record<any, any> | undefined } = {};
      Object.keys(parsedRowWithoutHeadingCol).forEach((runUuid) => {
        newParsedRow[runUuid] = parsedRowWithoutHeadingCol[runUuid]?.[key];
      });

      return {
        key: key,
        [DIFFERENCE_PLOT_HEADING_COLUMN_ID]: key,
        children: getChildren(newParsedRow, schema[key]),
      };
    });
  };

  const isAllElementsJSON = (row: { [key: string]: string | number }) => {
    let jsonSchema: Record<any, any> | undefined = undefined;
    let isAllJson = true;
    const parsedRow: Record<string, any> = {};

    Object.keys(row).forEach((runUuid) => {
      if (runUuid !== DIFFERENCE_PLOT_HEADING_COLUMN_ID) {
        if (row[runUuid] !== DIFFERENCE_CHART_DEFAULT_EMPTY_VALUE) {
          const json = validateParseJSON(row[runUuid] as string);
          parsedRow[runUuid] = json;
          if (json === null) {
            isAllJson = false;
          } else {
            const commonSchema = extractMaximumCommonSchema(jsonSchema, json);
            if (commonSchema === null) {
              isAllJson = false;
            } else {
              jsonSchema = commonSchema;
            }
          }
        }
      }
    });
    if (isAllJson && jsonSchema !== undefined) {
      try {
        return {
          [DIFFERENCE_PLOT_HEADING_COLUMN_ID]: row[DIFFERENCE_PLOT_HEADING_COLUMN_ID],
          children: getChildren(parsedRow, jsonSchema),
          key: row[DIFFERENCE_PLOT_HEADING_COLUMN_ID],
        };
      } catch {
        return row;
      }
    } else {
      return row;
    }
  };
  return data.map(isAllElementsJSON);
};
