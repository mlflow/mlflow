import type { RunsChartAxisDef, RunsChartsRunData } from './components/RunsCharts.common';
import { RunsChartsLineChartXAxisType } from './components/RunsCharts.common';
import { getUUID } from '../../../common/utils/ActionUtils';
import type { ChartSectionConfig } from '../../types';
import { MetricEntitiesByName } from '../../types';
import {
  MLFLOW_MODEL_METRIC_PREFIX,
  MLFLOW_SYSTEM_METRIC_PREFIX,
  MLFLOW_MODEL_METRIC_NAME,
  MLFLOW_SYSTEM_METRIC_NAME,
  DEFAULT_IMAGE_GRID_CHART_NAME,
} from '../../constants';
import { isNil, uniq } from 'lodash';
import { customMetricBehaviorDefs } from '../experiment-page/utils/customMetricBehaviorUtils';

/**
 * Enum for all recognized chart types used in runs charts
 */
export enum RunsChartType {
  BAR = 'BAR',
  LINE = 'LINE',
  SCATTER = 'SCATTER',
  CONTOUR = 'CONTOUR',
  PARALLEL = 'PARALLEL',
  DIFFERENCE = 'DIFFERENCE',
  IMAGE = 'IMAGE',
}

const MIN_NUMBER_OF_STEP_FOR_LINE_COMPARISON = 1;

/**
 * Simple interface corresponding to `RunsChartsCardConfig`.
 * Its role is to distinguish between stateful class instance and a simple POJO,
 * it is meant to be contained in a serializable, persisted state.
 */
export type SerializedRunsChartsCardConfigCard = RunsChartsCardConfig;

// A function to iterate across run/group data traces and determine if any metric has multiple epochs.
// This helps to decide if we should seed the line chart or a bar chart.
const dataTraceMetricsContainMultipleEpochs = (dataTrace: RunsChartsRunData, metricKey: string): boolean =>
  Boolean(dataTrace.metrics?.[metricKey]?.step >= MIN_NUMBER_OF_STEP_FOR_LINE_COMPARISON);

/**
 * Main class used for represent a single configured chart card with its type, configuration options etc.
 * Meant to be extended by various chart type classes with `type` field being frozen to a single value.
 */
export abstract class RunsChartsCardConfig {
  uuid?: string;
  type: RunsChartType = RunsChartType.BAR;
  runsCountToCompare?: number = 10;
  metricSectionId?: string = '';
  deleted = false;
  isGenerated = false;

  // Custom title of the chart. If not provided, it's inferred from configuration (e.g. metric key).
  displayName?: string;

  constructor(isGenerated: boolean, uuid?: string, metricSectionId?: string) {
    this.isGenerated = isGenerated;
    this.uuid = uuid;
    this.metricSectionId = metricSectionId;
  }

  /**
   * Serializes chart entry, i.e. strips all unnecessary fields (and/or methods) so
   * it can be saved in persistable memory.
   */
  static serialize(entity: Partial<RunsChartsCardConfig>): SerializedRunsChartsCardConfigCard {
    // TODO: strip unnecessary fields if any
    return (Object.keys(entity) as (keyof SerializedRunsChartsCardConfigCard)[]).reduce(
      (result, key) => ({ ...result, [key]: entity[key] }),
      {} as SerializedRunsChartsCardConfigCard,
    );
  }

  /**
   * Creates empty chart (card) config basing on a type.
   * TODO: consume visible run set and determine best configuration of metrics, params etc.
   */
  static getEmptyChartCardByType(type: RunsChartType, isGenerated: boolean, uuid?: string, metricSectionId?: string) {
    if (type === RunsChartType.BAR) {
      return new RunsChartsBarCardConfig(isGenerated, uuid, metricSectionId);
    } else if (type === RunsChartType.SCATTER) {
      return new RunsChartsScatterCardConfig(isGenerated, uuid, metricSectionId);
    } else if (type === RunsChartType.PARALLEL) {
      return new RunsChartsParallelCardConfig(isGenerated, uuid, metricSectionId);
    } else if (type === RunsChartType.LINE) {
      return new RunsChartsLineCardConfig(isGenerated, uuid, metricSectionId);
    } else if (type === RunsChartType.DIFFERENCE) {
      return new RunsChartsDifferenceCardConfig(isGenerated, uuid, metricSectionId);
    } else if (type === RunsChartType.IMAGE) {
      return new RunsChartsImageCardConfig(isGenerated, uuid, metricSectionId);
    } else {
      // Must be contour
      return new RunsChartsContourCardConfig(isGenerated, uuid, metricSectionId);
    }
  }

  static getBaseChartConfigs(primaryMetricKey: string, runsData: RunsChartsRunData[]) {
    const resultChartSet: RunsChartsCardConfig[] = [];
    const MAX_NUMBER_OF_METRICS_TO_RENDER = 30;

    const allMetricKeys = uniq(runsData.flatMap((run) => Object.keys(run.metrics)));

    const metricsToRender: Set<string> = new Set();
    // Add primary_metric to render first
    if (primaryMetricKey) {
      metricsToRender.add(primaryMetricKey);
    }

    // Adding other metrics to render
    for (const metricsKey of allMetricKeys) {
      metricsToRender.add(metricsKey);
    }

    // Render only first N metrics
    const renderFirstNMetrics: string[] = [...metricsToRender].slice(0, MAX_NUMBER_OF_METRICS_TO_RENDER);

    renderFirstNMetrics.forEach((metricsKey) => {
      // If the metric has multiple epochs, add a line chart. Otherwise, add a bar chart
      const anyRunHasMultipleEpochs = runsData.some((dataTrace) =>
        dataTraceMetricsContainMultipleEpochs(dataTrace, metricsKey),
      );
      const chartType = anyRunHasMultipleEpochs ? RunsChartType.LINE : RunsChartType.BAR;

      // Add a bar metric chart only if at least one metric key is detected
      resultChartSet.push({
        ...RunsChartsCardConfig.getEmptyChartCardByType(chartType, true, getUUID()),
        metricKey: metricsKey,
      } as RunsChartsBarCardConfig);
    });

    // If no other charts exist, show empty parallel coordinates plot
    if (resultChartSet.length === 0) {
      resultChartSet.push(RunsChartsCardConfig.getEmptyChartCardByType(RunsChartType.PARALLEL, false, getUUID()));
    }

    return resultChartSet;
  }

  // Extract chart section from metric key
  static extractChartSectionName = (metricKey: string, delimiter = '/') => {
    const displayMetricName = customMetricBehaviorDefs[metricKey]?.displayName ?? metricKey;

    const parts = displayMetricName.split(delimiter);
    const section = parts.slice(0, -1).join(delimiter);
    if (section === MLFLOW_MODEL_METRIC_PREFIX) {
      return MLFLOW_MODEL_METRIC_NAME;
    } else if (section + delimiter === MLFLOW_SYSTEM_METRIC_PREFIX) {
      return MLFLOW_SYSTEM_METRIC_NAME;
    }
    return section;
  };

  static getBaseChartAndSectionConfigs({
    primaryMetricKey,
    runsData,
    useParallelCoordinatesChart = false,
    enabledSectionNames = [MLFLOW_MODEL_METRIC_NAME, MLFLOW_SYSTEM_METRIC_NAME],
    filterMetricNames,
  }: {
    primaryMetricKey?: string;
    useParallelCoordinatesChart?: boolean;
    runsData: RunsChartsRunData[];
    enabledSectionNames?: string[];
    filterMetricNames?: (metricName: string) => boolean;
  }) {
    const resultChartSet: RunsChartsCardConfig[] = [];

    const allMetricKeys = uniq(runsData.flatMap((run) => Object.keys(run.metrics))).filter((name) => {
      if (!filterMetricNames) {
        return true;
      }
      return filterMetricNames(name);
    });

    const metricsToRender: Set<string> = new Set();
    // Add primary_metric to render first
    if (primaryMetricKey) {
      metricsToRender.add(primaryMetricKey);
    }

    const imagesToRender = uniq(runsData.flatMap((run) => Object.keys(run.images)));

    // Adding other metrics to render
    for (const metricsKey of allMetricKeys) {
      metricsToRender.add(metricsKey);
    }

    const sectionName2Uuid: Record<string, string> = {};
    enabledSectionNames.forEach((sectionName) => {
      sectionName2Uuid[sectionName] = getUUID();
    });

    [...metricsToRender, ...imagesToRender].forEach((key) => {
      if (!sectionName2Uuid[RunsChartsCardConfig.extractChartSectionName(key)]) {
        sectionName2Uuid[RunsChartsCardConfig.extractChartSectionName(key)] = getUUID();
      }
    });

    Array.from(metricsToRender)
      .sort()
      .forEach((metricsKey) => {
        // If the metric has multiple epochs, add a line chart. Otherwise, add a bar chart
        const anyRunHasMultipleEpochs = runsData.some((dataTrace) =>
          dataTraceMetricsContainMultipleEpochs(dataTrace, metricsKey),
        );
        const chartType = anyRunHasMultipleEpochs ? RunsChartType.LINE : RunsChartType.BAR;

        const sectionId = sectionName2Uuid[RunsChartsCardConfig.extractChartSectionName(metricsKey)];

        // Add a bar metric chart only if at least one metric key is detected
        resultChartSet.push({
          ...RunsChartsCardConfig.getEmptyChartCardByType(chartType, true, getUUID(), sectionId),
          metricKey: metricsKey,
        } as RunsChartsBarCardConfig);
      });

    Array.from(imagesToRender)
      .sort()
      .forEach((imageKey) => {
        const chartType = RunsChartType.IMAGE;
        const sectionId = sectionName2Uuid[RunsChartsCardConfig.extractChartSectionName(imageKey)];

        // Add an image chart
        resultChartSet.push({
          ...RunsChartsCardConfig.getEmptyChartCardByType(chartType, true, getUUID(), sectionId),
          imageKeys: [imageKey],
        } as RunsChartsImageCardConfig);
      });

    // If no other charts exist, show empty parallel coordinates plot
    if (resultChartSet.length === 0 && useParallelCoordinatesChart) {
      const sectionId = sectionName2Uuid[MLFLOW_MODEL_METRIC_NAME];
      resultChartSet.push(
        RunsChartsCardConfig.getEmptyChartCardByType(RunsChartType.PARALLEL, false, getUUID(), sectionId),
      );
    }
    const rest = Object.keys(sectionName2Uuid)
      .filter((sectionName) => sectionName !== MLFLOW_MODEL_METRIC_NAME && sectionName !== MLFLOW_SYSTEM_METRIC_NAME)
      .sort();

    const sortedSectionNames = [
      ...rest,
      ...[MLFLOW_MODEL_METRIC_NAME, MLFLOW_SYSTEM_METRIC_NAME].filter((name) => enabledSectionNames.includes(name)),
    ];

    // Create section configs
    const resultSectionSet: ChartSectionConfig[] = sortedSectionNames.map((sectionName) => ({
      uuid: sectionName2Uuid[sectionName],
      name: sectionName,
      display: true,
      isReordered: false,
      deleted: false,
      isGenerated: true,
    }));

    return { resultChartSet, resultSectionSet };
  }

  static updateChartAndSectionConfigs({
    compareRunCharts,
    compareRunSections,
    isAccordionReordered,
    runsData,
    filterMetricNames,
  }: {
    compareRunCharts: RunsChartsCardConfig[];
    compareRunSections: ChartSectionConfig[];
    runsData: RunsChartsRunData[];
    isAccordionReordered: boolean;
    filterMetricNames?: (metricName: string) => boolean;
  }) {
    // Make copies of the current charts and sections
    const resultChartSet: RunsChartsCardConfig[] = compareRunCharts.slice();
    let resultSectionSet: ChartSectionConfig[] = compareRunSections.slice();
    // Flag for whether the section or chart set have been updated
    let isResultUpdated = false;

    const allMetricKeys = uniq(runsData.flatMap((run) => Object.keys(run.metrics))).filter((name) => {
      if (!filterMetricNames) {
        return true;
      }
      return filterMetricNames(name);
    });

    const imagesToRender = uniq(runsData.flatMap((run) => Object.keys(run.images)));

    // Create set of metrics to render based on runsData
    const metricsToRender: Set<string> = new Set();
    // Adding other metrics to render
    for (const metricsKey of allMetricKeys) {
      metricsToRender.add(metricsKey);
    }

    // Create sectionName2Uuid mappings from existing sections
    const sectionName2Uuid: Record<string, string> = {};
    compareRunSections.forEach((section) => (sectionName2Uuid[section.name] = section.uuid));

    imagesToRender.forEach((imageKey) => {
      const doesImageKeyExist =
        resultChartSet.findIndex((chart) => {
          const chartImageKeys = (chart as RunsChartsImageCardConfig).imageKeys;
          return chartImageKeys ? chartImageKeys.length === 1 && chartImageKeys.includes(imageKey) : false;
        }) >= 0;

      const chartType = RunsChartType.IMAGE;
      if (!doesImageKeyExist) {
        // result is updated when there is a new image key
        isResultUpdated = true;

        // Insert a new UUID if section doesn't exist
        const sectionName = RunsChartsCardConfig.extractChartSectionName(imageKey);
        if (!sectionName2Uuid[sectionName]) {
          sectionName2Uuid[sectionName] = getUUID();
        }

        // Get section for imageKey and check if it has been reordered
        const sectionId = sectionName2Uuid[sectionName];
        // If section is undefined, it may be a new section, so its not reordered
        const section = resultSectionSet.find((section) => section.uuid === sectionId);
        const isSectionReordered = section ? section.isReordered : false;

        const newChartConfig = {
          ...RunsChartsCardConfig.getEmptyChartCardByType(chartType, true, getUUID(), sectionId),
          imageKeys: [imageKey],
        } as RunsChartsImageCardConfig;

        if (isSectionReordered) {
          // If the section has been reordered, then append to the end of the section
          resultChartSet.push(newChartConfig);
        } else {
          // If section has not been reordered, then insert alphabetically
          const insertIndex = resultChartSet.findIndex((chart) => {
            const chartImageKeys = (chart as RunsChartsImageCardConfig).imageKeys;
            return chartImageKeys ? chartImageKeys[0]?.localeCompare(imageKey) >= 0 : false;
          });
          resultChartSet.splice(insertIndex, 0, newChartConfig);
        }
      }
    });

    // Append new charts at the end instead of alphabetically
    metricsToRender.forEach((metricKey) => {
      // Check if metricKey exists in the current chart set
      const doesMetricKeyExist =
        resultChartSet.findIndex((chart) => {
          const chartMetricKey = (chart as RunsChartsBarCardConfig).metricKey;
          return chartMetricKey ? chartMetricKey === metricKey : false;
        }) >= 0;

      // Check if there is a generated chart with metricKey
      const generatedChartIndex = resultChartSet.findIndex((chart) => {
        const chartMetricKey = (chart as RunsChartsBarCardConfig).metricKey;
        return chartMetricKey && chartMetricKey === metricKey && chart.isGenerated;
      });

      // If the metric has multiple epochs, add a line chart. Otherwise, add a bar chart
      const anyRunHasMultipleEpochs = runsData.some((dataTrace) =>
        dataTraceMetricsContainMultipleEpochs(dataTrace, metricKey),
      );
      const chartType = anyRunHasMultipleEpochs ? RunsChartType.LINE : RunsChartType.BAR;

      // This is a new metric key, so add it to the chart set
      if (!doesMetricKeyExist) {
        // result is updated when there is a new metric key
        isResultUpdated = true;

        // Insert a new UUID if section doesn't exist
        const sectionName = RunsChartsCardConfig.extractChartSectionName(metricKey);
        if (!sectionName2Uuid[sectionName]) {
          sectionName2Uuid[sectionName] = getUUID();
        }

        // Get section for metricKey and check if it has been reordered
        const sectionId = sectionName2Uuid[sectionName];
        // If section is undefined, it may be a new section, so its not reordered
        const section = resultSectionSet.find((section) => section.uuid === sectionId);
        const isSectionReordered = section ? section.isReordered : false;

        const newChartConfig = {
          ...RunsChartsCardConfig.getEmptyChartCardByType(chartType, true, getUUID(), sectionId),
          metricKey: metricKey,
        } as RunsChartsBarCardConfig;

        if (isSectionReordered) {
          // If the section has been reordered, then append to the end of the section
          resultChartSet.push(newChartConfig);
        } else {
          // If section has not been reordered, then insert alphabetically
          const insertIndex = resultChartSet.findIndex((chart) => {
            const chartMetricKey = (chart as RunsChartsBarCardConfig).metricKey;
            return chartMetricKey ? chartMetricKey.localeCompare(metricKey) >= 0 : false;
          });
          resultChartSet.splice(insertIndex, 0, newChartConfig);
        }
      } else if (
        generatedChartIndex >= 0 &&
        resultChartSet[generatedChartIndex].type === RunsChartType.BAR &&
        chartType === RunsChartType.LINE
      ) {
        isResultUpdated = true;
        // If the chart type has been updated to a line chart from a bar chart, then update the chart type
        const prevChart = resultChartSet[generatedChartIndex];
        resultChartSet[generatedChartIndex] = {
          ...RunsChartsCardConfig.getEmptyChartCardByType(
            chartType,
            prevChart.isGenerated,
            prevChart.uuid,
            prevChart.metricSectionId,
          ),
          metricKey: metricKey,
          deleted: prevChart.deleted,
        } as RunsChartsLineCardConfig;
      }
    });

    Object.keys(sectionName2Uuid).forEach((sectionName) => {
      // Check if it is a new section
      const doesSectionNameExist = resultSectionSet.findIndex((section) => section.name === sectionName) >= 0;
      if (!doesSectionNameExist) {
        resultSectionSet.push({
          uuid: sectionName2Uuid[sectionName],
          name: sectionName,
          display: true,
          isReordered: false,
        });
      }
    });

    if (!isAccordionReordered) {
      // If sections are in order (not been reordered), then sort alphabetically
      const rest = resultSectionSet.filter(
        (section) => section.name !== MLFLOW_MODEL_METRIC_NAME && section.name !== MLFLOW_SYSTEM_METRIC_NAME,
      );
      rest.sort((a, b) => a.name.localeCompare(b.name));
      resultSectionSet = [
        ...rest,
        compareRunSections[compareRunSections.length - 2],
        compareRunSections[compareRunSections.length - 1],
      ].filter((section) => !isNil(section));
    }

    return { resultChartSet, resultSectionSet, isResultUpdated };
  }
}

// TODO: add configuration fields relevant to scatter chart
export class RunsChartsScatterCardConfig extends RunsChartsCardConfig {
  type: RunsChartType.SCATTER = RunsChartType.SCATTER;
  xaxis: RunsChartAxisDef = { key: '', type: 'METRIC' };
  yaxis: RunsChartAxisDef = { key: '', type: 'METRIC' };
  runsCountToCompare = 100;
}

export interface ChartRange {
  xMin?: number;
  xMax?: number;
  yMin?: number;
  yMax?: number;
}

export enum RunsChartsLineChartYAxisType {
  METRIC = 'metric',
  EXPRESSION = 'expression',
}

export interface RunsChartsLineChartExpression {
  // The expression parsed in Reverse Polish Notation
  rpn: (string | number)[];
  // The parsed variables in the expression
  variables: string[];
  // The original input expression
  expression: string;
}

// TODO: add configuration fields relevant to line chart
export class RunsChartsLineCardConfig extends RunsChartsCardConfig {
  type: RunsChartType.LINE = RunsChartType.LINE;

  /**
   * A metric key used for chart's X axis
   */
  metricKey = '';

  /**
   * New key to support multiple metrics
   * NOTE: This key will not be present in older charts
   */
  selectedMetricKeys?: string[];

  /**
   * Smoothness
   */
  lineSmoothness = 0;

  /**
   * Scale type for X axis
   */
  xAxisScaleType: 'linear' | 'log' = 'linear';

  /**
   * Y axis mode
   */
  scaleType: 'linear' | 'log' = 'linear';

  /**
   * Range of Y axis, X axis.
   */
  range?: ChartRange = {
    xMin: undefined,
    xMax: undefined,
    yMin: undefined,
    yMax: undefined,
  };

  /**
   * Choose X axis mode - numeric step, relative time in seconds or absolute time value
   */
  xAxisKey: RunsChartsLineChartXAxisType = RunsChartsLineChartXAxisType.STEP;

  /**
   * Name of the metric to use for the X axis. Used when xAxisKey is set to 'metric'
   */
  selectedXAxisMetricKey = '';

  /**
   * Display points on the line chart. Undefined means "auto" mode, i.e. display points only when
   * there are fewer than 60 datapoints on the chart.
   */
  displayPoints?: boolean = undefined;

  /**
   * Show metrics directly or custom expressions on metrics
   */
  yAxisKey?: RunsChartsLineChartYAxisType = RunsChartsLineChartYAxisType.METRIC;

  /**
   * Custom expressions for Y axis
   */
  yAxisExpressions?: RunsChartsLineChartExpression[] = [];

  /**
   * Whether or not to ignore outliers. If true, the data will be clipped
   * to the 5th and 95th percentiles.
   */
  ignoreOutliers?: boolean = false;

  /**
   * Whether or not to use global X axis settings.
   */
  useGlobalXaxisKey?: boolean = true;

  /**
   * Whether or not to use global line smoothing setting.
   */
  useGlobalLineSmoothing?: boolean = true;
}

// TODO: add configuration fields relevant to bar chart
export class RunsChartsBarCardConfig extends RunsChartsCardConfig {
  type: RunsChartType.BAR = RunsChartType.BAR;

  /**
   * A metric key used for chart's X axis
   */
  metricKey = '';

  /**
   * If the chart is configured to use a particular dataset, this field will contain the dataset identifier.
   */
  datasetName?: string;

  /**
   * Present if the chart is configured to use a particular key to get data.
   */
  dataAccessKey?: string;
}

// TODO: add configuration fields relevant to contour chart
export class RunsChartsContourCardConfig extends RunsChartsCardConfig {
  type: RunsChartType.CONTOUR = RunsChartType.CONTOUR;
  xaxis: RunsChartAxisDef = { key: '', type: 'METRIC' };
  yaxis: RunsChartAxisDef = { key: '', type: 'METRIC' };
  zaxis: RunsChartAxisDef = { key: '', type: 'METRIC' };
}

// TODO: add configuration fields relevant to parallel coords chart
export class RunsChartsParallelCardConfig extends RunsChartsCardConfig {
  type: RunsChartType.PARALLEL = RunsChartType.PARALLEL;
  selectedParams: string[] = [];
  selectedMetrics: string[] = [];
  showAllRuns?: boolean = false;
}

export enum DifferenceCardConfigCompareGroup {
  MODEL_METRICS = 'Model metrics',
  SYSTEM_METRICS = 'System metrics',
  PARAMETERS = 'Parameters',
  ATTRIBUTES = 'Attributes',
  TAGS = 'Tags',
}

export const DISABLED_GROUP_WHEN_GROUPBY = [
  DifferenceCardConfigCompareGroup.PARAMETERS,
  DifferenceCardConfigCompareGroup.TAGS,
  DifferenceCardConfigCompareGroup.ATTRIBUTES,
];

export enum DifferenceCardAttributes {
  USER = 'User',
  SOURCE = 'Source',
  VERSION = 'Version',
  MODELS = 'Models',
}

// TODO: add configuration fields relevant to difference view chart
export class RunsChartsDifferenceCardConfig extends RunsChartsCardConfig {
  type: RunsChartType = RunsChartType.DIFFERENCE;
  compareGroups: DifferenceCardConfigCompareGroup[] = [];
  chartName = 'Runs difference view';
  showChangeFromBaseline = true;
  showDifferencesOnly = true;
  baselineColumnUuid = '';
}

export class RunsChartsImageCardConfig extends RunsChartsCardConfig {
  type: RunsChartType = RunsChartType.IMAGE;
  // image keys to show
  imageKeys: string[] = [];
  step = 0;
}

/**
 * Defines a metric entry in the chart configuration, used to access dataset-specific metrics.
 */
export interface RunsChartsMetricByDatasetEntry {
  dataAccessKey: string;
  metricKey: string;
  datasetName?: string;
}
