import type { RunsChartAxisDef, RunsChartsRunData } from '../runs-charts/components/RunsCharts.common';
import { getUUID } from '../../../common/utils/ActionUtils';
import { MetricEntitiesByName, ChartSectionConfig } from '../../types';
import {
  MLFLOW_MODEL_METRIC_PREFIX,
  MLFLOW_SYSTEM_METRIC_PREFIX,
  MLFLOW_MODEL_METRIC_NAME,
  MLFLOW_SYSTEM_METRIC_NAME,
} from '../../constants';
import { uniq } from 'lodash';

/**
 * Enum for all recognized chart types used in compare runs
 */
export enum RunsCompareChartType {
  BAR = 'BAR',
  LINE = 'LINE',
  SCATTER = 'SCATTER',
  CONTOUR = 'CONTOUR',
  PARALLEL = 'PARALLEL',
}

const MIN_NUMBER_OF_STEP_FOR_LINE_COMPARISON = 1;

/**
 * Simple interface corresponding to `RunsCompareChartCard`.
 * Its role is to distinguish between stateful class instance and a simple POJO,
 * it is meant to be contained in a serializable, persisted state.
 */
export type SerializedRunsCompareCardConfigCard = RunsCompareCardConfig;

/**
 * Main class used for represent a single configured chart card with its type, configuration options etc.
 * Meant to be extended by various chart type classes with `type` field being frozen to a single value.
 */
export abstract class RunsCompareCardConfig {
  uuid?: string;
  type: RunsCompareChartType = RunsCompareChartType.BAR;
  runsCountToCompare?: number = 10;
  metricSectionId?: string = '';
  deleted = false;
  isGenerated = false;

  constructor(isGenerated: boolean, uuid?: string, metricSectionId?: string) {
    this.isGenerated = isGenerated;
    this.uuid = uuid;
    this.metricSectionId = metricSectionId;
  }

  /**
   * Serializes chart entry, i.e. strips all unnecessary fields (and/or methods) so
   * it can be saved in persistable memory.
   */
  static serialize(entity: Partial<RunsCompareCardConfig>): SerializedRunsCompareCardConfigCard {
    // TODO: strip unnecessary fields if any
    return (Object.keys(entity) as (keyof SerializedRunsCompareCardConfigCard)[]).reduce(
      (result, key) => ({ ...result, [key]: entity[key] }),
      {} as SerializedRunsCompareCardConfigCard,
    );
  }

  /**
   * Creates empty chart (card) config basing on a type.
   * TODO: consume visible run set and determine best configuration of metrics, params etc.
   */
  static getEmptyChartCardByType(
    type: RunsCompareChartType,
    isGenerated: boolean,
    uuid?: string,
    metricSectionId?: string,
  ) {
    if (type === RunsCompareChartType.BAR) {
      return new RunsCompareBarCardConfig(isGenerated, uuid, metricSectionId);
    } else if (type === RunsCompareChartType.SCATTER) {
      return new RunsCompareScatterCardConfig(isGenerated, uuid, metricSectionId);
    } else if (type === RunsCompareChartType.PARALLEL) {
      return new RunsCompareParallelCardConfig(isGenerated, uuid, metricSectionId);
    } else if (type === RunsCompareChartType.LINE) {
      return new RunsCompareLineCardConfig(isGenerated, uuid, metricSectionId);
    } else {
      // Must be contour
      return new RunsCompareContourCardConfig(isGenerated, uuid, metricSectionId);
    }
  }

  static getBaseChartConfigs(primaryMetricKey: string, runsData: RunsChartsRunData[]) {
    const resultChartSet: RunsCompareCardConfig[] = [];
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
      const anyRunHasMultipleEpochs = runsData.some(
        (run) => run.metrics?.[metricsKey]?.step >= MIN_NUMBER_OF_STEP_FOR_LINE_COMPARISON,
      );
      const chartType = anyRunHasMultipleEpochs ? RunsCompareChartType.LINE : RunsCompareChartType.BAR;

      // Add a bar metric chart only if at least one metric key is detected
      resultChartSet.push({
        ...RunsCompareCardConfig.getEmptyChartCardByType(chartType, true, getUUID()),
        metricKey: metricsKey,
      } as RunsCompareBarCardConfig);
    });

    // If no other charts exist, show empty parallel coordinates plot
    if (resultChartSet.length === 0) {
      resultChartSet.push(
        RunsCompareCardConfig.getEmptyChartCardByType(RunsCompareChartType.PARALLEL, false, getUUID()),
      );
    }

    return resultChartSet;
  }

  // Extract chart section from metric key
  static extractChartSectionName = (metricKey: string, delimiter = '/') => {
    const parts = metricKey.split(delimiter);
    const section = parts.slice(0, -1).join(delimiter);
    if (section === MLFLOW_MODEL_METRIC_PREFIX) {
      return MLFLOW_MODEL_METRIC_NAME;
    } else if (section + delimiter === MLFLOW_SYSTEM_METRIC_PREFIX) {
      return MLFLOW_SYSTEM_METRIC_NAME;
    }
    return section;
  };

  static getBaseChartAndSectionConfigs(primaryMetricKey: string, runsData: RunsChartsRunData[]) {
    const resultChartSet: RunsCompareCardConfig[] = [];

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

    const sectionName2Uuid: Record<string, string> = {};
    sectionName2Uuid[MLFLOW_MODEL_METRIC_NAME] = getUUID();
    sectionName2Uuid[MLFLOW_SYSTEM_METRIC_NAME] = getUUID();

    metricsToRender.forEach((metricsKey) => {
      if (!sectionName2Uuid[RunsCompareCardConfig.extractChartSectionName(metricsKey)]) {
        sectionName2Uuid[RunsCompareCardConfig.extractChartSectionName(metricsKey)] = getUUID();
      }
    });

    Array.from(metricsToRender)
      .sort()
      .forEach((metricsKey) => {
        // If the metric has multiple epochs, add a line chart. Otherwise, add a bar chart
        const anyRunHasMultipleEpochs = runsData.some(
          (run) => run.metrics?.[metricsKey]?.step >= MIN_NUMBER_OF_STEP_FOR_LINE_COMPARISON,
        );
        const chartType = anyRunHasMultipleEpochs ? RunsCompareChartType.LINE : RunsCompareChartType.BAR;

        const sectionId = sectionName2Uuid[RunsCompareCardConfig.extractChartSectionName(metricsKey)];

        // Add a bar metric chart only if at least one metric key is detected
        resultChartSet.push({
          ...RunsCompareCardConfig.getEmptyChartCardByType(chartType, true, getUUID(), sectionId),
          metricKey: metricsKey,
        } as RunsCompareBarCardConfig);
      });

    // If no other charts exist, show empty parallel coordinates plot
    if (resultChartSet.length === 0) {
      const sectionId = sectionName2Uuid[MLFLOW_MODEL_METRIC_NAME];
      resultChartSet.push(
        RunsCompareCardConfig.getEmptyChartCardByType(RunsCompareChartType.PARALLEL, false, getUUID(), sectionId),
      );
    }
    const rest = Object.keys(sectionName2Uuid)
      .filter((sectionName) => sectionName !== MLFLOW_MODEL_METRIC_NAME && sectionName !== MLFLOW_SYSTEM_METRIC_NAME)
      .sort();
    const sortedSectionNames = [...rest, MLFLOW_MODEL_METRIC_NAME, MLFLOW_SYSTEM_METRIC_NAME];

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

  static updateChartAndSectionConfigs(
    compareRunCharts: RunsCompareCardConfig[],
    compareRunSections: ChartSectionConfig[],
    runsData: RunsChartsRunData[],
    isAccordionReordered: boolean,
  ) {
    // Make copies of the current charts and sections
    const resultChartSet: RunsCompareCardConfig[] = compareRunCharts.slice();
    let resultSectionSet: ChartSectionConfig[] = compareRunSections.slice();
    // Flag for whether the section or chart set have been updated
    let isResultUpdated = false;

    const allMetricKeys = uniq(runsData.flatMap((run) => Object.keys(run.metrics)));

    // Create set of metrics to render based on runsData
    const metricsToRender: Set<string> = new Set();
    // Adding other metrics to render
    for (const metricsKey of allMetricKeys) {
      metricsToRender.add(metricsKey);
    }

    // Create sectionName2Uuid mappings from existing sections
    const sectionName2Uuid: Record<string, string> = {};
    compareRunSections.forEach((section) => (sectionName2Uuid[section.name] = section.uuid));

    // Append new charts at the end instead of alphabetically
    metricsToRender.forEach((metricKey) => {
      // Check if metricKey exists in the current chart set
      const doesMetricKeyExist =
        resultChartSet.findIndex((chart) => {
          const chartMetricKey = (chart as RunsCompareBarCardConfig).metricKey;
          return chartMetricKey ? chartMetricKey === metricKey : false;
        }) >= 0;

      // Check if there is a generated chart with metricKey
      const generatedChartIndex = resultChartSet.findIndex((chart) => {
        const chartMetricKey = (chart as RunsCompareBarCardConfig).metricKey;
        return chartMetricKey && chartMetricKey === metricKey && chart.isGenerated;
      });

      // If the metric has multiple epochs, add a line chart. Otherwise, add a bar chart
      const anyRunHasMultipleEpochs = runsData.some(
        (run) => run.metrics?.[metricKey]?.step >= MIN_NUMBER_OF_STEP_FOR_LINE_COMPARISON,
      );
      const chartType = anyRunHasMultipleEpochs ? RunsCompareChartType.LINE : RunsCompareChartType.BAR;

      // This is a new metric key, so add it to the chart set
      if (!doesMetricKeyExist) {
        // result is updated when there is a new metric key
        isResultUpdated = true;

        // Insert a new UUID if section doesn't exist
        const sectionName = RunsCompareCardConfig.extractChartSectionName(metricKey);
        if (!sectionName2Uuid[sectionName]) {
          sectionName2Uuid[sectionName] = getUUID();
        }

        // Get section for metricKey and check if it has been reordered
        const sectionId = sectionName2Uuid[sectionName];
        // If section is undefined, it may be a new section, so its not reordered
        const section = resultSectionSet.find((section) => section.uuid === sectionId);
        const isSectionReordered = section ? section.isReordered : false;

        const newChartConfig = {
          ...RunsCompareCardConfig.getEmptyChartCardByType(chartType, true, getUUID(), sectionId),
          metricKey: metricKey,
        } as RunsCompareBarCardConfig;

        if (isSectionReordered) {
          // If the section has been reordered, then append to the end of the section
          resultChartSet.push(newChartConfig);
        } else {
          // If section has not been reordered, then insert alphabetically
          const insertIndex = resultChartSet.findIndex((chart) => {
            const chartMetricKey = (chart as RunsCompareBarCardConfig).metricKey;
            return chartMetricKey ? chartMetricKey.localeCompare(metricKey) >= 0 : false;
          });
          resultChartSet.splice(insertIndex, 0, newChartConfig);
        }
      } else if (
        generatedChartIndex >= 0 &&
        resultChartSet[generatedChartIndex].type === RunsCompareChartType.BAR &&
        chartType === RunsCompareChartType.LINE
      ) {
        isResultUpdated = true;
        // If the chart type has been updated to a line chart from a bar chart, then update the chart type
        const prevChart = resultChartSet[generatedChartIndex];
        resultChartSet[generatedChartIndex] = {
          ...RunsCompareCardConfig.getEmptyChartCardByType(
            chartType,
            prevChart.isGenerated,
            prevChart.uuid,
            prevChart.metricSectionId,
          ),
          metricKey: metricKey,
          deleted: prevChart.deleted,
        } as RunsCompareLineCardConfig;
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
      ];
    }

    return { resultChartSet, resultSectionSet, isResultUpdated };
  }
}

// TODO: add configuration fields relevant to scatter chart
export class RunsCompareScatterCardConfig extends RunsCompareCardConfig {
  type: RunsCompareChartType.SCATTER = RunsCompareChartType.SCATTER;
  xaxis: RunsChartAxisDef = { key: '', type: 'METRIC' };
  yaxis: RunsChartAxisDef = { key: '', type: 'METRIC' };
  runsCountToCompare = 100;
}

// TODO: add configuration fields relevant to line chart
export class RunsCompareLineCardConfig extends RunsCompareCardConfig {
  type: RunsCompareChartType.LINE = RunsCompareChartType.LINE;

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
   * Y axis mode
   */
  scaleType: 'linear' | 'log' = 'linear';

  /**
   * Choose X axis mode - numeric step, relative time in seconds or absolute time value
   */
  xAxisKey: 'step' | 'time' | 'time-relative' = 'step';
}

// TODO: add configuration fields relevant to bar chart
export class RunsCompareBarCardConfig extends RunsCompareCardConfig {
  type: RunsCompareChartType.BAR = RunsCompareChartType.BAR;

  /**
   * A metric key used for chart's X axis
   */
  metricKey = '';
}

// TODO: add configuration fields relevant to contour chart
export class RunsCompareContourCardConfig extends RunsCompareCardConfig {
  type: RunsCompareChartType.CONTOUR = RunsCompareChartType.CONTOUR;
  xaxis: RunsChartAxisDef = { key: '', type: 'METRIC' };
  yaxis: RunsChartAxisDef = { key: '', type: 'METRIC' };
  zaxis: RunsChartAxisDef = { key: '', type: 'METRIC' };
}

// TODO: add configuration fields relevant to parallel coords chart
export class RunsCompareParallelCardConfig extends RunsCompareCardConfig {
  type: RunsCompareChartType.PARALLEL = RunsCompareChartType.PARALLEL;
  selectedParams: string[] = [];
  selectedMetrics: string[] = [];
}
