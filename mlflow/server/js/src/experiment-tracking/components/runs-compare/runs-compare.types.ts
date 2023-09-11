import type {
  CompareRunsChartAxisDef,
  CompareChartRunData,
} from './charts/CompareRunsCharts.common';
import { getUUID } from '../../../common/utils/ActionUtils';
import { MetricEntitiesByName } from '../../types';

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

  constructor(uuid?: string) {
    this.uuid = uuid;
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
  static getEmptyChartCardByType(type: RunsCompareChartType, uuid?: string) {
    if (type === RunsCompareChartType.BAR) {
      return new RunsCompareBarCardConfig(uuid);
    } else if (type === RunsCompareChartType.SCATTER) {
      return new RunsCompareScatterCardConfig(uuid);
    } else if (type === RunsCompareChartType.PARALLEL) {
      return new RunsCompareParallelCardConfig(uuid);
    } else if (type === RunsCompareChartType.LINE) {
      return new RunsCompareLineCardConfig(uuid);
    } else {
      // Must be contour
      return new RunsCompareContourCardConfig(uuid);
    }
  }

  static getBaseChartConfigs(primaryMetricKey: string, primary_metric_data: MetricEntitiesByName) {
    const resultChartSet: RunsCompareCardConfig[] = [];
    const MAX_NUMBER_OF_METRICS_TO_RENDER = 30;
    const MIN_NUMBER_OF_STEP_FOR_LINE_COMPARISON = 2;

    const metricsToRender: Set<string> = new Set();
    // Add primary_metric to render first
    if (primaryMetricKey) {
      metricsToRender.add(primaryMetricKey);
    }

    // Adding other metrics to render
    for (const metricsKey of Object.keys(primary_metric_data)) {
      metricsToRender.add(metricsKey);
    }

    // Render only first N metrics
    const renderFirstNMetrics: string[] = [...metricsToRender].slice(
      0,
      MAX_NUMBER_OF_METRICS_TO_RENDER,
    );

    renderFirstNMetrics.forEach((metricsKey) => {
      let chartType = RunsCompareChartType.BAR;
      // If the metric has multiple epochs, add a line chart or else add a bar chart
      if (
        primary_metric_data[metricsKey]?.step !== undefined &&
        primary_metric_data[metricsKey]?.step >= MIN_NUMBER_OF_STEP_FOR_LINE_COMPARISON
      ) {
        chartType = RunsCompareChartType.LINE;
      }

      // Add a bar metric chart only if at least one metric key is detected
      resultChartSet.push({
        ...RunsCompareCardConfig.getEmptyChartCardByType(chartType, getUUID()),
        metricKey: metricsKey,
      } as RunsCompareBarCardConfig);
    });

    resultChartSet.push(
      RunsCompareCardConfig.getEmptyChartCardByType(RunsCompareChartType.PARALLEL, getUUID()),
    );

    return resultChartSet;
  }
}

// TODO: add configuration fields relevant to scatter chart
export class RunsCompareScatterCardConfig extends RunsCompareCardConfig {
  type: RunsCompareChartType.SCATTER = RunsCompareChartType.SCATTER;
  xaxis: CompareRunsChartAxisDef = { key: '', type: 'METRIC' };
  yaxis: CompareRunsChartAxisDef = { key: '', type: 'METRIC' };
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
   * Smoothness
   */
  lineSmoothness = 0;

  /**
   * Y axis mode
   */
  scaleType: 'linear' | 'log' = 'linear';

  /**
   * Choose X axis mode - numeric step or absolute time
   */
  xAxisKey: 'step' | 'time' = 'step';
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
  xaxis: CompareRunsChartAxisDef = { key: '', type: 'METRIC' };
  yaxis: CompareRunsChartAxisDef = { key: '', type: 'METRIC' };
  zaxis: CompareRunsChartAxisDef = { key: '', type: 'METRIC' };
}

// TODO: add configuration fields relevant to parallel coords chart
export class RunsCompareParallelCardConfig extends RunsCompareCardConfig {
  type: RunsCompareChartType.PARALLEL = RunsCompareChartType.PARALLEL;
  selectedParams: string[] = [];
  selectedMetrics: string[] = [];
}
