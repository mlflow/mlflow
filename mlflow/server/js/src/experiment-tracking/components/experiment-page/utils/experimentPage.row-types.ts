import type { ModelVersionInfoEntity, RunInfoEntity, RunDatasetWithTags, MetricEntity } from '../../../types';
import type { KeyValueEntity } from '../../../../common/types';
import type { LoggedModelProto } from '../../../types';

/**
 * Represents how eye icon should be displayed for a particular row in runs table.
 */
export enum RunRowVisibilityControl {
  /**
   * Eye icon button is enabled and visible.
   */
  Enabled = 0,
  /**
   * Eye icon button is disabled but visible.
   */
  Disabled = 1,
  /**
   * Eye icon button is hidden.
   */
  Hidden = 2,
}

/**
 * Represents a single ag-grid compatible row used in Experiment View runs table.
 */
export interface RunRowType {
  /**
   * Contains run UUID. Empty string for group rows.
   */
  runUuid: string;

  /**
   * Unique identifier for both run and group rows. Used by ag-grid to identify rows.
   */
  rowUuid: string;

  runInfo?: RunInfoEntity;
  experimentName?: { name: string; basename: string };
  experimentId?: string;
  duration: string | null;
  user?: string;
  pinned: boolean;
  hidden: boolean;
  pinnable: boolean;
  runName?: string;
  runStatus?: string;
  defaultColor?: string;
  tags?: Record<string, { key: string; value: string }>;
  params?: KeyValueEntity[];
  visibilityControl?: RunRowVisibilityControl;

  /**
   * Contains information about run's date, timing and hierarchy. Empty for group rows.
   */
  runDateAndNestInfo?: RunRowDateAndNestInfo;

  /**
   * Set if the row is a group header/parent. Contains information about contained runs and aggregated data.
   */
  groupParentInfo?: RunGroupParentInfo;

  models: RunRowModelsInfo | null;
  version?: RunRowVersionInfo;
  datasets: RunDatasetWithTags[];

  [k: string]: any;
}

/**
 * Represents information about run version used. Used in row data model.
 */
export interface RunRowVersionInfo {
  version: string;
  name: string;
  type: string;
}

/**
 * Represents information about trained models related to a experiment run. Used in row data model.
 */
export interface RunRowModelsInfo {
  // We use different data model for model info originating from the store...
  registeredModels: ModelVersionInfoEntity[];
  // ...and a different one for the data originating from tags
  loggedModels: {
    artifactPath: string;
    flavors: string[];
    utcTimeCreated: number;
  }[];
  loggedModelsV3?: LoggedModelProto[];
  experimentId: string;
  runUuid: string;
}

/**
 * Represents information about run's date, timing and hierarchy. Used in row data model.
 */
export interface RunRowDateAndNestInfo {
  startTime: number;
  referenceTime: Date;
  experimentId: string;
  runUuid: string;
  runStatus: string;
  isParent: boolean;
  hasExpander: boolean;
  belongsToGroup: boolean;
  expanderOpen?: boolean;
  childrenIds?: string[];
  level: number;
}

export enum RunGroupingMode {
  Dataset = 'dataset',
  Tag = 'tag',
  Param = 'param',
}

export enum RunGroupingAggregateFunction {
  Min = 'min',
  Average = 'average',
  Max = 'max',
}
export type RunGroupByValueType =
  | string
  | {
      name: string;
      digest: string;
    };

export type RunGroupByGroupingValue = {
  mode: RunGroupingMode;
  groupByData: any;
  value: RunGroupByValueType | null;
};

export interface RunGroupParentInfo {
  isRemainingRunsGroup: boolean;
  groupingValues: RunGroupByGroupingValue[];
  groupId: string;
  expanderOpen?: boolean;
  allRunsHidden?: boolean;
  // All run UUIDs in the group
  runUuids: string[];
  // Run UUIDs in the group selected to be included in the value aggregation
  runUuidsForAggregation?: string[];
  aggregatedMetricData: Record<string, { key: string; value: number; maxStep: number }>;
  aggregatedParamData: Record<string, { key: string; value: number }>;
  aggregateFunction?: RunGroupingAggregateFunction;
}

/**
 * An intermediate interface representing single row in agGrid (but not necessarily
 * a single run - these might be nested and not expanded). Is created from the data
 * originating from the store, then after enriching with metrics, params, attributed etc.
 * is being transformed to RunRowType which serves as a final agGrid compatible type.
 */
export interface RowRenderMetadata {
  index: number;
  isParent?: boolean;
  hasExpander?: boolean;
  expanderOpen?: boolean;
  belongsToGroup?: boolean;
  isPinnable?: boolean;
  runInfo: RunInfoEntity;
  level: number;
  childrenIds?: string[];
  params: KeyValueEntity[];
  metrics: MetricEntity[];
  tags: Record<string, KeyValueEntity>;
  datasets: RunDatasetWithTags[];
  isGroup?: false;
  rowUuid: string;
  hidden?: boolean;
  visibilityControl?: RunRowVisibilityControl;
}

export interface RowGroupRenderMetadata {
  groupId: string;
  isGroup: true;
  expanderOpen: boolean;
  // All run UUIDs in the group
  runUuids: string[];
  // Run UUIDs in the group selected to be included in the value aggregation
  runUuidsForAggregation?: string[];
  aggregatedMetricEntities: {
    key: string;
    value: number;
    maxStep: number;
  }[];
  aggregatedParamEntities: {
    key: string;
    value: number;
  }[];
  aggregateFunction: RunGroupingAggregateFunction;
  groupingValues: RunGroupByGroupingValue[];
  isRemainingRunsGroup: boolean;
  hidden?: boolean;
  allRunsHidden?: boolean;
  visibilityControl?: RunRowVisibilityControl;
}
