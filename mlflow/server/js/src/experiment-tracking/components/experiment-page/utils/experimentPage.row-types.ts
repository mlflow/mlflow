import type {
  ModelVersionInfoEntity,
  RunInfoEntity,
  RunDatasetWithTags,
  KeyValueEntity,
} from '../../../types';

/**
 * Represents a single ag-grid compatible row used in Experiment View runs table.
 */
export interface RunRowType {
  runUuid: string;
  runInfo: RunInfoEntity;
  experimentName: { name: string; basename: string };
  experimentId: string;
  duration: string | null;
  user: string;
  pinned: boolean;
  hidden: boolean;
  pinnable: boolean;
  runName: string;
  color?: string;
  tags: Record<string, { key: string; value: string }>;
  params: KeyValueEntity[];
  runDateAndNestInfo: RunRowDateAndNestInfo;
  models: RunRowModelsInfo;
  version: RunRowVersionInfo;
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
  expanderOpen?: boolean;
  childrenIds?: string[];
  level: number;
}
