import { ATTRIBUTE_COLUMN_LABELS, COLUMN_TYPES } from '../../../constants';
import { SerializedRunsChartsCardConfigCard } from '../../runs-charts/runs-charts.types';
import { makeCanonicalSortKey } from '../utils/experimentPage.common-utils';
import { ChartSectionConfig } from 'experiment-tracking/types';

export const EXPERIMENT_PAGE_UI_STATE_FIELDS = [
  'selectedColumns',
  'runsExpanded',
  'runsPinned',
  'runsHidden',
  'runsHiddenMode',
  'compareRunCharts',
  'compareRunSections',
  'viewMaximized',
  'runListHidden',
  'isAccordionReordered',
  'groupBy',
  'groupsExpanded',
];

const getDefaultSelectedColumns = () => {
  const result = [
    // "Source" and "Model" columns are visible by default
    makeCanonicalSortKey(COLUMN_TYPES.ATTRIBUTES, ATTRIBUTE_COLUMN_LABELS.SOURCE),
    makeCanonicalSortKey(COLUMN_TYPES.ATTRIBUTES, ATTRIBUTE_COLUMN_LABELS.MODELS),
    makeCanonicalSortKey(COLUMN_TYPES.ATTRIBUTES, ATTRIBUTE_COLUMN_LABELS.DATASET),
  ];

  return result;
};

export enum RUNS_VISIBILITY_MODE {
  SHOWALL = 'SHOW_ALL',
  HIDEALL = 'HIDE_ALL',
  FIRST_10_RUNS = 'FIRST_10_RUNS',
  FIRST_20_RUNS = 'FIRST_20_RUNS',
  CUSTOM = 'CUSTOM',
}

/**
 * An interface describing serializable, persistable configuration for charts displaying
 * experiment run data: metrics, parameters etc. Used in experiment page's runs compare view and
 * run page's charts view.
 */
export interface ExperimentRunsChartsUIConfiguration {
  /**
   * Currently configured charts for comparing runs, if any.
   */
  compareRunCharts?: SerializedRunsChartsCardConfigCard[];

  /**
   * Currently configured sections for grouping charts across runs
   */
  compareRunSections?: ChartSectionConfig[];
  /**
   * Determines if the sections have been reordered
   */
  isAccordionReordered: boolean;
}

/**
 * Defines model representing experiment page UI state.
 */
export interface ExperimentPageUIStateV2 extends ExperimentRunsChartsUIConfiguration {
  /**
   * Currently selected visible columns
   */
  selectedColumns: string[];

  /**
   * Object mapping run UUIDs (strings) to booleans, where a boolean value of true indicates that
   * a run has been expanded (its child runs are visible).
   */
  runsExpanded: Record<string, boolean>;

  /**
   * List of pinned row UUIDs
   */
  runsPinned: string[];

  /**
   * List of hidden row UUIDs
   */
  runsHidden: string[];

  runsHiddenMode: RUNS_VISIBILITY_MODE;

  /**
   * Determines if the experiment view is maximized
   */
  viewMaximized: boolean;

  /**
   * Determines if the run list is hidden
   */
  runListHidden: boolean;

  /**
   * Current group by key - contains mode (tag, param, dataset), value and aggregation function
   */
  groupBy: string;

  /**
   * Map of the currently expanded run groups
   */
  groupsExpanded: Record<string, boolean>;
}

/**
 * Creates a new instance of experiment page UI state.
 */
export const createExperimentPageUIStateV2 = (): ExperimentPageUIStateV2 => ({
  selectedColumns: getDefaultSelectedColumns(),
  runsExpanded: {},
  runsPinned: [],
  runsHidden: [],
  runsHiddenMode: RUNS_VISIBILITY_MODE.FIRST_10_RUNS,
  compareRunCharts: undefined,
  compareRunSections: undefined,
  viewMaximized: false,
  runListHidden: false,
  isAccordionReordered: false,
  groupBy: '',
  groupsExpanded: {},
});
