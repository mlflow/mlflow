import { ATTRIBUTE_COLUMN_LABELS, COLUMN_TYPES } from '../../../constants';
import type { RunsChartsLineCardConfig, SerializedRunsChartsCardConfigCard } from '../../runs-charts/runs-charts.types';
import { makeCanonicalSortKey } from '../utils/experimentPage.common-utils';
import type { ChartSectionConfig } from '@mlflow/mlflow/src/experiment-tracking/types';
import type { RunsGroupByConfig } from '../utils/experimentPage.group-row-utils';
import { RunsChartsLineChartXAxisType } from '../../runs-charts/components/RunsCharts.common';

export const EXPERIMENT_PAGE_UI_STATE_FIELDS = [
  'selectedColumns',
  'columnOrder',
  'columnWidths',
  'runsExpanded',
  'runsPinned',
  'runsHidden',
  'runsVisibilityMap',
  'runsHiddenMode',
  'compareRunCharts',
  'compareRunSections',
  'viewMaximized',
  'runListHidden',
  'isAccordionReordered',
  'groupBy',
  'groupsExpanded',
  'autoRefreshEnabled',
  'useGroupedValuesInCharts',
  'hideEmptyCharts',
  'globalLineChartConfig',
  'chartsSearchFilter',
];

/**
 * UI-state fields excluded from a shared link: per-run state keyed by run UUIDs that won't exist
 * for the recipient, plus personal/ephemeral preferences. The writer omits these when serializing
 * and the reader drops them again on apply, so the filter is symmetric and a hand-crafted link (or
 * a legacy tag predating this filter) can't smuggle them back into the recipient's view.
 */
export const NON_SHAREABLE_UI_STATE_FIELDS = [
  'runsExpanded',
  'runsPinned',
  'runsHidden',
  'runsVisibilityMap',
  'autoRefreshEnabled',
] as const;

export const getDefaultSelectedColumns = () => {
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
  HIDE_FINISHED_RUNS = 'HIDE_FINISHED_RUNS',
  CUSTOM = 'CUSTOM',
}

export type RunsChartsGlobalLineChartConfig = Partial<
  Pick<RunsChartsLineCardConfig, 'selectedXAxisMetricKey' | 'xAxisKey' | 'lineSmoothness'>
>;

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

  /**
   * Determines if the auto refresh of the chart data is enabled
   */
  autoRefreshEnabled: boolean;

  /**
   * Global line chart settings that are applied to all line charts
   */
  globalLineChartConfig?: RunsChartsGlobalLineChartConfig;

  /**
   * Regex string used to filter visible charts
   */
  chartsSearchFilter?: string;
}

/**
 * Defines model representing experiment page UI state.
 */
export interface ExperimentPageUIState extends ExperimentRunsChartsUIConfiguration {
  /**
   * Currently selected visible columns
   */
  selectedColumns: string[];

  /**
   * Full display order of column ids (canonical sort keys).
   * If empty, grid keeps its natural columnDefs order.
   */
  columnOrder: string[];

  /**
   * Map of column id (canonical sort key) to persisted pixel width.
   */
  columnWidths: Record<string, number>;

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
   * @deprecated Use "runsVisibilityMap" field instead which has better control over visibility
   */
  runsHidden: string[];

  /**
   * Determines default visibility mode for runs which are not explicitly specified by "runsVisibilityMap" field
   */
  runsHiddenMode: RUNS_VISIBILITY_MODE;

  /**
   * Object mapping run UUIDs (strings) to booleans, where a boolean value of true indicates that
   * a run has been hidden (its child runs are not visible).
   */
  runsVisibilityMap?: Record<string, boolean>;

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
  groupBy: string | RunsGroupByConfig | null;

  /**
   * Determines if the grouped and aggregated values should be displayed in charts
   */
  useGroupedValuesInCharts?: boolean;

  /**
   * Map of the currently expanded run groups
   */
  groupsExpanded: Record<string, boolean>;

  /**
   * Determines if charts with no corresponding data should be hidden
   */
  hideEmptyCharts?: boolean;
}

/**
 * Creates a new instance of experiment page UI state.
 */
export const createExperimentPageUIState = (): ExperimentPageUIState => ({
  selectedColumns: getDefaultSelectedColumns(),
  columnOrder: [],
  columnWidths: {},
  runsExpanded: {},
  runsPinned: [],
  runsHidden: [],
  runsVisibilityMap: {},
  runsHiddenMode: RUNS_VISIBILITY_MODE.FIRST_10_RUNS,
  compareRunCharts: undefined,
  compareRunSections: undefined,
  viewMaximized: false,
  runListHidden: false,
  isAccordionReordered: false,
  useGroupedValuesInCharts: true,
  hideEmptyCharts: true,
  groupBy: null,
  groupsExpanded: {},
  // Auto-refresh is enabled by default
  autoRefreshEnabled: true,
  globalLineChartConfig: {
    xAxisKey: RunsChartsLineChartXAxisType.STEP,
    lineSmoothness: 0,
    selectedXAxisMetricKey: '',
  },
});
