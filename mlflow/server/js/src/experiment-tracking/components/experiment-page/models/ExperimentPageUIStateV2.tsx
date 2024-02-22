import { ATTRIBUTE_COLUMN_LABELS, COLUMN_TYPES } from '../../../constants';
import { SerializedRunsCompareCardConfigCard } from '../../runs-compare/runs-compare.types';
import { makeCanonicalSortKey } from '../utils/experimentPage.common-utils';
import { shouldEnableExperimentDatasetTracking } from '../../../../common/utils/FeatureUtils';
import { ChartSectionConfig } from 'experiment-tracking/types';

export const EXPERIMENT_PAGE_UI_STATE_FIELDS = [
  'selectedColumns',
  'runsExpanded',
  'runsPinned',
  'runsHidden',
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
  ];

  if (shouldEnableExperimentDatasetTracking()) {
    result.push(makeCanonicalSortKey(COLUMN_TYPES.ATTRIBUTES, ATTRIBUTE_COLUMN_LABELS.DATASET));
  }

  return result;
};

/**
 * Defines model representing experiment page UI state.
 */
export interface ExperimentPageUIStateV2 {
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

  /**
   * Currently configured charts for comparing runs, if any.
   */
  compareRunCharts?: SerializedRunsCompareCardConfigCard[];

  /**
   * Currently configured sections for grouping charts across runs
   */
  compareRunSections?: ChartSectionConfig[];

  /**
   * Determines if the sections have been reordered
   */
  isAccordionReordered: boolean;

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
  compareRunCharts: undefined,
  compareRunSections: undefined,
  viewMaximized: false,
  runListHidden: false,
  isAccordionReordered: false,
  groupBy: '',
  groupsExpanded: {},
});
