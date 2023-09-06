/**
 * Defines non-persistable, local state that
 * controls visibility of various UI elements in the
 * runs table and controls
 */
export class SearchExperimentRunsViewState {
  /**
   * Currently selected runs
   */
  runsSelected: Record<string, boolean> = {};

  /**
   * Currently hidden, selected child runs
   */
  hiddenChildRunsSelected: Record<string, boolean> = {};

  /**
   * Display "select columns" dropdown
   */
  columnSelectorVisible = false;

  /**
   * Display "select columns" dropdown
   */
  previewPaneVisible = false;

  /**
   * Determines if the experiment table is maximized
   */
  viewMaximized = false;

  /**
   * Persists controls state in the evaluation artifact compare mode
   */
  artifactViewState: {
    selectedTables?: string[];
    groupByCols?: string[];
    outputColumn?: string;
    intersectingOnly?: boolean;
  } = {
    selectedTables: [],
    groupByCols: [],
    outputColumn: '',
    intersectingOnly: false,
  };
}
