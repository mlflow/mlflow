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
}
