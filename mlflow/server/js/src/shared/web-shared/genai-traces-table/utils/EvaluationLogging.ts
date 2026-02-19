/** Constants for logging. */

// Pages
export const RUN_EVALUATION_RESULTS_TAB_SINGLE_RUN = 'mlflow.evaluations_review.run_evaluation_results_single_run_page';
export const RUN_EVALUATION_RESULTS_TAB_COMPARE_RUNS =
  'mlflow.evaluations_review.run_evaluation_results_compare_runs_page';

// When a user opens a single item in the evaluations table, the page ID for the evaluations review page modal.
export const RUN_EVALUATIONS_SINGLE_ITEM_REVIEW_UI_PAGE_ID = 'mlflow.evaluations_review.review_ui';

// Views
// Counts the number of times the expanded assessment details is clicked, showing how many times users view rationales.
export const EXPANDED_ASSESSMENT_DETAILS_VIEW: Record<string, string> = {
  // Important note: Overall is always expanded.
  overall: 'mlflow.evaluations_review.expanded_overall_assessment_details_view',
  response: 'mlflow.evaluations_review.expanded_response_assessment_details_view',
  retrieval: 'mlflow.evaluations_review.expanded_retrieval_assessment_details_view',
};

export const ASSESSMENT_RATIONAL_HOVER_DETAILS_VIEW =
  'mlflow.evaluations_review.assessment_rationale_hover_details_view';

// Buttons
// The component ID for the compare-to-run dropdown in the evaluations table.
export const COMPARE_TO_RUN_DROPDOWN_COMPONENT_ID = 'mlflow.evaluations_review.table_ui.compare_to_run_combobox';

// The component ID for the filter dropdown in the evaluations table.
export const FILTER_DROPDOWN_COMPONENT_ID = 'mlflow.evaluations_review.table_ui.filter_form';

// The component ID for the column selector dropdown in the evaluations table.
export const COLUMN_SELECTOR_DROPDOWN_COMPONENT_ID = 'mlflow.evaluations_review.table_ui.column_filter_combobox';
