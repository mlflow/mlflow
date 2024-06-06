import { keyBy } from 'lodash';
import { LLM_JUDGE_CORRECTNESS_RATING_PERCENTAGE_METRIC } from '../../../constants';

/**
 * Defines the behavior of a custom metric in the experiment page.
 */
interface CustomMetricBehaviorDef {
  /**
   * Display name of the metric. Used in column header, chart title, sort by and column selector.
   */
  displayName: string;
  /**
   * Formatter for the metric value. Used in table and chart tooltips.
   */
  valueFormatter: ({ value }: { value: number | string }) => string;
  /**
   * Initial width of the column in the table in pixels.
   */
  initialColumnWidth?: number;
  /**
   * Format of the axis tick labels in the chart.
   */
  chartAxisTickFormat?: string;
}

/**
 * Custom metric behavior definitions.
 */
export const customMetricBehaviorDefs: Record<string, CustomMetricBehaviorDef> = {
  // Customization for `response/llm_judged/correctness/rating/percentage` metric
  [LLM_JUDGE_CORRECTNESS_RATING_PERCENTAGE_METRIC]: {
    displayName: 'Overall assessment: Correct',
    chartAxisTickFormat: '.2%',
    initialColumnWidth: 250,
    valueFormatter: ({ value }) => {
      if (typeof value === 'number') {
        return `${(value * 100).toFixed(2)}%`;
      }
      return value;
    },
  },
};
