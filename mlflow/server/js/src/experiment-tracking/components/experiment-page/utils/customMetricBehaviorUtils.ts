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
  // empty
};
