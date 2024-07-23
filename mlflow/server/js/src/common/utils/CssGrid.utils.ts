const DEFAULT_MIN_COLUMNS = 1;
const DEFAULT_MAX_COLUMNS = 3;
const DEFAULT_MIN_COLUMN_WIDTH = 330;
const DEFAULT_GAP = 16;

/**
 * Creates a CSS grid column setup with min/max number of columns.
 * See: https://stackoverflow.com/a/69154193
 */
export const getGridColumnSetup = ({
  minColumns = DEFAULT_MIN_COLUMNS,
  maxColumns = DEFAULT_MAX_COLUMNS,
  minColumnWidth = DEFAULT_MIN_COLUMN_WIDTH,
  gap = DEFAULT_GAP,
  additionalBreakpoints = [],
}: {
  /**
   * Minimum number of columns to display
   */
  minColumns?: number;
  /**
   * Maximum number of columns to display
   */
  maxColumns?: number;
  /**
   * Minimum column width
   */
  minColumnWidth?: number;
  /**
   * Gap between columns, in pixels
   */
  gap?: number;
  /**
   * Additional breakpoints to add to the grid. Defines the min breakpoint width and the minimum column width for that breakpoint.
   * Does not use minimum and maximum use of columns.
   */
  additionalBreakpoints?: { breakpointWidth: number; minColumnWidthForBreakpoint?: number }[];
}) => ({
  display: 'grid',
  gridTemplateColumns: `repeat(
      auto-fit,
      minmax(
        min(
          calc(
            100%/${minColumns} - ${gap}px
          ),
          max(
            ${minColumnWidth}px,
            calc(100%/${maxColumns} - ${gap}px)
          )
        ),
        1fr
      )
    )`,
  gap,
  ...additionalBreakpoints.reduce(
    (acc, { breakpointWidth, minColumnWidthForBreakpoint }) => ({
      ...acc,
      [`@media (min-width: ${breakpointWidth}px)`]: {
        gridTemplateColumns: `repeat(
        auto-fit,
        minmax(
          ${minColumnWidthForBreakpoint}px,
          1fr
        )
      )`,
      },
    }),
    {},
  ),
});
