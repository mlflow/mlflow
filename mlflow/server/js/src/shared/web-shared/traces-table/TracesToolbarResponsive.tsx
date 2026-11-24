/**
 * Responsive helpers that collapse the traces-table toolbar's icon+text controls to icon-only when
 * it runs short of width (ML-68743/68769). Keyed off a container query on the toolbar itself, not
 * the viewport: the toolbar can be narrower than the window (a side panel eats into it), where a
 * viewport breakpoint leaves the trailing controls clipped before the labels collapse. Collapsing
 * controls must keep an aria-label. Threshold sits just above the ~1018px full-label width.
 */
export const TRACES_TOOLBAR_COLLAPSE_MAX_WIDTH = 1100;

export const TRACES_TOOLBAR_CONTAINER_NAME = 'traces-toolbar';

export const TRACES_TOOLBAR_COLLAPSE_QUERY = `@container ${TRACES_TOOLBAR_CONTAINER_NAME} (max-width: ${TRACES_TOOLBAR_COLLAPSE_MAX_WIDTH}px)`;

export interface ToolbarCollapsibleLabelProps {
  children: React.ReactNode;
}

/** Hides a toolbar button's text (leaving its icon) when the toolbar is narrow; button keeps its aria-label. */
export const ToolbarCollapsibleLabel = ({ children }: ToolbarCollapsibleLabelProps): JSX.Element => (
  <span css={{ [TRACES_TOOLBAR_COLLAPSE_QUERY]: { display: 'none' } }}>{children}</span>
);
