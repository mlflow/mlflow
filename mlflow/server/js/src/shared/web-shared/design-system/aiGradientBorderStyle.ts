import type { ThemeType } from '@databricks/design-system';

/**
 * Border treatment for AI-powered affordances: keeps the page surface as the fill and paints the
 * AI gradient onto the 1px border. Shared by the Detect Issues button and the assistant entry
 * points so they read as one visual family and stay in sync with the design-system gradient.
 */
export const getAiGradientBorderStyle = (theme: ThemeType) => ({
  border: '1px solid transparent !important',
  background: `linear-gradient(${theme.colors.backgroundPrimary}, ${theme.colors.backgroundPrimary}) padding-box, ${theme.gradients.aiBorderGradient} border-box`,
});
