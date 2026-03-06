import { useDesignSystemTheme } from '@databricks/design-system';
import type { RunsChartsHistogramCardConfig } from '../../runs-charts.types';

/**
 * Preview for histogram chart configuration.
 * Shows a placeholder since we can't fetch artifact data during configuration.
 */
export const RunsChartsConfigureHistogramChartPreview = ({
  cardConfig,
}: {
  previewData: unknown[];
  cardConfig: RunsChartsHistogramCardConfig;
}) => {
  const { theme } = useDesignSystemTheme();
  const selectedHistogramKey = cardConfig.histogramKeys?.[0];

  return (
    <div
      css={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        color: theme.colors.textSecondary,
        padding: theme.spacing.lg,
        textAlign: 'center',
      }}
    >
      {selectedHistogramKey ? (
        <>
          <div css={{ marginBottom: theme.spacing.sm }}>
            <strong>Selected histogram:</strong>
          </div>
          <div
            css={{
              fontFamily: 'monospace',
              backgroundColor: theme.colors.tagDefault,
              padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
              borderRadius: theme.general.borderRadiusBase,
              marginBottom: theme.spacing.lg,
            }}
          >
            {selectedHistogramKey}
          </div>
          <div css={{ fontSize: theme.typography.fontSizeSm, opacity: 0.8 }}>
            3D visualization will appear after saving
          </div>
        </>
      ) : (
        <div css={{ opacity: 0.7 }}>Select a histogram from the dropdown</div>
      )}
    </div>
  );
};
