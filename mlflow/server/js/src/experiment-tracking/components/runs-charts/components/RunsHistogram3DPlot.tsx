import React, { useMemo } from 'react';
import type { Layout, Data, Config } from 'plotly.js';
import { LazyPlot } from '../../LazyPlot';
import { useDesignSystemTheme } from '@databricks/design-system';
import { commonRunsChartStyles, createThemedPlotlyLayout } from './RunsCharts.common';
import { RunsChartCardLoadingPlaceholder } from './cards/ChartCard.common';

export interface HistogramData {
  name: string;
  step: number;
  timestamp: number;
  bin_edges: number[];
  counts: number[];
  min_value?: number;
  max_value?: number;
}

export interface RunsHistogram3DPlotProps {
  histograms: HistogramData[];
  logScale?: boolean;
  className?: string;
}

const PLOT_CONFIG: Partial<Config> = {
  displayModeBar: true,
  displaylogo: false,
  scrollZoom: true,
  modeBarButtonsToRemove: ['sendDataToCloud', 'autoScale2d', 'toImage'],
};

/**
 * Converts histogram data into a 3D surface plot format
 */
const convertHistogramsTo3DSurface = (histograms: HistogramData[], logScale: boolean = false) => {
  if (histograms.length === 0) {
    return { x: [], y: [], z: [] };
  }

  const sortedHistograms = [...histograms].sort((a, b) => a.step - b.step);

  const steps = sortedHistograms.map((h) => h.step);

  const firstHist = sortedHistograms[0];
  const binCenters = firstHist.bin_edges.slice(0, -1).map((edge, i) => {
    return (edge + firstHist.bin_edges[i + 1]) / 2;
  });

  const zMatrix = sortedHistograms.map((hist) => {
    return hist.counts.map((count) => (logScale && count > 0 ? Math.log10(count + 1) : count));
  });

  return {
    x: binCenters,
    y: steps,
    z: zMatrix,
  };
};

/**
 * 3D Histogram Plot Component
 *
 * Visualizes how histogram distributions evolve over training steps.
 * Uses a 3D surface plot where:
 * - X axis: Bin values (e.g., weight values)
 * - Y axis: Training step
 * - Z axis: Count/frequency
 */
export const RunsHistogram3DPlot: React.FC<RunsHistogram3DPlotProps> = ({
  histograms,
  logScale = false,
  className,
}) => {
  const { theme } = useDesignSystemTheme();

  const plotData = useMemo(() => {
    const { x, y, z } = convertHistogramsTo3DSurface(histograms, logScale);

    if (x.length === 0 || y.length === 0 || z.length === 0) {
      return [];
    }

    const trace: Partial<Data> = {
      type: 'surface',
      x,
      y,
      z,
      colorscale: 'Viridis',
      showscale: true,
      colorbar: {
        title: logScale ? 'log₁₀(Count + 1)' : 'Count',
        titleside: 'right',
        len: 0.6,
        thickness: 15,
      },
      hovertemplate: 'Value: %{x:.3f}<br>Step: %{y}<br>Count: %{z:.0f}<extra></extra>',
    };

    return [trace];
  }, [histograms, logScale]);

  const layout = useMemo<Partial<Layout>>(() => {
    const baseLayout = createThemedPlotlyLayout(theme);

    return {
      ...baseLayout,
      autosize: true,
      margin: { l: 0, r: 0, t: 0, b: 0, pad: 0 },
      scene: {
        xaxis: {
          title: { text: 'Value', font: { size: 11 } },
          gridcolor: theme.colors.borderDecorative,
          showbackground: true,
          backgroundcolor: theme.colors.backgroundPrimary,
        },
        yaxis: {
          title: { text: 'Step', font: { size: 11 } },
          gridcolor: theme.colors.borderDecorative,
          showbackground: true,
          backgroundcolor: theme.colors.backgroundPrimary,
        },
        zaxis: {
          title: { text: logScale ? 'log₁₀(Count)' : 'Count', font: { size: 11 } },
          gridcolor: theme.colors.borderDecorative,
          showbackground: true,
          backgroundcolor: theme.colors.backgroundPrimary,
        },
        camera: {
          eye: { x: 1.8, y: 1.8, z: 1.2 },
        },
        aspectmode: 'cube',
      },
    };
  }, [logScale, theme]);

  if (histograms.length === 0) {
    return null;
  }

  return (
    <div css={commonRunsChartStyles.chartWrapper(theme)} className={className}>
      <LazyPlot
        data={plotData}
        layout={layout}
        config={PLOT_CONFIG}
        useResizeHandler
        css={commonRunsChartStyles.chart(theme)}
        fallback={<RunsChartCardLoadingPlaceholder />}
      />
    </div>
  );
};

export default RunsHistogram3DPlot;
