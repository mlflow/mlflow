import { useCallback, useRef, useState } from 'react';
import { type Data, type Layout, type Config, downloadImage } from 'plotly.js';

export type ExperimentChartImageDownloadFileFormat = 'svg' | 'png';
export type ExperimentChartImageDownloadHandler = (
  format: ExperimentChartImageDownloadFileFormat,
  chartTitle: string,
) => void;

const experimentChartImageDefaultDownloadLayout: Partial<Layout> = {
  paper_bgcolor: 'white',
  plot_bgcolor: 'white',
};

const experimentChartImageDefaultDownloadSettings = {
  width: 1200,
  height: 600,
};

const experimentChartImageDefaultDownloadPlotConfig: Partial<Config> = {
  displaylogo: false,
  modeBarButtonsToRemove: ['toImage'],
};

export const createChartImageDownloadHandler =
  (data: Data[], layout: Partial<Layout>) => (format: 'svg' | 'png', title: string) =>
    downloadImage(
      {
        data,
        layout: { ...layout, ...experimentChartImageDefaultDownloadLayout },
        config: experimentChartImageDefaultDownloadPlotConfig,
      },
      { ...experimentChartImageDefaultDownloadSettings, format, filename: title },
    );

/**
 * Returns a memoized download handler for chart images.
 * Uses ref-based caching to ensure that the download handler is not recreated on every render.
 */
export const useChartImageDownloadHandler = () => {
  const downloadHandlerRef = useRef<ExperimentChartImageDownloadHandler | null>(null);
  const [downloadHandler, setDownloadHandler] = useState<ExperimentChartImageDownloadHandler | null>(null);

  const setDownloadHandlerCached = useCallback((downloadHandler: ExperimentChartImageDownloadHandler) => {
    downloadHandlerRef.current = downloadHandler;
    setDownloadHandler((existingHandler: ExperimentChartImageDownloadHandler | null) => {
      if (existingHandler) {
        return existingHandler;
      }

      return (format: ExperimentChartImageDownloadFileFormat, chartTitle: string) =>
        downloadHandlerRef.current?.(format, chartTitle);
    });
  }, []);

  return [downloadHandler, setDownloadHandlerCached] as const;
};
