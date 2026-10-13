import { useCallback, useRef, useState } from 'react';
import type { Config, Data, Layout } from 'plotly.js';
import Utils from '../../../../common/utils/Utils';

export type ExperimentChartImageDownloadFileFormat = 'svg' | 'png';
export type ExperimentChartImageDownloadHandler = (
  format: ExperimentChartImageDownloadFileFormat,
  chartTitle: string,
) => void | Promise<void>;

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

let plotlyFactoryPromise: Promise<typeof import('../../PlotlyFactory')> | undefined;

const getPlotly = async () => {
  plotlyFactoryPromise ??= import('../../PlotlyFactory').catch((error) => {
    plotlyFactoryPromise = undefined;
    throw error;
  });
  return (await plotlyFactoryPromise).Plotly;
};

const getChartImageDownloadErrorMessage = (error: unknown) => {
  if (error instanceof Error) {
    return error.message;
  }

  if (typeof error === 'string') {
    return error;
  }

  return 'Unknown error';
};

export const createChartImageDownloadHandler =
  (data: Data[], layout: Partial<Layout>) => async (format: 'svg' | 'png', title: string) => {
    try {
      const Plotly = await getPlotly();
      await Plotly.downloadImage(
        {
          data,
          layout: { ...layout, ...experimentChartImageDefaultDownloadLayout },
          config: experimentChartImageDefaultDownloadPlotConfig,
        },
        { ...experimentChartImageDefaultDownloadSettings, format, filename: title },
      );
    } catch (error) {
      Utils.displayGlobalErrorNotification(
        `Failed to export chart image. Error: ${getChartImageDownloadErrorMessage(error)}`,
      );
    }
  };

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
